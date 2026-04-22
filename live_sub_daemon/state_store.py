from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import CueRecord, SourceCue

READY_STATUSES = ("TRANSLATED", "FALLBACK_READY")


class StateStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA wal_autocheckpoint=1000;
                CREATE TABLE IF NOT EXISTS cues (
                    source_key TEXT PRIMARY KEY,
                    source_kind TEXT NOT NULL,
                    srt_index INTEGER,
                    start_ms INTEGER,
                    end_ms INTEGER,
                    jp_raw TEXT NOT NULL,
                    jp_aggregated TEXT,
                    jp_canonicalized TEXT,
                    jp_corrected TEXT,
                    zh_text TEXT,
                    status TEXT NOT NULL,
                    t_seen_mono_ms INTEGER NOT NULL,
                    due_mono_ms INTEGER NOT NULL,
                    translated_mono_ms INTEGER,
                    dropped_late INTEGER NOT NULL DEFAULT 0,
                    llm_latency_ms INTEGER,
                    last_error TEXT,
                    inflight_owner TEXT,
                    inflight_since_mono_ms INTEGER,
                    context_miss INTEGER NOT NULL DEFAULT 0,
                    stage1_provider TEXT,
                    stage1_model TEXT,
                    stage2_provider TEXT,
                    stage2_model TEXT,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    stage1_latency_ms INTEGER,
                    stage2_latency_ms INTEGER,
                    s1_skipped INTEGER NOT NULL DEFAULT 0,
                    aggregator_reason TEXT,
                    manual_zh_text TEXT,
                    manual_locked INTEGER NOT NULL DEFAULT 0,
                    deleted_soft INTEGER NOT NULL DEFAULT 0,
                    display_suppressed INTEGER NOT NULL DEFAULT 0,
                    join_target_source_key TEXT,
                    displayed_at_mono_ms INTEGER,
                    updated_by TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cues_status_due ON cues(status, due_mono_ms);
                CREATE INDEX IF NOT EXISTS idx_cues_status_seen ON cues(status, t_seen_mono_ms);
                CREATE INDEX IF NOT EXISTS idx_cues_srt_idx ON cues(srt_index);

                CREATE TABLE IF NOT EXISTS metrics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stage TEXT NOT NULL,
                    batch_size INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    ok INTEGER NOT NULL,
                    timed_out INTEGER NOT NULL DEFAULT 0,
                    cached_tokens INTEGER,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._ensure_cues_columns(cur)
            self._recover_inflight_rows(cur, reason="recovered_on_startup")
            self._conn.commit()

    @staticmethod
    def _ensure_cues_columns(cur: sqlite3.Cursor) -> None:
        rows = cur.execute("PRAGMA table_info(cues)").fetchall()
        existing = {str(row[1]) for row in rows}
        add_specs = {
            "jp_aggregated": "TEXT",
            "jp_canonicalized": "TEXT",
            "inflight_owner": "TEXT",
            "inflight_since_mono_ms": "INTEGER",
            "context_miss": "INTEGER NOT NULL DEFAULT 0",
            "last_error": "TEXT",
            "stage1_provider": "TEXT",
            "stage1_model": "TEXT",
            "stage2_provider": "TEXT",
            "stage2_model": "TEXT",
            "fallback_used": "INTEGER NOT NULL DEFAULT 0",
            "stage1_latency_ms": "INTEGER",
            "stage2_latency_ms": "INTEGER",
            "s1_skipped": "INTEGER NOT NULL DEFAULT 0",
            "aggregator_reason": "TEXT",
            "manual_zh_text": "TEXT",
            "manual_locked": "INTEGER NOT NULL DEFAULT 0",
            "deleted_soft": "INTEGER NOT NULL DEFAULT 0",
            "display_suppressed": "INTEGER NOT NULL DEFAULT 0",
            "join_target_source_key": "TEXT",
            "displayed_at_mono_ms": "INTEGER",
            "updated_by": "TEXT",
            "revision": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, spec in add_specs.items():
            if name not in existing:
                cur.execute(f"ALTER TABLE cues ADD COLUMN {name} {spec}")

    @staticmethod
    def _recover_inflight_rows(cur: sqlite3.Cursor, reason: str) -> int:
        now = time.time()
        cur.execute(
            """
            UPDATE cues
            SET status='NEW',
                inflight_owner=NULL,
                inflight_since_mono_ms=NULL,
                last_error=COALESCE(last_error, ?),
                updated_at=?
            WHERE status='INFLIGHT'
            """,
            (reason, now),
        )
        return int(cur.rowcount or 0)

    def run_wal_checkpoint(self, mode: str = "PASSIVE") -> Dict[str, int]:
        mode_norm = str(mode).strip().upper()
        if mode_norm not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
            mode_norm = "PASSIVE"
        with self._lock:
            row = self._conn.execute(f"PRAGMA wal_checkpoint({mode_norm})").fetchone()
        if row is None:
            return {"busy": 0, "log": 0, "checkpointed": 0}
        return {"busy": int(row[0]), "log": int(row[1]), "checkpointed": int(row[2])}

    def get_wal_size_bytes(self) -> int:
        wal_path = Path(f"{self.db_path}-wal")
        if not wal_path.exists():
            return 0
        try:
            return int(wal_path.stat().st_size)
        except OSError:
            return 0

    @staticmethod
    def _normalize_scope_kinds(allowed_source_kinds: Optional[Sequence[str]]) -> List[str]:
        if not allowed_source_kinds:
            return []
        out: List[str] = []
        seen: set[str] = set()
        for raw in allowed_source_kinds:
            value = str(raw or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    @classmethod
    def _scope_sql(
        cls,
        *,
        allowed_source_kinds: Optional[Sequence[str]],
        updated_after_unix: Optional[float],
        table_alias: str = "",
    ) -> tuple[str, List[Any]]:
        alias = str(table_alias or "")
        if alias and not alias.endswith("."):
            alias = f"{alias}."
        parts: List[str] = []
        params: List[Any] = []
        kinds = cls._normalize_scope_kinds(allowed_source_kinds)
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            parts.append(f"{alias}source_kind IN ({placeholders})")
            params.extend(kinds)
        if updated_after_unix is not None:
            parts.append(f"{alias}updated_at >= ?")
            params.append(float(updated_after_unix))
        if not parts:
            return "", []
        return " AND " + " AND ".join(parts), params

    def cleanup_runtime_scope(
        self,
        *,
        now_mono_ms: int,
        allowed_source_kinds: Optional[Sequence[str]],
        stale_unfinished_sec: float,
        monotonic_guard_sec: float,
        anomaly_remaining_max_sec: float,
    ) -> Dict[str, int]:
        now_unix = time.time()
        now_mono_ms = int(now_mono_ms)
        kinds = self._normalize_scope_kinds(allowed_source_kinds)
        archived_count = 0
        excluded_source_count = 0
        stale_unfinished_count = 0
        anomaly_archived_count = 0
        guard_ms = max(0, int(float(monotonic_guard_sec) * 1000.0))
        max_due_mono_ms = now_mono_ms + max(0, int(float(anomaly_remaining_max_sec) * 1000.0))
        stale_unfinished_sec = max(0.0, float(stale_unfinished_sec))

        with self._lock:
            cur = self._conn.cursor()
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                cur.execute(
                    f"""
                    UPDATE cues
                    SET deleted_soft=1,
                        displayed_at_mono_ms=COALESCE(displayed_at_mono_ms, ?),
                        updated_by='system',
                        updated_at=?
                    WHERE source_kind NOT IN ({placeholders})
                    """,
                    (now_mono_ms, now_unix, *kinds),
                )
                excluded_source_count = int(cur.rowcount or 0)
                archived_count += excluded_source_count

            # Mark stale unfinished rows as archived from active view.
            if stale_unfinished_sec > 0:
                cur.execute(
                    """
                    UPDATE cues
                    SET deleted_soft=1,
                        displayed_at_mono_ms=COALESCE(displayed_at_mono_ms, ?),
                        updated_by='system',
                        updated_at=?
                    WHERE status IN ('NEW','INFLIGHT')
                      AND updated_at < ?
                    """,
                    (now_mono_ms, now_unix, now_unix - stale_unfinished_sec),
                )
                stale_unfinished_count = int(cur.rowcount or 0)
                archived_count += stale_unfinished_count

            max_translated = cur.execute(
                "SELECT MAX(translated_mono_ms) FROM cues WHERE translated_mono_ms IS NOT NULL"
            ).fetchone()
            max_translated_mono_ms = (
                int(max_translated[0]) if max_translated and max_translated[0] is not None else None
            )
            if (
                max_translated_mono_ms is not None
                and max_translated_mono_ms > (now_mono_ms + guard_ms)
            ):
                scope_sql, scope_params = self._scope_sql(
                    allowed_source_kinds=kinds or None,
                    updated_after_unix=None,
                )
                cur.execute(
                    f"""
                    UPDATE cues
                    SET deleted_soft=1,
                        displayed_at_mono_ms=COALESCE(displayed_at_mono_ms, ?),
                        updated_by='system',
                        updated_at=?
                    WHERE displayed_at_mono_ms IS NULL
                      AND deleted_soft=0
                      AND display_suppressed=0
                      AND dropped_late=0
                      AND status IN ('TRANSLATED','FALLBACK_READY')
                      AND due_mono_ms > ?
                      {scope_sql}
                    """,
                    (now_mono_ms, now_unix, max_due_mono_ms, *scope_params),
                )
                anomaly_archived_count = int(cur.rowcount or 0)
                archived_count += anomaly_archived_count
            self._conn.commit()

        return {
            "archived_count": int(archived_count),
            "excluded_source_count": int(excluded_source_count),
            "stale_unfinished_count": int(stale_unfinished_count),
            "anomaly_archived_count": int(anomaly_archived_count),
        }

    def upsert_source_cues(self, cues: Sequence[SourceCue], now_mono_ms: int, delay_ms: int) -> int:
        if not cues:
            return 0
        inserted = 0
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            for cue in cues:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO cues(
                        source_key, source_kind, srt_index, start_ms, end_ms,
                        jp_raw, jp_aggregated, jp_canonicalized, aggregator_reason,
                        status, t_seen_mono_ms, due_mono_ms, updated_at, context_miss, updated_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NEW', ?, ?, ?, 0, 'system')
                    """,
                    (
                        cue.source_key,
                        cue.source_kind,
                        cue.srt_index,
                        cue.start_ms,
                        cue.end_ms,
                        cue.jp_raw,
                        cue.jp_aggregated if cue.jp_aggregated is not None else cue.jp_raw,
                        cue.jp_canonicalized if cue.jp_canonicalized is not None else cue.jp_raw,
                        cue.aggregator_reason,
                        now_mono_ms,
                        now_mono_ms + delay_ms,
                        now,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
            self._conn.commit()
        return inserted

    def fetch_new_batch(self, limit: int) -> List[CueRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM cues
                WHERE status = 'NEW'
                ORDER BY t_seen_mono_ms ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_cue(row) for row in rows]

    def fetch_and_claim_batch(
        self,
        limit: int,
        owner: str,
        now_mono_ms: int,
        contextless_only: bool = False,
        mark_context_miss: bool = False,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> List[CueRecord]:
        if limit <= 0:
            return []
        now = time.time()
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            cur = self._conn.cursor()
            if contextless_only:
                rows = cur.execute(
                    f"""
                    SELECT source_key FROM cues
                    WHERE status='NEW' AND context_miss=1
                    {scope_sql}
                    ORDER BY t_seen_mono_ms ASC
                    LIMIT ?
                    """,
                    (*scope_params, limit),
                ).fetchall()
            else:
                rows = cur.execute(
                    f"""
                    SELECT source_key FROM cues
                    WHERE status='NEW'
                    {scope_sql}
                    ORDER BY t_seen_mono_ms ASC
                    LIMIT ?
                    """,
                    (*scope_params, limit),
                ).fetchall()
            keys = [str(row[0]) for row in rows]
            if not keys:
                return []

            claimed_keys: List[str] = []
            for key in keys:
                cur.execute(
                    """
                    UPDATE cues
                    SET status='INFLIGHT',
                        inflight_owner=?,
                        inflight_since_mono_ms=?,
                        context_miss=CASE WHEN ?=1 THEN 1 ELSE context_miss END,
                        updated_at=?
                    WHERE source_key=? AND status='NEW'
                    """,
                    (owner, now_mono_ms, int(mark_context_miss), now, key),
                )
                if cur.rowcount > 0:
                    claimed_keys.append(key)

            if not claimed_keys:
                self._conn.commit()
                return []

            placeholders = ",".join("?" for _ in claimed_keys)
            rows = cur.execute(
                f"""
                SELECT * FROM cues
                WHERE source_key IN ({placeholders})
                ORDER BY t_seen_mono_ms ASC
                """,
                claimed_keys,
            ).fetchall()
            self._conn.commit()
        return [self._row_to_cue(row) for row in rows]

    def release_stale_inflight(
        self,
        now_mono_ms: int,
        stale_after_ms: int,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> int:
        if stale_after_ms <= 0:
            return 0
        threshold = now_mono_ms - stale_after_ms
        now = time.time()
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                UPDATE cues
                SET status='NEW',
                    inflight_owner=NULL,
                    inflight_since_mono_ms=NULL,
                    updated_at=?
                WHERE status='INFLIGHT'
                  AND inflight_since_mono_ms IS NOT NULL
                  AND inflight_since_mono_ms <= ?
                  {scope_sql}
                """,
                (now, threshold, *scope_params),
            )
            released = cur.rowcount
            self._conn.commit()
        return int(released)

    def release_claimed_batch(self, source_keys: Sequence[str], error_message: Optional[str] = None) -> int:
        keys = [str(k) for k in source_keys if k]
        if not keys:
            return 0
        now = time.time()
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                UPDATE cues
                SET status='NEW',
                    inflight_owner=NULL,
                    inflight_since_mono_ms=NULL,
                    last_error=COALESCE(?, last_error),
                    updated_at=?
                WHERE source_key IN ({placeholders})
                  AND status='INFLIGHT'
                """,
                (error_message, now, *keys),
            )
            count = cur.rowcount
            self._conn.commit()
        return int(count)

    def has_older_inflight_than(
        self,
        seen_mono_ms: int,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> bool:
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT 1 FROM cues
                WHERE status='INFLIGHT'
                  AND t_seen_mono_ms < ?
                  {scope_sql}
                LIMIT 1
                """,
                (seen_mono_ms, *scope_params),
            ).fetchone()
        return row is not None

    def mark_oldest_new_context_miss(
        self,
        limit: int,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> int:
        if limit <= 0:
            return 0
        now = time.time()
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                f"""
                SELECT source_key FROM cues
                WHERE status='NEW'
                {scope_sql}
                ORDER BY t_seen_mono_ms ASC
                LIMIT ?
                """,
                (*scope_params, limit),
            ).fetchall()
            keys = [str(row[0]) for row in rows]
            if not keys:
                return 0
            placeholders = ",".join("?" for _ in keys)
            cur.execute(
                f"""
                UPDATE cues
                SET context_miss=1, updated_at=?
                WHERE source_key IN ({placeholders})
                """,
                (now, *keys),
            )
            updated = cur.rowcount
            self._conn.commit()
        return int(updated)

    def save_pipeline_results(
        self,
        cues: Sequence[CueRecord],
        corrected_texts: dict[str, str],
        translated_texts: dict[str, str],
        translated_at_mono_ms: int,
        fallback_mode: str,
        llm_latency_ms: Optional[int],
        error_message: Optional[str] = None,
        stage1_provider: Optional[str] = None,
        stage1_model: Optional[str] = None,
        stage2_provider: Optional[str] = None,
        stage2_model: Optional[str] = None,
        fallback_used: bool = False,
        stage1_latency_ms: Optional[int] = None,
        stage2_latency_ms: Optional[int] = None,
        s1_skipped: bool = False,
        display_suppressed_map: Optional[Dict[str, bool]] = None,
        join_target_map: Optional[Dict[str, str]] = None,
    ) -> None:
        now = time.time()
        ordered = sorted(cues, key=lambda c: (c.due_mono_ms, c.t_seen_mono_ms, c.source_key))
        display_suppressed_map = display_suppressed_map or {}
        join_target_map = join_target_map or {}
        with self._lock:
            cur = self._conn.cursor()
            for cue in ordered:
                corrected = corrected_texts.get(cue.source_key, cue.jp_raw)
                translated = translated_texts.get(cue.source_key)
                display_suppressed = bool(display_suppressed_map.get(cue.source_key, False))
                join_target_source_key = join_target_map.get(cue.source_key)
                if display_suppressed:
                    if translated is None:
                        translated = ""
                    status = "TRANSLATED"
                elif translated is None:
                    translated = "" if fallback_mode == "empty" else cue.jp_raw
                    status = "FALLBACK_READY"
                else:
                    status = "TRANSLATED"

                dropped_late = int(translated_at_mono_ms > cue.due_mono_ms)
                cur.execute(
                    """
                    UPDATE cues
                    SET jp_corrected = ?,
                        zh_text = ?,
                        status = ?,
                        translated_mono_ms = ?,
                        dropped_late = ?,
                        llm_latency_ms = ?,
                        last_error = ?,
                        stage1_provider = ?,
                        stage1_model = ?,
                        stage2_provider = ?,
                        stage2_model = ?,
                        fallback_used = ?,
                        stage1_latency_ms = ?,
                        stage2_latency_ms = ?,
                        s1_skipped = ?,
                        display_suppressed = ?,
                        join_target_source_key = ?,
                        updated_by = CASE WHEN manual_locked=1 THEN updated_by ELSE 'system' END,
                        revision = revision + 1,
                        inflight_owner = NULL,
                        inflight_since_mono_ms = NULL,
                        updated_at = ?
                    WHERE source_key = ?
                    """,
                    (
                        corrected,
                        translated,
                        status,
                        translated_at_mono_ms,
                        dropped_late,
                        llm_latency_ms,
                        error_message,
                        stage1_provider,
                        stage1_model,
                        stage2_provider,
                        stage2_model,
                        int(bool(fallback_used)),
                        stage1_latency_ms,
                        stage2_latency_ms,
                        int(bool(s1_skipped)),
                        int(display_suppressed),
                        join_target_source_key,
                        now,
                        cue.source_key,
                    ),
                )
            self._conn.commit()

    def fetch_render_candidates(
        self,
        now_mono_ms: int,
        limit: int = 2,
        delay_adjust_ms: int = 0,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> List[CueRecord]:
        placeholders = ",".join("?" for _ in READY_STATUSES)
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM cues
                WHERE (due_mono_ms + ?) <= ?
                  AND dropped_late = 0
                  AND deleted_soft = 0
                  AND display_suppressed = 0
                  AND status IN ({placeholders})
                  {scope_sql}
                ORDER BY (due_mono_ms + ?) DESC
                LIMIT ?
                """,
                (int(delay_adjust_ms), now_mono_ms, *READY_STATUSES, *scope_params, int(delay_adjust_ms), limit),
            ).fetchall()
        return [self._row_to_cue(row) for row in rows]

    def fetch_due_unshown_cues(
        self,
        now_mono_ms: int,
        limit: int = 50,
        delay_adjust_ms: int = 0,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> List[CueRecord]:
        placeholders = ",".join("?" for _ in READY_STATUSES)
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM cues
                WHERE (due_mono_ms + ?) <= ?
                  AND dropped_late = 0
                  AND deleted_soft = 0
                  AND display_suppressed = 0
                  AND displayed_at_mono_ms IS NULL
                  AND status IN ({placeholders})
                  {scope_sql}
                ORDER BY (due_mono_ms + ?) ASC, t_seen_mono_ms ASC, source_key ASC
                LIMIT ?
                """,
                (int(delay_adjust_ms), now_mono_ms, *READY_STATUSES, *scope_params, int(delay_adjust_ms), int(limit)),
            ).fetchall()
        return [self._row_to_cue(row) for row in rows]

    def fetch_latest_ready_unshown(
        self,
        delay_adjust_ms: int = 0,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        placeholders = ",".join("?" for _ in READY_STATUSES)
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT source_key,
                       translated_mono_ms,
                       updated_at,
                       (due_mono_ms + ?) AS due_effective_mono_ms
                FROM cues
                WHERE displayed_at_mono_ms IS NULL
                  AND deleted_soft = 0
                  AND display_suppressed = 0
                  AND dropped_late = 0
                  AND translated_mono_ms IS NOT NULL
                  AND status IN ({placeholders})
                  {scope_sql}
                ORDER BY updated_at DESC, (due_mono_ms + ?) DESC
                LIMIT 1
                """,
                (int(delay_adjust_ms), *READY_STATUSES, *scope_params, int(delay_adjust_ms)),
            ).fetchone()
        if row is None:
            return None
        return {
            "source_key": str(row["source_key"]),
            "translated_mono_ms": int(row["translated_mono_ms"]),
            "updated_at": float(row["updated_at"]),
            "due_effective_mono_ms": int(row["due_effective_mono_ms"]),
        }

    def fetch_srt_ready(
        self,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> List[CueRecord]:
        placeholders = ",".join("?" for _ in READY_STATUSES)
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM cues
                WHERE srt_index IS NOT NULL
                  AND (zh_text IS NOT NULL OR manual_zh_text IS NOT NULL)
                  AND deleted_soft = 0
                  AND display_suppressed = 0
                  AND status IN ({placeholders})
                  {scope_sql}
                ORDER BY srt_index ASC, t_seen_mono_ms ASC
                """,
                (*READY_STATUSES, *scope_params),
            ).fetchall()
        return [self._row_to_cue(row) for row in rows]

    def fetch_new_queue_stats(
        self,
        now_mono_ms: int,
        delay_adjust_ms: int = 0,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> Dict[str, Optional[int]]:
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT
                  SUM(CASE WHEN status='NEW' THEN 1 ELSE 0 END) AS new_count,
                  SUM(CASE WHEN status='INFLIGHT' THEN 1 ELSE 0 END) AS inflight_count,
                  SUM(CASE WHEN status IN ('NEW','INFLIGHT') AND (due_mono_ms + ?) <= ? THEN 1 ELSE 0 END) AS overdue_unfinished_count,
                  MIN(CASE WHEN status='NEW' THEN t_seen_mono_ms END) AS oldest_new_seen,
                  MIN(CASE WHEN status IN ('NEW', 'INFLIGHT') THEN due_mono_ms + ? END) AS min_due_unfinished
                FROM cues
                WHERE 1=1
                {scope_sql}
                """,
                (int(delay_adjust_ms), int(now_mono_ms), int(delay_adjust_ms), *scope_params),
            ).fetchone()
        new_count = int(row["new_count"] or 0)
        inflight_count = int(row["inflight_count"] or 0)
        overdue_unfinished_count = int(row["overdue_unfinished_count"] or 0)
        oldest_seen = row["oldest_new_seen"]
        min_due = row["min_due_unfinished"]
        oldest_wait = None if oldest_seen is None else max(0, now_mono_ms - int(oldest_seen))
        return {
            "new_count": new_count,
            "inflight_count": inflight_count,
            "overdue_unfinished_count": overdue_unfinished_count,
            "unfinished_count": new_count + inflight_count,
            "oldest_new_seen_mono_ms": None if oldest_seen is None else int(oldest_seen),
            "oldest_new_wait_ms": oldest_wait,
            "min_due_unfinished_mono_ms": None if min_due is None else int(min_due),
        }

    def fetch_recent_cues(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT source_key, status, t_seen_mono_ms, due_mono_ms, translated_mono_ms,
                       dropped_late, llm_latency_ms, last_error, context_miss,
                       stage1_provider, stage1_model, stage2_provider, stage2_model, fallback_used,
                       stage1_latency_ms, stage2_latency_ms, s1_skipped, aggregator_reason,
                       display_suppressed, join_target_source_key,
                       jp_raw, jp_aggregated, jp_canonicalized, jp_corrected, zh_text, manual_zh_text, manual_locked,
                       deleted_soft, displayed_at_mono_ms, updated_by, revision
                FROM cues
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "source_key": str(row["source_key"]),
                    "status": str(row["status"]),
                    "t_seen_mono_ms": int(row["t_seen_mono_ms"]),
                    "due_mono_ms": int(row["due_mono_ms"]),
                    "translated_mono_ms": None if row["translated_mono_ms"] is None else int(row["translated_mono_ms"]),
                    "dropped_late": bool(row["dropped_late"]),
                    "llm_latency_ms": None if row["llm_latency_ms"] is None else int(row["llm_latency_ms"]),
                    "last_error": row["last_error"],
                    "context_miss": bool(row["context_miss"]),
                    "stage1_provider": row["stage1_provider"],
                    "stage1_model": row["stage1_model"],
                    "stage2_provider": row["stage2_provider"],
                    "stage2_model": row["stage2_model"],
                    "fallback_used": bool(row["fallback_used"]),
                    "stage1_latency_ms": None if row["stage1_latency_ms"] is None else int(row["stage1_latency_ms"]),
                    "stage2_latency_ms": None if row["stage2_latency_ms"] is None else int(row["stage2_latency_ms"]),
                    "s1_skipped": bool(row["s1_skipped"]),
                    "aggregator_reason": row["aggregator_reason"],
                    "display_suppressed": bool(row["display_suppressed"]),
                    "join_target_source_key": row["join_target_source_key"],
                    "jp_raw": row["jp_raw"],
                    "jp_aggregated": row["jp_aggregated"],
                    "jp_canonicalized": row["jp_canonicalized"],
                    "jp_corrected": row["jp_corrected"],
                    "zh_text": row["zh_text"],
                    "manual_zh_text": row["manual_zh_text"],
                    "manual_locked": bool(row["manual_locked"]),
                    "deleted_soft": bool(row["deleted_soft"]),
                    "displayed_at_mono_ms": None
                    if row["displayed_at_mono_ms"] is None
                    else int(row["displayed_at_mono_ms"]),
                    "updated_by": row["updated_by"],
                    "revision": int(row["revision"] or 0),
                }
            )
        return out

    def fetch_translated_cues(self) -> List[CueRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM cues
                WHERE translated_mono_ms IS NOT NULL
                ORDER BY t_seen_mono_ms ASC, source_key ASC
                """
            ).fetchall()
        return [self._row_to_cue(row) for row in rows]

    def fetch_cues_window(
        self,
        *,
        now_mono_ms: int,
        past_window_ms: int,
        future_window_ms: int,
        delay_adjust_ms: int,
        limit: int,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
        cursor_due_effective_mono_ms: Optional[int] = None,
        cursor_source_key: Optional[str] = None,
    ) -> List[CueRecord]:
        low_due = int(now_mono_ms - max(0, past_window_ms))
        high_due = int(now_mono_ms + max(0, future_window_ms))
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
            table_alias="c",
        )
        sql = """
            SELECT * FROM (
                SELECT cues.*, (due_mono_ms + ?) AS due_effective_mono_ms
                FROM cues
            ) AS c
            WHERE c.due_effective_mono_ms BETWEEN ? AND ?
        """
        params: List[Any] = [int(delay_adjust_ms), low_due, high_due]
        if scope_sql:
            sql += scope_sql
            params.extend(scope_params)
        if cursor_due_effective_mono_ms is not None and cursor_source_key is not None:
            sql += " AND (c.due_effective_mono_ms < ? OR (c.due_effective_mono_ms = ? AND c.source_key < ?))"
            params.extend([int(cursor_due_effective_mono_ms), int(cursor_due_effective_mono_ms), str(cursor_source_key)])
        sql += """
            ORDER BY c.due_effective_mono_ms DESC, c.source_key DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_cue(row) for row in rows]

    def fetch_cue_by_key(self, source_key: str) -> Optional[CueRecord]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM cues WHERE source_key = ?", (source_key,)).fetchone()
        if row is None:
            return None
        return self._row_to_cue(row)

    def upsert_manual_translation(
        self,
        *,
        source_key: str,
        text: str,
        now_mono_ms: int,
        delay_adjust_ms: int = 0,
        expected_revision: Optional[int],
    ) -> Dict[str, Any]:
        now = time.time()
        new_text = str(text).strip()
        if not new_text:
            return {"ok": False, "code": "empty_text", "detail": "manual text must not be empty"}
        with self._lock:
            row = self._conn.execute("SELECT * FROM cues WHERE source_key = ?", (source_key,)).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "detail": "cue not found"}

            cue = self._row_to_cue(row)
            check = self._validate_manual_editable(cue, now_mono_ms, delay_adjust_ms)
            if check is not None:
                return check
            if expected_revision is not None and int(cue.revision) != int(expected_revision):
                return {"ok": False, "code": "revision_conflict", "detail": "revision conflict"}

            self._conn.execute(
                """
                UPDATE cues
                SET manual_zh_text = ?,
                    manual_locked = 1,
                    updated_by = 'manual',
                    revision = revision + 1,
                    updated_at = ?
                WHERE source_key = ?
                """,
                (new_text, now, source_key),
            )
            self._conn.commit()
        return {"ok": True}

    def soft_delete_cue(
        self,
        *,
        source_key: str,
        now_mono_ms: int,
        delay_adjust_ms: int = 0,
        expected_revision: Optional[int],
    ) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT * FROM cues WHERE source_key = ?", (source_key,)).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "detail": "cue not found"}
            cue = self._row_to_cue(row)
            check = self._validate_pre_display_mutation(cue, now_mono_ms, delay_adjust_ms)
            if check is not None:
                return check
            if expected_revision is not None and int(cue.revision) != int(expected_revision):
                return {"ok": False, "code": "revision_conflict", "detail": "revision conflict"}
            self._conn.execute(
                """
                UPDATE cues
                SET deleted_soft = 1,
                    updated_by = 'manual',
                    revision = revision + 1,
                    updated_at = ?
                WHERE source_key = ?
                """,
                (now, source_key),
            )
            self._conn.commit()
        return {"ok": True}

    def restore_cue(self, *, source_key: str, expected_revision: Optional[int]) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT * FROM cues WHERE source_key = ?", (source_key,)).fetchone()
            if row is None:
                return {"ok": False, "code": "not_found", "detail": "cue not found"}
            cue = self._row_to_cue(row)
            if expected_revision is not None and int(cue.revision) != int(expected_revision):
                return {"ok": False, "code": "revision_conflict", "detail": "revision conflict"}
            self._conn.execute(
                """
                UPDATE cues
                SET deleted_soft = 0,
                    updated_by = 'manual',
                    revision = revision + 1,
                    updated_at = ?
                WHERE source_key = ?
                """,
                (now, source_key),
            )
            self._conn.commit()
        return {"ok": True}

    def mark_displayed(self, source_keys: Sequence[str], displayed_at_mono_ms: int) -> int:
        keys = [str(k) for k in source_keys if k]
        if not keys:
            return 0
        placeholders = ",".join("?" for _ in keys)
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                UPDATE cues
                SET displayed_at_mono_ms = ?,
                    updated_at = ?
                WHERE source_key IN ({placeholders})
                  AND displayed_at_mono_ms IS NULL
                """,
                (int(displayed_at_mono_ms), now, *keys),
            )
            count = cur.rowcount
            self._conn.commit()
        return int(count)

    def mark_due_unshown_as_displayed(
        self,
        now_mono_ms: int,
        delay_adjust_ms: int = 0,
        *,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> int:
        now = time.time()
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                UPDATE cues
                SET displayed_at_mono_ms = ?,
                    updated_at = ?
                WHERE (due_mono_ms + ?) <= ?
                  AND displayed_at_mono_ms IS NULL
                  AND deleted_soft = 0
                  AND display_suppressed = 0
                  AND dropped_late = 0
                  AND status IN ('TRANSLATED', 'FALLBACK_READY')
                  {scope_sql}
                """,
                (int(now_mono_ms), now, int(delay_adjust_ms), int(now_mono_ms), *scope_params),
            )
            count = cur.rowcount
            self._conn.commit()
        return int(count)

    def flush_unfinished(
        self,
        *,
        now_mono_ms: int,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> int:
        now = time.time()
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                UPDATE cues
                SET status='FLUSHED',
                    dropped_late=1,
                    deleted_soft=1,
                    displayed_at_mono_ms=COALESCE(displayed_at_mono_ms, ?),
                    inflight_owner=NULL,
                    inflight_since_mono_ms=NULL,
                    updated_by='manual',
                    last_error=COALESCE(last_error, 'flushed_by_operator'),
                    revision=revision+1,
                    updated_at=?
                WHERE status IN ('NEW', 'INFLIGHT')
                {scope_sql}
                """,
                (int(now_mono_ms), now, *scope_params),
            )
            count = int(cur.rowcount or 0)
            self._conn.commit()
        return count

    def fetch_window_stats(
        self,
        *,
        now_mono_ms: int,
        past_window_ms: int,
        future_window_ms: int,
        delay_adjust_ms: int,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> Dict[str, int]:
        low_due = int(now_mono_ms - max(0, past_window_ms))
        high_due = int(now_mono_ms + max(0, future_window_ms))
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN status IN ('TRANSLATED','FALLBACK_READY') THEN 1 ELSE 0 END) AS translated_count,
                    SUM(CASE WHEN deleted_soft=1 THEN 1 ELSE 0 END) AS deleted_count,
                    SUM(CASE WHEN displayed_at_mono_ms IS NOT NULL THEN 1 ELSE 0 END) AS displayed_count,
                    SUM(CASE
                        WHEN status IN ('TRANSLATED','FALLBACK_READY')
                          AND (due_mono_ms + ?) > ?
                          AND deleted_soft=0
                          AND display_suppressed=0
                          AND displayed_at_mono_ms IS NULL
                        THEN 1 ELSE 0 END) AS editable_count
                FROM cues
                WHERE (due_mono_ms + ?) BETWEEN ? AND ?
                {scope_sql}
                """,
                (int(delay_adjust_ms), int(now_mono_ms), int(delay_adjust_ms), low_due, high_due, *scope_params),
            ).fetchone()
        return {
            "total_count": int(row["total_count"] or 0),
            "translated_count": int(row["translated_count"] or 0),
            "deleted_count": int(row["deleted_count"] or 0),
            "displayed_count": int(row["displayed_count"] or 0),
            "editable_count": int(row["editable_count"] or 0),
        }

    def fetch_latest_editable_source_key(
        self,
        *,
        now_mono_ms: int,
        past_window_ms: int,
        future_window_ms: int,
        delay_adjust_ms: int,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        updated_after_unix: Optional[float] = None,
    ) -> Optional[str]:
        low_due = int(now_mono_ms - max(0, past_window_ms))
        high_due = int(now_mono_ms + max(0, future_window_ms))
        scope_sql, scope_params = self._scope_sql(
            allowed_source_kinds=allowed_source_kinds,
            updated_after_unix=updated_after_unix,
        )
        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT source_key
                FROM cues
                WHERE (due_mono_ms + ?) BETWEEN ? AND ?
                  AND status IN ('TRANSLATED','FALLBACK_READY')
                  AND (due_mono_ms + ?) > ?
                  AND deleted_soft = 0
                  AND display_suppressed = 0
                  AND displayed_at_mono_ms IS NULL
                  {scope_sql}
                ORDER BY (due_mono_ms + ?) DESC, source_key DESC
                LIMIT 1
                """,
                (
                    int(delay_adjust_ms),
                    low_due,
                    high_due,
                    int(delay_adjust_ms),
                    int(now_mono_ms),
                    *scope_params,
                    int(delay_adjust_ms),
                ),
            ).fetchone()
        if row is None:
            return None
        return str(row["source_key"])

    @staticmethod
    def _validate_pre_display_mutation(cue: CueRecord, now_mono_ms: int, delay_adjust_ms: int = 0) -> Optional[Dict[str, Any]]:
        if cue.displayed_at_mono_ms is not None:
            return {"ok": False, "code": "already_displayed", "detail": "cue already displayed"}
        effective_due_mono_ms = int(cue.due_mono_ms) + int(delay_adjust_ms)
        if effective_due_mono_ms <= int(now_mono_ms):
            return {"ok": False, "code": "past_due", "detail": "cue already due"}
        return None

    @classmethod
    def _validate_manual_editable(
        cls,
        cue: CueRecord,
        now_mono_ms: int,
        delay_adjust_ms: int = 0,
    ) -> Optional[Dict[str, Any]]:
        pre = cls._validate_pre_display_mutation(cue, now_mono_ms, delay_adjust_ms)
        if pre is not None:
            return pre
        if cue.deleted_soft:
            return {"ok": False, "code": "deleted", "detail": "cue was soft deleted"}
        if cue.display_suppressed:
            return {"ok": False, "code": "suppressed", "detail": "cue is merged into next line"}
        if cue.status not in READY_STATUSES:
            return {"ok": False, "code": "not_translated", "detail": "cue is not translated yet"}
        return None

    def record_metric(
        self,
        stage: str,
        batch_size: int,
        latency_ms: int,
        ok: bool,
        timed_out: bool,
        cached_tokens: Optional[int],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO metrics_events(stage, batch_size, latency_ms, ok, timed_out, cached_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stage,
                    batch_size,
                    latency_ms,
                    int(ok),
                    int(timed_out),
                    cached_tokens,
                    time.time(),
                ),
            )
            self._conn.commit()

    def fetch_recent_metric_latencies(self, stage: str, limit: int = 200) -> List[int]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT latency_ms FROM metrics_events
                WHERE stage = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (stage, limit),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO meta(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    @staticmethod
    def _row_to_cue(row: sqlite3.Row) -> CueRecord:
        return CueRecord(
            source_key=str(row["source_key"]),
            source_kind=str(row["source_kind"]),
            srt_index=row["srt_index"],
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            jp_raw=str(row["jp_raw"]),
            jp_aggregated=row["jp_aggregated"],
            jp_canonicalized=row["jp_canonicalized"],
            jp_corrected=row["jp_corrected"],
            zh_text=row["zh_text"],
            status=str(row["status"]),
            t_seen_mono_ms=int(row["t_seen_mono_ms"]),
            due_mono_ms=int(row["due_mono_ms"]),
            translated_mono_ms=row["translated_mono_ms"],
            dropped_late=bool(row["dropped_late"]),
            llm_latency_ms=row["llm_latency_ms"],
            last_error=row["last_error"],
            context_miss=bool(row["context_miss"]),
            inflight_owner=row["inflight_owner"],
            inflight_since_mono_ms=row["inflight_since_mono_ms"],
            stage1_provider=row["stage1_provider"],
            stage1_model=row["stage1_model"],
            stage2_provider=row["stage2_provider"],
            stage2_model=row["stage2_model"],
            fallback_used=bool(row["fallback_used"]),
            stage1_latency_ms=row["stage1_latency_ms"],
            stage2_latency_ms=row["stage2_latency_ms"],
            s1_skipped=bool(row["s1_skipped"]),
            aggregator_reason=row["aggregator_reason"],
            manual_zh_text=row["manual_zh_text"],
            manual_locked=bool(row["manual_locked"]),
            deleted_soft=bool(row["deleted_soft"]),
            display_suppressed=bool(row["display_suppressed"]),
            join_target_source_key=row["join_target_source_key"],
            displayed_at_mono_ms=row["displayed_at_mono_ms"],
            updated_by=row["updated_by"],
            revision=int(row["revision"] or 0),
        )
