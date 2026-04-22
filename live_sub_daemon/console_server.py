from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]

from .io_utils import atomic_write_text
from .models import CueRecord
from .renderer import build_render_decision
from .runtime_scope import normalize_scope_kinds_for_json
from .runtime_control import RuntimeController
from .state_store import READY_STATUSES, StateStore


class CueRevisionPayload(BaseModel):
    revision: Optional[int] = None


class CuePatchPayload(BaseModel):
    text: str
    revision: Optional[int] = None


class DelayAdjustPayload(BaseModel):
    delta_sec: int


class RuntimePatchPayload(BaseModel):
    updates: Dict[str, Any]


class ModePayload(BaseModel):
    mode: str


class GlossaryUpsertPayload(BaseModel):
    ja: str
    zh: str


class GlossaryDeletePayload(BaseModel):
    ja: str


class NameUpsertPayload(BaseModel):
    content: Optional[str] = None
    name: Optional[str] = None
    sounds_like: Optional[List[str]] = None
    prev_content: Optional[str] = None


class NameDeletePayload(BaseModel):
    content: Optional[str] = None
    name: Optional[str] = None


class ConsoleServer(threading.Thread):
    def __init__(
        self,
        host: str,
        port: int,
        state: StateStore,
        runtime: RuntimeController,
        *,
        glossary_path: Optional[Path] = None,
        names_path: Optional[Path] = None,
        delay_sec: float = 180.0,
        asr_delay_sec: float = 0.0,
        char_threshold: int = 18,
        max_total_chars: int = 42,
        max_lines: int = 2,
        allowed_source_kinds: Optional[List[str]] = None,
        scope_freshness_sec: float = 0.0,
        runtime_scope_monotonic_guard_sec: float = 600.0,
    ):
        super().__init__(daemon=True, name="console-server")
        self.host = host
        self.port = port
        self.state = state
        self.runtime = runtime
        self.glossary_path = glossary_path or Path("glossary_ja_zh.tsv")
        self.names_path = names_path or Path("names_whitelist_ja.txt")
        self.delay_sec = max(0.0, float(delay_sec))
        self.asr_delay_sec = max(0.0, float(asr_delay_sec))
        self.char_threshold = int(char_threshold)
        self.max_total_chars = max(0, int(max_total_chars))
        self.max_lines = max(1, int(max_lines))
        self.allowed_source_kinds = normalize_scope_kinds_for_json(allowed_source_kinds)
        self.scope_freshness_sec = max(0.0, float(scope_freshness_sec))
        self.runtime_scope_monotonic_guard_sec = max(0.0, float(runtime_scope_monotonic_guard_sec))
        self._stop_event = threading.Event()
        self._server: Any = None
        self.started = False
        self.start_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.should_exit = True

    def run(self) -> None:
        try:
            import uvicorn
        except Exception as exc:  # noqa: BLE001
            self.start_error = f"console disabled, FastAPI/uvicorn unavailable: {exc}"
            return

        app = build_console_app(
            state=self.state,
            runtime=self.runtime,
            stop_event=self._stop_event,
            glossary_path=self.glossary_path,
            names_path=self.names_path,
            char_threshold=self.char_threshold,
            max_total_chars=self.max_total_chars,
            max_lines=self.max_lines,
            allowed_source_kinds=self.allowed_source_kinds,
            scope_freshness_sec=self.scope_freshness_sec,
            runtime_scope_monotonic_guard_sec=self.runtime_scope_monotonic_guard_sec,
        )

        cfg = uvicorn.Config(app=app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(cfg)
        self.started = True
        self._server.run()


def build_console_app(
    *,
    state: StateStore,
    runtime: RuntimeController,
    stop_event: threading.Event,
    glossary_path: Path,
    names_path: Path,
    char_threshold: int,
    max_total_chars: int = 42,
    max_lines: int = 2,
    allowed_source_kinds: Optional[List[str]] = None,
    scope_freshness_sec: float = 0.0,
    runtime_scope_monotonic_guard_sec: float = 600.0,
) -> Any:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Live Sub Console", version="3.0")
    dist_dir = Path(__file__).resolve().parents[1] / "web_console" / "dist"
    has_dist = (dist_dir / "index.html").exists()
    if has_dist and (dist_dir / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(dist_dir / "assets")), name="assets")

    terms_lock = threading.Lock()
    allowed_scope_kinds = normalize_scope_kinds_for_json(allowed_source_kinds)
    freshness_sec = max(0.0, float(scope_freshness_sec))
    monotonic_guard_sec = max(0.0, float(runtime_scope_monotonic_guard_sec))

    def terms_version() -> Dict[str, int]:
        g = int(glossary_path.stat().st_mtime_ns) if glossary_path.exists() else 0
        n = int(names_path.stat().st_mtime_ns) if names_path.exists() else 0
        return {"glossary_mtime_ns": g, "names_mtime_ns": n}

    def scope_query_params() -> Dict[str, Any]:
        updated_after_unix = None
        if freshness_sec > 0:
            updated_after_unix = time.time() - freshness_sec
        return {
            "allowed_source_kinds": allowed_scope_kinds,
            "updated_after_unix": updated_after_unix,
        }

    def fast_catalog() -> Dict[str, Dict[str, str]]:
        return runtime.get_fast_model_catalog()

    @app.get("/", include_in_schema=False)
    async def index() -> Any:
        if has_dist:
            return FileResponse(dist_dir / "index.html")
        return HTMLResponse(_LEGACY_HTML)

    @app.get("/api/status", response_class=JSONResponse)
    async def api_status() -> Dict[str, Any]:
        status = dict(runtime.get_status())
        status["terms_version"] = terms_version()
        status["scope_allowed_source_kinds"] = allowed_scope_kinds
        status["scope_freshness_sec"] = freshness_sec
        return status

    @app.get("/api/metrics/recent", response_class=JSONResponse)
    async def api_metrics_recent() -> Dict[str, Any]:
        status = runtime.get_status()
        return {
            "arrival_rate_lps": status.get("arrival_rate_lps", 0.0),
            "service_rate_lps": status.get("service_rate_lps", 0.0),
            "late_rate_recent": status.get("late_rate_recent", 0.0),
            "stage": status.get("stage", {}),
        }

    @app.get("/api/queue", response_class=JSONResponse)
    async def api_queue() -> Dict[str, Any]:
        status = runtime.get_status()
        return dict(status.get("queue", {}))

    @app.get("/api/cues/recent", response_class=JSONResponse)
    async def api_cues_recent(limit: int = 50) -> Dict[str, Any]:
        rows = state.fetch_recent_cues(limit=max(1, min(limit, 500)))
        return {"items": rows}

    @app.get("/api/cues/window", response_class=JSONResponse)
    async def api_cues_window(window_sec: Optional[float] = None, limit: int = 300, cursor: Optional[str] = None) -> Dict[str, Any]:
        now_mono_ms = int(time.monotonic() * 1000)
        delay_adjust_ms = runtime.delay_adjust_ms()
        effective_delay_sec = runtime.effective_delay_sec()
        delay_eff_ms = int(max(0.0, effective_delay_sec) * 1000)
        model_catalog = fast_catalog()
        past_window_sec = max(1.0, 0.5 * effective_delay_sec)
        future_window_sec = float(window_sec) if window_sec is not None else max(1.0, 1.5 * effective_delay_sec)
        cursor_due, cursor_key = _parse_cursor(cursor)

        cues = state.fetch_cues_window(
            now_mono_ms=now_mono_ms,
            past_window_ms=int(past_window_sec * 1000),
            future_window_ms=int(future_window_sec * 1000),
            delay_adjust_ms=delay_adjust_ms,
            limit=max(1, min(limit, 1000)),
            cursor_due_effective_mono_ms=cursor_due,
            cursor_source_key=cursor_key,
            **scope_query_params(),
        )
        items = [
            _serialize_cue(
                cue,
                now_mono_ms=now_mono_ms,
                delay_eff_ms=delay_eff_ms,
                delay_adjust_ms=delay_adjust_ms,
                fast_model_catalog=model_catalog,
            )
            for cue in cues
        ]
        next_cursor = None
        if cues:
            last = cues[-1]
            next_cursor = _make_cursor(last.due_mono_ms + delay_adjust_ms, last.source_key)
        return {
            "items": items,
            "next_cursor": next_cursor,
            "window_sec": future_window_sec,
            "past_window_sec": past_window_sec,
            "future_window_sec": future_window_sec,
            "now_mono_ms": now_mono_ms,
            "delay_eff_ms": delay_eff_ms,
            "delay_adjust_sec": float(delay_adjust_ms / 1000.0),
            "effective_delay_sec": effective_delay_sec,
        }

    @app.get("/api/cues/{source_key:path}", response_class=JSONResponse)
    async def api_cues_detail(source_key: str) -> Dict[str, Any]:
        now_mono_ms = int(time.monotonic() * 1000)
        model_catalog = fast_catalog()
        cue = state.fetch_cue_by_key(source_key)
        if cue is None:
            raise HTTPException(status_code=404, detail="cue not found")
        return {
            "item": _serialize_cue(
                cue,
                now_mono_ms=now_mono_ms,
                delay_eff_ms=runtime.effective_delay_ms(),
                delay_adjust_ms=runtime.delay_adjust_ms(),
                fast_model_catalog=model_catalog,
            )
        }

    @app.patch("/api/cues/{source_key:path}", response_class=JSONResponse)
    async def api_cues_patch(source_key: str, payload: CuePatchPayload = Body(...)) -> Dict[str, Any]:
        result = state.upsert_manual_translation(
            source_key=source_key,
            text=payload.text,
            now_mono_ms=int(time.monotonic() * 1000),
            delay_adjust_ms=runtime.delay_adjust_ms(),
            expected_revision=payload.revision,
        )
        _raise_for_state_error(result)
        return {"ok": True}

    @app.post("/api/cues/{source_key:path}/delete", response_class=JSONResponse)
    async def api_cues_delete(
        source_key: str,
        payload: Optional[CueRevisionPayload] = Body(default=None),
    ) -> Dict[str, Any]:
        result = state.soft_delete_cue(
            source_key=source_key,
            now_mono_ms=int(time.monotonic() * 1000),
            delay_adjust_ms=runtime.delay_adjust_ms(),
            expected_revision=payload.revision if payload is not None else None,
        )
        _raise_for_state_error(result)
        return {"ok": True}

    @app.post("/api/cues/{source_key:path}/restore", response_class=JSONResponse)
    async def api_cues_restore(
        source_key: str,
        payload: Optional[CueRevisionPayload] = Body(default=None),
    ) -> Dict[str, Any]:
        result = state.restore_cue(
            source_key=source_key,
            expected_revision=payload.revision if payload is not None else None,
        )
        _raise_for_state_error(result)
        return {"ok": True}

    @app.get("/api/subtitle/current", response_class=JSONResponse)
    async def api_subtitle_current() -> Dict[str, Any]:
        now_mono_ms = int(time.monotonic() * 1000)
        delay_adjust_ms = runtime.delay_adjust_ms()
        model_catalog = fast_catalog()
        candidates = state.fetch_render_candidates(
            now_mono_ms=now_mono_ms,
            limit=2,
            delay_adjust_ms=delay_adjust_ms,
            **scope_query_params(),
        )
        decision = build_render_decision(
            candidates_desc=candidates,
            char_threshold=char_threshold,
        )
        return {
            "text": decision.text,
            "cue_keys": decision.cue_keys,
            "items": [
                _serialize_cue(
                    cue,
                    now_mono_ms=now_mono_ms,
                    delay_eff_ms=runtime.effective_delay_ms(),
                    delay_adjust_ms=delay_adjust_ms,
                    fast_model_catalog=model_catalog,
                )
                for cue in candidates
            ],
        }

    @app.get("/api/terms/glossary", response_class=JSONResponse)
    async def api_terms_glossary() -> Dict[str, Any]:
        entries = _read_glossary_entries(glossary_path)
        return {"items": entries, "version": terms_version()}

    @app.post("/api/terms/glossary/upsert", response_class=JSONResponse)
    async def api_terms_glossary_upsert(payload: GlossaryUpsertPayload = Body(...)) -> Dict[str, Any]:
        ja = str(payload.ja).strip()
        zh = str(payload.zh).strip()
        if not ja or not zh:
            raise HTTPException(status_code=400, detail="ja and zh are required")
        with terms_lock:
            entries = _read_glossary_entries(glossary_path)
            index = {it["ja"]: idx for idx, it in enumerate(entries)}
            if ja in index:
                entries[index[ja]]["zh"] = zh
            else:
                entries.append({"ja": ja, "zh": zh})
            _write_glossary_entries(glossary_path, entries)
            line_count = len(entries)
        return {
            "ok": True,
            "version": terms_version(),
            "file_path": str(glossary_path.resolve()),
            "line_count": line_count,
        }

    @app.post("/api/terms/glossary/delete", response_class=JSONResponse)
    async def api_terms_glossary_delete(payload: GlossaryDeletePayload = Body(...)) -> Dict[str, Any]:
        ja = str(payload.ja).strip()
        if not ja:
            raise HTTPException(status_code=400, detail="ja is required")
        with terms_lock:
            entries = [it for it in _read_glossary_entries(glossary_path) if it["ja"] != ja]
            _write_glossary_entries(glossary_path, entries)
            line_count = len(entries)
        return {
            "ok": True,
            "version": terms_version(),
            "file_path": str(glossary_path.resolve()),
            "line_count": line_count,
        }

    @app.get("/api/terms/names", response_class=JSONResponse)
    async def api_terms_names() -> Dict[str, Any]:
        names = _read_names_entries(names_path)
        return {
            "items": names,
            "legacy_items": [str(it.get("content", "")) for it in names if it.get("content")],
            "version": terms_version(),
        }

    @app.post("/api/terms/names/upsert", response_class=JSONResponse)
    async def api_terms_names_upsert(payload: NameUpsertPayload = Body(...)) -> Dict[str, Any]:
        content = _extract_name_content(payload.content, payload.name)
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        prev_content = str(payload.prev_content or "").strip()
        sounds_like = _normalize_sounds_like(payload.sounds_like)
        with terms_lock:
            names = _read_names_entries(names_path)
            if prev_content and prev_content != content:
                names = [it for it in names if str(it.get("content", "")) != prev_content]
            found = False
            for item in names:
                if str(item.get("content", "")) == content:
                    item["sounds_like"] = list(sounds_like)
                    found = True
                    break
            if not found:
                names.append({"content": content, "sounds_like": list(sounds_like)})
            _write_names_entries(names_path, names)
            line_count = len(_read_names_entries(names_path))
        return {
            "ok": True,
            "version": terms_version(),
            "file_path": str(names_path.resolve()),
            "line_count": line_count,
        }

    @app.post("/api/terms/names/add", response_class=JSONResponse)
    async def api_terms_names_add(payload: NameUpsertPayload = Body(...)) -> Dict[str, Any]:
        # Backward compatible alias; routes to structured upsert.
        return await api_terms_names_upsert(payload)

    @app.post("/api/terms/names/delete", response_class=JSONResponse)
    async def api_terms_names_delete(payload: NameDeletePayload = Body(...)) -> Dict[str, Any]:
        content = _extract_name_content(payload.content, payload.name)
        if not content:
            raise HTTPException(status_code=400, detail="content is required")
        with terms_lock:
            names = [it for it in _read_names_entries(names_path) if str(it.get("content", "")) != content]
            _write_names_entries(names_path, names)
            line_count = len(names)
        return {
            "ok": True,
            "version": terms_version(),
            "file_path": str(names_path.resolve()),
            "line_count": line_count,
        }

    @app.post("/api/control/runtime", response_class=JSONResponse)
    async def api_control_runtime(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="payload must be object")
        updates = runtime.apply_runtime_updates(payload)
        return {"ok": True, "applied": updates}

    @app.post("/api/control/mode", response_class=JSONResponse)
    async def api_control_mode(payload: ModePayload = Body(...)) -> Dict[str, Any]:
        mode = str(payload.mode).strip() if payload.mode is not None else ""
        if not mode:
            raise HTTPException(status_code=400, detail="mode required")
        normalized = runtime.set_mode(mode)
        return {"ok": True, "mode": normalized}

    @app.post("/api/control/delay_adjust", response_class=JSONResponse)
    async def api_control_delay_adjust(payload: DelayAdjustPayload = Body(...)) -> Dict[str, Any]:
        updated = runtime.adjust_delay(payload.delta_sec)
        return {
            "ok": True,
            "delay_adjust_sec": updated,
            "effective_delay_sec": runtime.effective_delay_sec(),
            "base_delay_sec": runtime.base_delay_sec(),
            "asr_delay_sec": runtime.asr_delay_sec(),
        }

    @app.post("/api/control/delay_reset", response_class=JSONResponse)
    async def api_control_delay_reset() -> Dict[str, Any]:
        updated = runtime.reset_delay_adjust()
        return {
            "ok": True,
            "delay_adjust_sec": updated,
            "effective_delay_sec": runtime.effective_delay_sec(),
            "base_delay_sec": runtime.base_delay_sec(),
            "asr_delay_sec": runtime.asr_delay_sec(),
        }

    @app.post("/api/control/jump_to_latest", response_class=JSONResponse)
    async def api_control_jump_to_latest() -> Dict[str, Any]:
        now_mono_ms = int(time.monotonic() * 1000)
        delay_adjust_ms = runtime.delay_adjust_ms()
        skipped = state.mark_due_unshown_as_displayed(
            now_mono_ms=now_mono_ms,
            delay_adjust_ms=delay_adjust_ms,
            **scope_query_params(),
        )
        state.set_meta("render_jump_to_latest_seq", str(now_mono_ms))
        return {
            "ok": True,
            "skipped_count": skipped,
            "timestamp_mono_ms": now_mono_ms,
        }

    @app.post("/api/control/flush_pending", response_class=JSONResponse)
    async def api_control_flush_pending() -> Dict[str, Any]:
        now_mono_ms = int(time.monotonic() * 1000)
        flushed = state.flush_unfinished(
            now_mono_ms=now_mono_ms,
            **scope_query_params(),
        )
        return {
            "ok": True,
            "flushed_count": int(flushed),
            "timestamp_mono_ms": now_mono_ms,
        }

    @app.post("/api/control/cleanup_stale", response_class=JSONResponse)
    async def api_control_cleanup_stale() -> Dict[str, Any]:
        now_mono_ms = int(time.monotonic() * 1000)
        result = state.cleanup_runtime_scope(
            now_mono_ms=now_mono_ms,
            allowed_source_kinds=allowed_scope_kinds,
            stale_unfinished_sec=freshness_sec,
            monotonic_guard_sec=monotonic_guard_sec,
            anomaly_remaining_max_sec=max(300.0, runtime.effective_delay_sec() * 3.0),
        )
        return {"ok": True, **result}

    @app.get("/api/stream")
    async def api_stream() -> StreamingResponse:
        async def gen() -> Any:
            while not stop_event.is_set():
                now_mono_ms = int(time.monotonic() * 1000)
                delay_adjust_ms = runtime.delay_adjust_ms()
                effective_delay_sec = runtime.effective_delay_sec()
                past_window_ms = int(max(1000.0, 0.5 * effective_delay_sec * 1000.0))
                future_window_ms = int(max(1000.0, 1.5 * effective_delay_sec * 1000.0))
                payload = dict(runtime.get_status())
                payload["window_stats"] = state.fetch_window_stats(
                    now_mono_ms=now_mono_ms,
                    past_window_ms=past_window_ms,
                    future_window_ms=future_window_ms,
                    delay_adjust_ms=delay_adjust_ms,
                    **scope_query_params(),
                )
                payload["latest_editable_source_key"] = state.fetch_latest_editable_source_key(
                    now_mono_ms=now_mono_ms,
                    past_window_ms=past_window_ms,
                    future_window_ms=future_window_ms,
                    delay_adjust_ms=delay_adjust_ms,
                    **scope_query_params(),
                )
                payload["terms_version"] = terms_version()
                payload["now_mono_ms"] = now_mono_ms
                payload["past_window_sec"] = float(past_window_ms / 1000.0)
                payload["future_window_sec"] = float(future_window_ms / 1000.0)
                payload["window_sec"] = float(future_window_ms / 1000.0)
                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield line.encode("utf-8")
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        if has_dist:
            return FileResponse(dist_dir / "index.html")
        return HTMLResponse(_LEGACY_HTML)

    return app


def _serialize_cue(
    cue: CueRecord,
    *,
    now_mono_ms: int,
    delay_eff_ms: int,
    delay_adjust_ms: int,
    fast_model_catalog: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    display_text = _cue_display_text(cue)
    status = str(cue.status)
    due_effective_mono_ms = int(cue.due_mono_ms) + int(delay_adjust_ms)
    editable = (
        status in READY_STATUSES
        and due_effective_mono_ms > int(now_mono_ms)
        and not cue.deleted_soft
        and not cue.display_suppressed
        and cue.displayed_at_mono_ms is None
    )
    countdown_sec = (due_effective_mono_ms - int(now_mono_ms)) / 1000.0
    progress = 0.0
    if delay_eff_ms > 0:
        progress = (int(now_mono_ms) - cue.t_seen_mono_ms) / float(delay_eff_ms)
        progress = max(0.0, min(1.0, progress))
    used_fast_model = _cue_used_fast_model(cue, fast_model_catalog or {})
    return {
        "source_key": cue.source_key,
        "status": cue.status,
        "t_seen_mono_ms": cue.t_seen_mono_ms,
        "due_mono_ms": cue.due_mono_ms,
        "due_effective_mono_ms": due_effective_mono_ms,
        "translated_mono_ms": cue.translated_mono_ms,
        "displayed_at_mono_ms": cue.displayed_at_mono_ms,
        "deleted_soft": cue.deleted_soft,
        "display_suppressed": cue.display_suppressed,
        "join_target_source_key": cue.join_target_source_key,
        "dropped_late": cue.dropped_late,
        "countdown_sec": countdown_sec,
        "progress": progress,
        "editable": editable,
        "display_text": display_text,
        "jp_raw": cue.jp_raw,
        "jp_aggregated": cue.jp_aggregated,
        "jp_canonicalized": cue.jp_canonicalized,
        "jp_corrected": cue.jp_corrected,
        "zh_text": cue.zh_text,
        "manual_zh_text": cue.manual_zh_text,
        "manual_locked": cue.manual_locked,
        "updated_by": cue.updated_by,
        "revision": cue.revision,
        "stage1_provider": cue.stage1_provider,
        "stage1_model": cue.stage1_model,
        "stage2_provider": cue.stage2_provider,
        "stage2_model": cue.stage2_model,
        "used_fast_model": used_fast_model,
        "fallback_used": cue.fallback_used,
        "stage1_latency_ms": cue.stage1_latency_ms,
        "stage2_latency_ms": cue.stage2_latency_ms,
        "s1_skipped": cue.s1_skipped,
        "aggregator_reason": cue.aggregator_reason,
        "pipeline_latency_ms": cue.llm_latency_ms,
        "last_error": cue.last_error,
        "source_kind": cue.source_kind,
        "srt_index": cue.srt_index,
        "start_ms": cue.start_ms,
        "end_ms": cue.end_ms,
    }


def _cue_used_fast_model(cue: CueRecord, fast_model_catalog: Dict[str, Dict[str, str]]) -> bool:
    stage2_provider = str(cue.stage2_provider or "").strip()
    stage2_model = str(cue.stage2_model or "").strip()

    if stage2_provider and stage2_model:
        stage2_fast = str(fast_model_catalog.get(stage2_provider, {}).get("fast_translate_model", "")).strip()
        if stage2_fast and stage2_model == stage2_fast:
            return True
    return False


def _cue_display_text(cue: CueRecord) -> str:
    if cue.manual_zh_text:
        return cue.manual_zh_text
    if cue.zh_text:
        return cue.zh_text
    return cue.jp_raw


def _raise_for_state_error(result: Dict[str, Any]) -> None:
    if result.get("ok"):
        return
    code = str(result.get("code") or "invalid")
    detail = str(result.get("detail") or code)
    status_map = {
        "not_found": 404,
        "revision_conflict": 409,
        "already_displayed": 409,
        "past_due": 409,
        "not_translated": 409,
        "deleted": 409,
        "suppressed": 409,
        "empty_text": 400,
    }
    from fastapi import HTTPException

    raise HTTPException(status_code=status_map.get(code, 400), detail=detail)


def _parse_cursor(cursor: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    if not cursor:
        return None, None
    if "|" not in cursor:
        return None, None
    due_raw, key = cursor.split("|", 1)
    try:
        due = int(due_raw)
    except ValueError:
        return None, None
    if not key:
        return None, None
    return due, key


def _make_cursor(due_mono_ms: int, source_key: str) -> str:
    return f"{int(due_mono_ms)}|{source_key}"


def _read_glossary_entries(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    items: List[Dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            ja, zh = line.split("\t", 1)
        elif "," in line:
            ja, zh = line.split(",", 1)
        else:
            continue
        ja = ja.strip()
        zh = zh.strip()
        if ja and zh:
            items.append({"ja": ja, "zh": zh})
    return items


def _write_glossary_entries(path: Path, items: List[Dict[str, str]]) -> None:
    lines = [f"{it['ja']}\t{it['zh']}" for it in items if it.get("ja") and it.get("zh")]
    text = "\n".join(lines)
    if text:
        text += "\n"
    atomic_write_text(path, text, encoding="utf-8")


def _read_names_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        content = line
        sounds_like_raw = ""
        if "\t" in line:
            content, sounds_like_raw = line.split("\t", 1)
        elif "|" in line:
            content, sounds_like_raw = line.split("|", 1)
        content = content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        items.append({"content": content, "sounds_like": _split_sounds_like_text(sounds_like_raw)})
    return items


def _write_names_entries(path: Path, items: List[Dict[str, Any]]) -> None:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        content = str(item.get("content", "")).strip()
        if not content or content in seen:
            continue
        seen.add(content)
        sounds_like = _normalize_sounds_like(item.get("sounds_like"))
        deduped.append({"content": content, "sounds_like": sounds_like})
    lines: List[str] = []
    for item in deduped:
        content = str(item["content"]).strip()
        sounds_like = _normalize_sounds_like(item.get("sounds_like"))
        if sounds_like:
            lines.append(f"{content}\t{','.join(sounds_like)}")
        else:
            lines.append(content)
    text = "\n".join(lines)
    if text:
        text += "\n"
    atomic_write_text(path, text, encoding="utf-8")


def _extract_name_content(content: Optional[str], name: Optional[str]) -> str:
    if content is not None and str(content).strip():
        return str(content).strip()
    if name is not None and str(name).strip():
        return str(name).strip()
    return ""


def _split_sounds_like_text(raw: str) -> List[str]:
    normalized = str(raw or "").replace(chr(0xFF0C), ",").replace(chr(0x3001), ",")
    out: List[str] = []
    seen: set[str] = set()
    for token in normalized.split(","):
        value = token.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_sounds_like(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return _split_sounds_like_text(values)
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _split_sounds_like_text(str(value)):
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
    return out


_LEGACY_HTML = """<!doctype html>
<html><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Console</title><style>body{font-family:Segoe UI,Tahoma,sans-serif;background:#0b1220;color:#dce8ff;padding:24px}code{background:#111a2c;padding:2px 6px;border-radius:6px}</style></head>
<body>
  <h2>Console V3 Frontend Not Built</h2>
  <p>Run <code>cd web_console && npm install && npm run build</code>, then refresh this page.</p>
</body></html>"""
