from __future__ import annotations

import asyncio
import re
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SpeechmaticsConfig, normalize_speechmatics_vocab_source
from .models import SourceCue
from .source_base import BaseSourceReader


class SpeechmaticsSourceReader(BaseSourceReader):
    def __init__(
        self,
        *,
        config: SpeechmaticsConfig,
        glossary_path: Path,
        names_path: Path,
        names_phonetic_canonicalize_enabled: bool = True,
        names_phonetic_max_rules: int = 500,
        runtime: Optional[Any] = None,
        autostart: bool = True,
    ):
        self.config = config
        self.glossary_path = glossary_path
        self.names_path = names_path
        self.names_phonetic_canonicalize_enabled = bool(names_phonetic_canonicalize_enabled)
        self.names_phonetic_max_rules = max(0, int(names_phonetic_max_rules))
        self.runtime = runtime

        self._queue: "queue.Queue[SourceCue]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name="speechmatics-source")
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._capture_proc_lock = threading.Lock()
        self._capture_proc: Optional[asyncio.subprocess.Process] = None
        self._aggregate_lock = threading.Lock()
        self._pending_raw_parts: List[str] = []
        self._pending_canonical_parts: List[str] = []
        self._pending_canonical_hits: int = 0
        self._pending_started_mono_ms: Optional[int] = None
        self._pending_last_mono_ms: Optional[int] = None
        self._canonicalizer = NameCanonicalizer(
            names_path=self.names_path,
            enabled=self.names_phonetic_canonicalize_enabled,
            max_rules=self.names_phonetic_max_rules,
        )
        if autostart:
            self._thread.start()

    def poll(self) -> List[SourceCue]:
        out: List[SourceCue] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self) -> None:
        self._stop_event.set()
        self._terminate_capture_proc_sync()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_forever())
        except Exception as exc:  # noqa: BLE001
            self._set_asr_error(f"speechmatics worker crashed: {exc}")
            self._set_asr_connected(False)

    async def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._run_single_session()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001
                self._set_asr_error(str(exc))
                self._set_asr_connected(False)
                if self._stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(15.0, backoff * 2.0)

    async def _run_single_session(self) -> None:
        sdk = _load_speechmatics_sdk()
        additional_vocab = self._build_additional_vocab(self.config.additional_vocab_limit)

        client = sdk["AsyncClient"](api_key=self._resolve_api_key())
        self._bind_handlers(client, sdk)

        config_kwargs = self._build_transcription_config_kwargs(additional_vocab)
        try:
            transcription_config = sdk["TranscriptionConfig"](**config_kwargs)
        except TypeError as exc:
            if "max_delay_mode" in config_kwargs:
                config_kwargs.pop("max_delay_mode", None)
                transcription_config = sdk["TranscriptionConfig"](**config_kwargs)
                self._set_asr_error(f"max_delay_mode unsupported by installed speechmatics-rt: {exc}")
            else:
                raise
        audio_format = sdk["AudioFormat"](
            encoding=sdk["AudioEncoding"].PCM_S16LE,
            sample_rate=self.config.sample_rate,
        )

        capture_cmd = self.config.capture_cmd.format(
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
        )
        proc = await asyncio.create_subprocess_shell(
            capture_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        with self._capture_proc_lock:
            self._capture_proc = proc

        try:
            await client.start_session(
                transcription_config=transcription_config,
                audio_format=audio_format,
            )
            self._set_asr_connected(True)
            self._set_asr_error(None)
            while not self._stop_event.is_set():
                if proc.stdout is None:
                    raise RuntimeError("Speechmatics capture process stdout unavailable")
                chunk = await proc.stdout.read(self.config.chunk_size)
                if not chunk:
                    if self._stop_event.is_set():
                        break
                    raise RuntimeError("Speechmatics capture process produced no audio data")
                await client.send_audio(chunk)
                self._flush_pending_if_due(now_mono_ms=int(time.monotonic() * 1000))
        finally:
            self._flush_pending_if_due(now_mono_ms=int(time.monotonic() * 1000), force=True)
            self._set_asr_connected(False)
            try:
                await client.close()
            except Exception:  # noqa: BLE001
                pass
            await _terminate_subprocess(proc)
            with self._capture_proc_lock:
                if self._capture_proc is proc:
                    self._capture_proc = None

    def _bind_handlers(self, client: Any, sdk: Dict[str, Any]) -> None:
        server_message_type = sdk["ServerMessageType"]
        transcript_result = sdk["TranscriptResult"]

        @client.on(server_message_type.ADD_PARTIAL_TRANSCRIPT)
        def on_partial(message: Any) -> None:
            text = _extract_transcript_text(transcript_result, message)
            if text:
                self._observe_asr_partial(text)

        @client.on(server_message_type.ADD_TRANSCRIPT)
        def on_final(message: Any) -> None:
            text = _extract_transcript_text(transcript_result, message)
            if not text:
                return
            now_mono_ms = int(time.monotonic() * 1000)
            self._handle_final_text(text=text, now_mono_ms=now_mono_ms)

    def _next_source_key(self, now_mono_ms: int) -> str:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        return f"speechmatics:{seq}:{now_mono_ms}"

    def _resolve_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        if self.config.api_key_env:
            value = os.getenv(self.config.api_key_env, "").strip()
            if value:
                return value
        raise RuntimeError(
            "Speechmatics API key missing. Set source.speechmatics.api_key "
            f"or env {self.config.api_key_env}"
        )

    def _build_transcription_config_kwargs(self, additional_vocab: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "language": self.config.language,
            "operating_point": self.config.operating_point,
            "enable_partials": self.config.enable_partials,
            "max_delay": self.config.max_delay,
            "additional_vocab": additional_vocab,
        }
        if self.config.max_delay_mode:
            payload["max_delay_mode"] = self.config.max_delay_mode
        return payload

    def _handle_final_text(self, text: str, now_mono_ms: int) -> None:
        raw_clean = text.strip()
        if not raw_clean:
            return
        canonical_text, hit_count = self._canonicalizer.apply(raw_clean)
        if not self.config.final_aggregate_enabled:
            self._emit_final_cue(
                raw_text=raw_clean,
                canonical_text=canonical_text,
                now_mono_ms=now_mono_ms,
                aggregator_reason="disabled",
                fragment_count=1,
                canonical_hit_count=hit_count,
            )
            return

        with self._aggregate_lock:
            gap_ms = int(max(0.0, float(self.config.final_aggregate_gap_sec)) * 1000)
            max_window_ms = int(max(0.0, float(self.config.final_aggregate_max_sec)) * 1000)
            force_emit_ms = int(max(0.0, float(self.config.final_aggregate_force_emit_sec)) * 1000)
            max_chars = max(1, int(self.config.final_aggregate_max_chars))
            min_chars = max(1, int(self.config.final_aggregate_min_chars))
            aggregate_mode = str(self.config.final_aggregate_mode).strip().lower()

            if self._pending_canonical_parts and self._pending_last_mono_ms is not None and gap_ms > 0:
                if now_mono_ms - self._pending_last_mono_ms >= gap_ms:
                    self._flush_pending_locked(now_mono_ms=now_mono_ms, reason="gap_timeout")

            if self._pending_canonical_parts and self._pending_canonical_parts[-1] == canonical_text:
                self._pending_last_mono_ms = now_mono_ms
                self._pending_canonical_hits += hit_count
                return

            if not self._pending_canonical_parts:
                self._pending_started_mono_ms = now_mono_ms
            append_raw = raw_clean
            append_canonical = canonical_text
            if aggregate_mode == "v2" and self.config.final_aggregate_overlap_dedup and self._pending_canonical_parts:
                append_raw = _trim_leading_overlap(self._pending_raw_parts[-1], raw_clean)
                append_canonical = _trim_leading_overlap(self._pending_canonical_parts[-1], canonical_text)
            if append_raw:
                self._pending_raw_parts.append(append_raw)
            if append_canonical:
                self._pending_canonical_parts.append(append_canonical)
            self._pending_canonical_hits += hit_count
            self._pending_last_mono_ms = now_mono_ms
            total_chars = sum(len(p) for p in self._pending_canonical_parts)

            reason: Optional[str] = None
            if self.config.final_aggregate_flush_on_punct and _ends_with_sentence_punct(append_canonical or canonical_text):
                reason = "sentence_punct"
            elif total_chars >= max_chars:
                reason = "max_chars"
            elif (
                max_window_ms > 0
                and self._pending_started_mono_ms is not None
                and (now_mono_ms - self._pending_started_mono_ms) >= max_window_ms
            ):
                reason = "max_window"
            elif (
                force_emit_ms > 0
                and self._pending_started_mono_ms is not None
                and (now_mono_ms - self._pending_started_mono_ms) >= force_emit_ms
            ):
                reason = "force_emit"

            if reason is not None:
                if aggregate_mode == "v2":
                    force_reasons = {"force_emit", "max_window", "max_chars", "sentence_punct"}
                    if total_chars < min_chars and reason not in force_reasons:
                        return
                self._flush_pending_locked(now_mono_ms=now_mono_ms, reason=reason)

    def _flush_pending_if_due(self, now_mono_ms: int, force: bool = False) -> None:
        if not self.config.final_aggregate_enabled:
            return
        with self._aggregate_lock:
            if not self._pending_canonical_parts:
                return
            if force:
                self._flush_pending_locked(now_mono_ms=now_mono_ms, reason="force_close")
                return
            gap_ms = int(max(0.0, float(self.config.final_aggregate_gap_sec)) * 1000)
            max_window_ms = int(max(0.0, float(self.config.final_aggregate_max_sec)) * 1000)
            force_emit_ms = int(max(0.0, float(self.config.final_aggregate_force_emit_sec)) * 1000)
            min_chars = max(1, int(self.config.final_aggregate_min_chars))
            aggregate_mode = str(self.config.final_aggregate_mode).strip().lower()

            idle_timeout = (
                self._pending_last_mono_ms is not None
                and gap_ms > 0
                and (now_mono_ms - self._pending_last_mono_ms) >= gap_ms
            )
            over_window = (
                self._pending_started_mono_ms is not None
                and max_window_ms > 0
                and (now_mono_ms - self._pending_started_mono_ms) >= max_window_ms
            )
            force_emit = (
                self._pending_started_mono_ms is not None
                and force_emit_ms > 0
                and (now_mono_ms - self._pending_started_mono_ms) >= force_emit_ms
            )
            total_chars = sum(len(p) for p in self._pending_canonical_parts)
            if force_emit:
                self._flush_pending_locked(now_mono_ms=now_mono_ms, reason="force_emit")
                return
            if over_window:
                self._flush_pending_locked(now_mono_ms=now_mono_ms, reason="max_window")
                return
            if idle_timeout:
                if aggregate_mode == "v2" and total_chars < min_chars:
                    return
                self._flush_pending_locked(now_mono_ms=now_mono_ms, reason="idle_gap")

    def _flush_pending_locked(self, now_mono_ms: int, reason: str) -> None:
        if not self._pending_canonical_parts:
            return
        raw_text = _merge_final_parts(self._pending_raw_parts)
        canonical_text = _merge_final_parts(self._pending_canonical_parts)
        fragment_count = len(self._pending_canonical_parts)
        canonical_hit_count = int(self._pending_canonical_hits)
        self._pending_raw_parts = []
        self._pending_canonical_parts = []
        self._pending_canonical_hits = 0
        self._pending_started_mono_ms = None
        self._pending_last_mono_ms = None
        if canonical_text:
            self._emit_final_cue(
                raw_text=raw_text or canonical_text,
                canonical_text=canonical_text,
                now_mono_ms=now_mono_ms,
                aggregator_reason=reason,
                fragment_count=fragment_count,
                canonical_hit_count=canonical_hit_count,
            )

    def _emit_final_cue(
        self,
        *,
        raw_text: str,
        canonical_text: str,
        now_mono_ms: int,
        aggregator_reason: str,
        fragment_count: int,
        canonical_hit_count: int,
    ) -> None:
        cue = SourceCue(
            source_key=self._next_source_key(now_mono_ms),
            source_kind="speechmatics",
            srt_index=None,
            start_ms=None,
            end_ms=None,
            jp_raw=raw_text,
            jp_aggregated=canonical_text,
            jp_canonicalized=canonical_text,
            aggregator_reason=aggregator_reason,
        )
        self._queue.put(cue)
        self._observe_asr_final(now_mono_ms)
        self._observe_asr_source_emit(
            emitted_count=1,
            fragment_count=fragment_count,
            canonical_hit_count=canonical_hit_count,
            now_mono_ms=now_mono_ms,
        )

    def _build_additional_vocab(self, limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        source_mode = normalize_speechmatics_vocab_source(self.config.additional_vocab_from)
        entries: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def add_entry(content: str, sounds_like: Optional[List[str]] = None) -> bool:
            term = content.strip()
            if not term or term in seen:
                return False
            seen.add(term)
            payload: Dict[str, Any] = {"content": term}
            if sounds_like:
                payload["sounds_like"] = [s for s in sounds_like if s]
            entries.append(payload)
            return len(entries) >= limit

        if source_mode in ("names_whitelist", "names_whitelist_glossary"):
            for item in _load_names_vocab(self.names_path):
                if add_entry(str(item.get("content", "")), item.get("sounds_like")):
                    return entries

        if source_mode in ("glossary", "names_whitelist_glossary"):
            glossary = _load_glossary(self.glossary_path)
            for key in glossary.keys():
                if add_entry(key):
                    return entries

        return entries

    def _set_asr_connected(self, connected: bool) -> None:
        if self.runtime is None:
            return
        updater = getattr(self.runtime, "set_asr_connected", None)
        if callable(updater):
            updater(connected)

    def _set_asr_error(self, message: Optional[str]) -> None:
        if self.runtime is None:
            return
        updater = getattr(self.runtime, "set_asr_error", None)
        if callable(updater):
            updater(message)

    def _observe_asr_partial(self, text: str) -> None:
        if self.runtime is None:
            return
        updater = getattr(self.runtime, "observe_asr_partial", None)
        if callable(updater):
            updater(text=text, now_mono_ms=int(time.monotonic() * 1000))

    def _observe_asr_final(self, now_mono_ms: int) -> None:
        if self.runtime is None:
            return
        updater = getattr(self.runtime, "observe_asr_final", None)
        if callable(updater):
            updater(now_mono_ms=now_mono_ms)

    def _observe_asr_source_emit(
        self,
        *,
        emitted_count: int,
        fragment_count: int,
        canonical_hit_count: int,
        now_mono_ms: int,
    ) -> None:
        if self.runtime is None:
            return
        updater = getattr(self.runtime, "observe_asr_source_emit", None)
        if callable(updater):
            updater(
                emitted_count=emitted_count,
                fragment_count=fragment_count,
                canonical_hit_count=canonical_hit_count,
                now_mono_ms=now_mono_ms,
            )

    def _terminate_capture_proc_sync(self) -> None:
        with self._capture_proc_lock:
            proc = self._capture_proc
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.1)
        if proc.returncode is None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def _extract_transcript_text(transcript_result_cls: Any, message: Any) -> str:
    try:
        parsed = transcript_result_cls.from_message(message)
        text = str(getattr(parsed.metadata, "transcript", "") or "").strip()
        return text
    except Exception:  # noqa: BLE001
        if isinstance(message, dict):
            metadata = message.get("metadata", {})
            if isinstance(metadata, dict):
                return str(metadata.get("transcript", "") or "").strip()
        return ""


_ASCII_TAIL_RE = re.compile(r"[A-Za-z0-9]$")
_ASCII_HEAD_RE = re.compile(r"^[A-Za-z0-9]")


def _merge_final_parts(parts: List[str]) -> str:
    merged = ""
    for raw in parts:
        part = raw.strip()
        if not part:
            continue
        if not merged:
            merged = part
            continue
        if _ASCII_TAIL_RE.search(merged) and _ASCII_HEAD_RE.search(part):
            merged = f"{merged} {part}"
        else:
            merged = f"{merged}{part}"
    return merged.strip()


def _ends_with_sentence_punct(text: str) -> bool:
    if not text:
        return False
    tail = text.rstrip()
    if not tail:
        return False
    return tail[-1] in {"。", "！", "？", ".", "!", "?", "…"}


def _trim_leading_overlap(previous: str, current: str, max_overlap: int = 32) -> str:
    if not previous or not current:
        return current
    size = min(len(previous), len(current), max(1, int(max_overlap)))
    for n in range(size, 0, -1):
        if previous[-n:] == current[:n]:
            return current[n:]
    return current


@dataclass(frozen=True)
class CanonicalRule:
    sounds_like: str
    canonical: str


class NameCanonicalizer:
    def __init__(self, *, names_path: Path, enabled: bool, max_rules: int):
        self.enabled = bool(enabled)
        self.max_rules = max(0, int(max_rules))
        self.rules: List[CanonicalRule] = []
        if not self.enabled or self.max_rules <= 0:
            return
        vocab_items = _load_names_vocab(names_path)
        rules: List[CanonicalRule] = []
        for item in vocab_items:
            canonical = str(item.get("content", "")).strip()
            if not canonical:
                continue
            for sound in item.get("sounds_like") or []:
                source = str(sound).strip()
                if not source or source == canonical:
                    continue
                rules.append(CanonicalRule(sounds_like=source, canonical=canonical))
        rules.sort(key=lambda r: len(r.sounds_like), reverse=True)
        self.rules = rules[: self.max_rules]

    def apply(self, text: str) -> tuple[str, int]:
        if not self.enabled or not self.rules:
            return text, 0
        updated = text
        total_hits = 0
        for rule in self.rules:
            hit = updated.count(rule.sounds_like)
            if hit <= 0:
                continue
            total_hits += hit
            updated = updated.replace(rule.sounds_like, rule.canonical)
        return updated, total_hits


def _load_speechmatics_sdk() -> Dict[str, Any]:
    try:
        from speechmatics.rt import (  # type: ignore
            AsyncClient,
            AudioEncoding,
            AudioFormat,
            ServerMessageType,
            TranscriptResult,
            TranscriptionConfig,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Speechmatics mode requires 'speechmatics-rt'. Install with: "
            "pip install speechmatics-rt"
        ) from exc

    return {
        "AsyncClient": AsyncClient,
        "TranscriptionConfig": TranscriptionConfig,
        "AudioFormat": AudioFormat,
        "AudioEncoding": AudioEncoding,
        "ServerMessageType": ServerMessageType,
        "TranscriptResult": TranscriptResult,
    }


async def _terminate_subprocess(proc: asyncio.subprocess.Process) -> None:
    try:
        if proc.returncode is None:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
        try:
            await asyncio.wait_for(proc.communicate(), timeout=0.5)
        except Exception:  # noqa: BLE001
            pass
    finally:
        for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            transport = getattr(stream, "_transport", None)
            if transport is not None:
                try:
                    transport.close()
                except Exception:  # noqa: BLE001
                    pass


def _load_glossary(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
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
            mapping[ja] = zh
    return mapping


def _load_names_vocab(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
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
        if not content:
            continue
        item: Dict[str, Any] = {"content": content}
        sounds_like: List[str] = []
        if sounds_like_raw:
            normalized = sounds_like_raw.replace(chr(0xFF0C), ",").replace(chr(0x3001), ",")
            for token in normalized.split(","):
                value = token.strip()
                if value:
                    sounds_like.append(value)
        if sounds_like:
            item["sounds_like"] = sounds_like
        items.append(item)
    return items
