from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import List, Optional, Sequence

from .io_utils import atomic_write_text
from .models import CueRecord
from .state_store import StateStore


@dataclass
class RenderDecision:
    text: str
    cue_keys: List[str]


@dataclass
class _LineSlice:
    source_key: str
    text: str
    is_first_slice: bool


class Renderer:
    def __init__(
        self,
        state: StateStore,
        zh_txt_path: Path,
        zh_srt_path: Path,
        char_threshold: int,
        max_total_chars: int = 0,
        max_lines: int = 2,
        *,
        two_line_roll_enabled: bool = True,
        min_hold_sec: float = 1.6,
        target_cps: float = 13.0,
        max_hold_sec: float = 4.0,
        backlog_relax_threshold: int = 10,
        backlog_relaxed_min_hold_sec: float = 1.0,
        allowed_source_kinds: Optional[Sequence[str]] = None,
        scope_freshness_sec: float = 0.0,
    ):
        self.state = state
        self.zh_txt_path = zh_txt_path
        self.zh_srt_path = zh_srt_path
        self.char_threshold = char_threshold
        self.max_total_chars = max(0, int(max_total_chars))
        self.max_lines = max(1, int(max_lines))
        self.two_line_roll_enabled = bool(two_line_roll_enabled)
        self.min_hold_sec = max(0.1, float(min_hold_sec))
        self.target_cps = max(1.0, float(target_cps))
        self.max_hold_sec = max(self.min_hold_sec, float(max_hold_sec))
        self.backlog_relax_threshold = max(0, int(backlog_relax_threshold))
        self.backlog_relaxed_min_hold_sec = max(
            0.1, min(self.min_hold_sec, float(backlog_relaxed_min_hold_sec))
        )
        self.allowed_source_kinds = [str(v).strip().lower() for v in (allowed_source_kinds or []) if str(v).strip()]
        self.scope_freshness_sec = max(0.0, float(scope_freshness_sec))

        self._line_queue: List[_LineSlice] = []
        self._queued_keys: set[str] = set()
        self._next_shift_mono_ms = 0
        self._last_render_text = ""
        self._jump_seq = ""
        self._lock = threading.Lock()

    def tick(self, now_mono_ms: int, delay_adjust_ms: int = 0) -> RenderDecision:
        with self._lock:
            self._sync_jump_signal()
            self._enqueue_due_cues(now_mono_ms=now_mono_ms, delay_adjust_ms=delay_adjust_ms)
            self._advance_window_if_needed(now_mono_ms=now_mono_ms)
            decision = self._current_decision()
            if decision.text:
                self._last_render_text = decision.text
            if decision.cue_keys:
                self.state.mark_displayed(decision.cue_keys, displayed_at_mono_ms=now_mono_ms)
                for key in decision.cue_keys:
                    self._queued_keys.discard(key)

        atomic_write_text(self.zh_txt_path, decision.text)
        srt_text = self._build_zh_srt_text()
        atomic_write_text(self.zh_srt_path, srt_text)
        return decision

    def _sync_jump_signal(self) -> None:
        seq = self.state.get_meta("render_jump_to_latest_seq") or ""
        if seq == self._jump_seq:
            return
        self._jump_seq = seq
        self._line_queue = []
        self._queued_keys.clear()
        self._next_shift_mono_ms = 0

    def _scope_updated_after_unix(self) -> Optional[float]:
        if self.scope_freshness_sec <= 0:
            return None
        return time.time() - self.scope_freshness_sec

    def _line_char_capacity(self) -> int:
        if self.max_total_chars <= 0:
            return 0
        return max(1, int(math.floor(self.max_total_chars / max(1, self.max_lines))))

    def _enqueue_due_cues(self, now_mono_ms: int, delay_adjust_ms: int) -> None:
        due = self.state.fetch_due_unshown_cues(
            now_mono_ms=now_mono_ms,
            limit=64,
            delay_adjust_ms=delay_adjust_ms,
            allowed_source_kinds=self.allowed_source_kinds or None,
            updated_after_unix=self._scope_updated_after_unix(),
        )
        line_cap = self._line_char_capacity()
        for cue in due:
            if cue.source_key in self._queued_keys:
                continue
            lines = split_text_to_lines(_cue_display_text(cue), line_char_cap=line_cap)
            if not lines:
                continue
            for idx, line in enumerate(lines):
                self._line_queue.append(
                    _LineSlice(
                        source_key=cue.source_key,
                        text=line,
                        is_first_slice=(idx == 0),
                    )
                )
            self._queued_keys.add(cue.source_key)
            if self._next_shift_mono_ms <= 0:
                self._next_shift_mono_ms = now_mono_ms + self._resolve_hold_ms(self._line_queue[0].text)

    def _resolve_hold_ms(self, top_line_text: str) -> int:
        backlog = len(self._line_queue)
        min_hold = (
            self.backlog_relaxed_min_hold_sec
            if backlog > self.backlog_relax_threshold
            else self.min_hold_sec
        )
        chars = max(1, len(str(top_line_text).strip()))
        hold_sec = chars / self.target_cps
        hold_sec = max(min_hold, min(self.max_hold_sec, hold_sec))
        return int(round(hold_sec * 1000.0))

    def _advance_window_if_needed(self, now_mono_ms: int) -> None:
        if not self._line_queue:
            self._next_shift_mono_ms = 0
            return
        if self._next_shift_mono_ms <= 0:
            self._next_shift_mono_ms = now_mono_ms + self._resolve_hold_ms(self._line_queue[0].text)
            return
        if now_mono_ms < self._next_shift_mono_ms:
            return
        if not self.two_line_roll_enabled:
            if len(self._line_queue) >= 2:
                self._line_queue.pop(0)
        else:
            if len(self._line_queue) >= 2:
                self._line_queue.pop(0)
        if self._line_queue:
            self._next_shift_mono_ms = now_mono_ms + self._resolve_hold_ms(self._line_queue[0].text)
        else:
            self._next_shift_mono_ms = 0

    def _current_decision(self) -> RenderDecision:
        if not self._line_queue:
            return RenderDecision(text=self._last_render_text, cue_keys=[])

        top = self._line_queue[0]
        if not self.two_line_roll_enabled:
            return RenderDecision(
                text=top.text,
                cue_keys=[top.source_key] if top.is_first_slice else [],
            )

        bottom = self._line_queue[1] if len(self._line_queue) > 1 else None
        cue_keys: List[str] = []
        if top.is_first_slice:
            cue_keys.append(top.source_key)
        if bottom is not None and bottom.is_first_slice and bottom.source_key not in cue_keys:
            cue_keys.append(bottom.source_key)
        if bottom is None:
            return RenderDecision(text=top.text, cue_keys=cue_keys)
        return RenderDecision(text=f"{top.text}\n{bottom.text}", cue_keys=cue_keys)

    def _build_zh_srt_text(self) -> str:
        cues = self.state.fetch_srt_ready(
            allowed_source_kinds=self.allowed_source_kinds or None,
            updated_after_unix=self._scope_updated_after_unix(),
        )
        chunks: List[str] = []
        for cue in cues:
            if cue.srt_index is None or cue.start_ms is None or cue.end_ms is None:
                continue
            text = _cue_display_text(cue)
            chunks.append(
                "\n".join(
                    [
                        str(cue.srt_index),
                        f"{_format_srt_ts(cue.start_ms)} --> {_format_srt_ts(cue.end_ms)}",
                        text,
                    ]
                )
            )
        if not chunks:
            return ""
        return "\n\n".join(chunks) + "\n"


def build_render_decision(
    candidates_desc: List[CueRecord],
    char_threshold: int,
) -> RenderDecision:
    _ = char_threshold
    if not candidates_desc:
        return RenderDecision(text="", cue_keys=[])
    latest = candidates_desc[0]
    latest_text = _cue_display_text(latest).strip()
    return RenderDecision(text=latest_text, cue_keys=[latest.source_key])


def _cue_display_text(cue: CueRecord) -> str:
    if cue.manual_zh_text:
        return cue.manual_zh_text
    if cue.zh_text is not None:
        return cue.zh_text
    return cue.jp_raw


def split_text_to_lines(text: str, line_char_cap: int) -> List[str]:
    clean_lines = [line.strip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not clean_lines:
        return []
    if line_char_cap <= 0:
        return clean_lines
    out: List[str] = []
    for line in clean_lines:
        cursor = 0
        while cursor < len(line):
            out.append(line[cursor : cursor + line_char_cap])
            cursor += line_char_cap
    return out


def paginate_overlay_text(text: str, max_total_chars: int, max_lines: int) -> List[str]:
    # Legacy helper retained for tests/tools; renderer now uses split_text_to_lines + two-line rolling.
    clean_lines = [line.strip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if not clean_lines:
        return []
    max_lines = max(1, int(max_lines))
    plain = "".join(clean_lines)
    if not plain:
        return []

    if max_total_chars <= 0:
        return ["\n".join(clean_lines[:max_lines])]

    window_chars = max(1, int(max_total_chars))
    per_line = max(1, int(math.ceil(window_chars / max_lines)))
    pages: List[str] = []
    cursor = 0
    while cursor < len(plain):
        chunk = plain[cursor : cursor + window_chars]
        lines: List[str] = []
        line_cursor = 0
        while line_cursor < len(chunk) and len(lines) < max_lines:
            lines.append(chunk[line_cursor : line_cursor + per_line])
            line_cursor += per_line
        pages.append("\n".join(lines))
        cursor += window_chars
    return pages


def _format_srt_ts(ms: int) -> str:
    hh = ms // 3600000
    mm = (ms % 3600000) // 60000
    ss = (ms % 60000) // 1000
    mmm = ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}"
