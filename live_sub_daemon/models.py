from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceCue:
    source_key: str
    source_kind: str
    srt_index: Optional[int]
    start_ms: Optional[int]
    end_ms: Optional[int]
    jp_raw: str
    jp_aggregated: Optional[str] = None
    jp_canonicalized: Optional[str] = None
    aggregator_reason: Optional[str] = None


@dataclass
class CueRecord:
    source_key: str
    source_kind: str
    srt_index: Optional[int]
    start_ms: Optional[int]
    end_ms: Optional[int]
    jp_raw: str
    jp_aggregated: Optional[str]
    jp_canonicalized: Optional[str]
    jp_corrected: Optional[str]
    zh_text: Optional[str]
    status: str
    t_seen_mono_ms: int
    due_mono_ms: int
    translated_mono_ms: Optional[int]
    dropped_late: bool
    llm_latency_ms: Optional[int]
    last_error: Optional[str] = None
    context_miss: bool = False
    inflight_owner: Optional[str] = None
    inflight_since_mono_ms: Optional[int] = None
    stage1_provider: Optional[str] = None
    stage1_model: Optional[str] = None
    stage2_provider: Optional[str] = None
    stage2_model: Optional[str] = None
    fallback_used: bool = False
    stage1_latency_ms: Optional[int] = None
    stage2_latency_ms: Optional[int] = None
    s1_skipped: bool = False
    aggregator_reason: Optional[str] = None
    manual_zh_text: Optional[str] = None
    manual_locked: bool = False
    deleted_soft: bool = False
    display_suppressed: bool = False
    join_target_source_key: Optional[str] = None
    displayed_at_mono_ms: Optional[int] = None
    updated_by: Optional[str] = None
    revision: int = 0
