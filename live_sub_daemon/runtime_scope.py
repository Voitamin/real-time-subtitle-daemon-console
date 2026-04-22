from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import AppConfig


@dataclass(frozen=True)
class QueryScope:
    allowed_source_kinds: Optional[List[str]]
    updated_after_unix: Optional[float]
    freshness_sec: float


def active_source_kinds(config: AppConfig) -> Optional[List[str]]:
    if not bool(config.runtime_scope.enabled):
        return None
    if not bool(config.runtime_scope.only_active_source_kind):
        return None
    mode = str(config.source.mode or "").strip().lower()
    if mode == "speechmatics":
        return ["speechmatics"]
    if mode == "file":
        return ["srt", "txt"]
    return None


def resolve_scope_freshness_sec(config: AppConfig, effective_delay_sec: float) -> float:
    configured = max(0.0, float(config.runtime_scope.freshness_sec))
    dynamic = max(0.0, float(effective_delay_sec) * 12.0)
    return max(configured, dynamic)


def build_query_scope(
    config: AppConfig,
    *,
    effective_delay_sec: float,
    now_unix: Optional[float] = None,
) -> QueryScope:
    kinds = active_source_kinds(config)
    freshness_sec = resolve_scope_freshness_sec(config, effective_delay_sec)
    if now_unix is None:
        now_unix = time.time()
    updated_after = None
    if freshness_sec > 0:
        updated_after = float(now_unix) - float(freshness_sec)
    return QueryScope(
        allowed_source_kinds=kinds,
        updated_after_unix=updated_after,
        freshness_sec=freshness_sec,
    )


def normalize_scope_kinds_for_json(value: Optional[Sequence[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    out: List[str] = []
    seen: set[str] = set()
    for raw in value:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out or None
