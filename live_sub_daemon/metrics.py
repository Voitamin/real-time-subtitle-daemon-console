from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

from .state_store import StateStore


@dataclass
class DelaySuggestion:
    p95_sec: float
    suggested_delay_sec: float


class MetricsReporter:
    def __init__(self, state: StateStore):
        self.state = state

    def summarize(self) -> Dict[str, float]:
        correct = self.state.fetch_recent_metric_latencies("correct", 200)
        translate = self.state.fetch_recent_metric_latencies("translate", 200)
        total = self.state.fetch_recent_metric_latencies("pipeline_total", 200)
        return {
            "correct_p95_ms": _p95(correct),
            "translate_p95_ms": _p95(translate),
            "pipeline_p95_ms": _p95(total),
        }

    def suggest_delay(self, safety_margin_sec: float = 3.0) -> DelaySuggestion:
        total = self.state.fetch_recent_metric_latencies("pipeline_total", 200)
        p95_ms = _p95(total)
        p95_sec = p95_ms / 1000.0
        return DelaySuggestion(p95_sec=p95_sec, suggested_delay_sec=p95_sec + safety_margin_sec)


def _p95(values: List[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, int(len(ordered) * 0.95) - 1)
    return float(ordered[idx])
