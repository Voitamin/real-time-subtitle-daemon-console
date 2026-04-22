from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from .config import AppConfig


@dataclass(frozen=True)
class WorkerPlan:
    mode: str
    batch_max_wait_sec: float
    batch_max_lines: int
    context_wait_timeout_sec: float
    context_window: int
    use_fast_model: bool
    contextless_only: bool
    max_retries: int
    s1_mode: str
    s1_lite_punctuation_only: bool


class RuntimeController:
    DELAY_ADJUST_MIN_SEC = -60
    DELAY_ADJUST_MAX_SEC = 60

    def __init__(self, config: AppConfig):
        self._cfg = config
        self._lock = threading.Lock()
        self._window_sec = 60.0
        self._arrivals: Deque[Tuple[float, int]] = deque()
        self._completions: Deque[Tuple[float, int, int]] = deque()  # ts, total, late
        self._stage_lat: Dict[str, Deque[Tuple[float, int]]] = {
            "correct": deque(),
            "translate": deque(),
            "pipeline": deque(),
        }
        self._translate_routes: Deque[Tuple[float, int, int, int, int]] = deque()  # ts, primary_ok, fallback_used, fallback_ok, fast_used
        self._asr_source_events: Deque[Tuple[float, int, int, int]] = deque()  # ts, emitted, fragments, canonical_hits
        self._s1_skipped_events: Deque[Tuple[float, int]] = deque()  # ts, count
        self._s1_join_events: Deque[Tuple[float, int, int]] = deque()  # ts, total, suppressed
        self._render_frames: Deque[Tuple[float, int]] = deque()  # ts, second_line_used
        self._last_fast_blocked_by_latest_ready = False

        self._forced_mode = config.adaptive.mode if config.adaptive.mode != "auto" else "auto"
        self._mode = "green"
        self._overrides: Dict[str, Any] = {}
        self._asr_state: Dict[str, Any] = {
            "asr_provider": "speechmatics" if config.source.mode == "speechmatics" else "file",
            "asr_connected": True if config.source.mode == "file" else False,
            "asr_last_final_mono_ms": None,
            "asr_last_error": None,
        }
        self._last_status: Dict[str, Any] = {
            "mode": self._mode,
            "slack_ms": None,
            "arrival_rate_lps": 0.0,
            "service_rate_lps": 0.0,
            "base_delay_sec": float(config.align.delay_sec),
            "asr_delay_sec": float(config.align.asr_delay_sec),
            "delay_adjust_sec": 0.0,
            "effective_delay_sec": max(0.0, float(config.align.delay_sec) - float(config.align.asr_delay_sec)),
            "latest_ready_source_key": None,
            "latest_ready_due_effective_mono_ms": None,
            "latest_ready_remaining_ms": None,
            "latest_ready_anomaly": False,
            "alert_level": "green",
            "deadline_remaining_ms": None,
            "risk_ratio": None,
            "predict_percentile": int(config.adaptive_predict.pipeline_service_percentile),
            "warmup_active": False,
            "fast_blocked_by_latest_ready": False,
            "stage_route": {
                "correct_provider": config.llm.correct_provider,
                "translate_provider": config.llm.translate_provider,
                "translate_fallback_provider": config.llm.translate_fallback_provider,
                "s1_mode": config.pipeline.s1_mode,
            },
            "translate_primary_ok_recent": 0,
            "translate_fallback_count_recent": 0,
            "translate_fallback_ratio_recent": 0.0,
            "translate_fallback_success_count_recent": 0,
            "translate_fallback_success_rate_recent": 0.0,
            "translate_fast_count_recent": 0,
            "translate_fast_ratio_recent": 0.0,
            "asr_aggregate_emit_count_recent": 0,
            "asr_aggregate_fragment_count_recent": 0,
            "asr_canonical_hit_count_recent": 0,
            "s1_skipped_count_recent": 0,
            "s1_join_suppressed_count_recent": 0,
            "s1_join_total_count_recent": 0,
            "s1_join_ratio_recent": 0.0,
            "render_second_line_used_recent": 0,
            "render_frame_count_recent": 0,
            "line_utilization_ratio_recent": 0.0,
            **self._asr_state,
        }

    def observe_arrival(self, count: int, now_mono_ms: int) -> None:
        if count <= 0:
            return
        now = now_mono_ms / 1000.0
        with self._lock:
            self._arrivals.append((now, int(count)))
            self._trim(now)

    def observe_completion(
        self,
        total_count: int,
        late_count: int,
        stage1_ms: int,
        stage2_ms: int,
        pipeline_ms: int,
        now_mono_ms: int,
    ) -> None:
        now = now_mono_ms / 1000.0
        with self._lock:
            self._completions.append((now, max(0, int(total_count)), max(0, int(late_count))))
            self._stage_lat["correct"].append((now, max(0, int(stage1_ms))))
            self._stage_lat["translate"].append((now, max(0, int(stage2_ms))))
            self._stage_lat["pipeline"].append((now, max(0, int(pipeline_ms))))
            self._trim(now)

    def apply_runtime_updates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "batch_max_wait_sec",
            "batch_max_lines",
            "context_wait_timeout_sec",
            "adaptive_enabled",
            "mode",
            "green_slack_ms",
            "red_slack_ms",
            "backlog_red_threshold",
            "red_parallel_contextless",
            "service_rate_floor_lps",
            "tail_quiet_arrival_lps",
            "tail_small_backlog_threshold",
            "tail_red_guard",
            "s1_mode",
            "delay_adjust_sec",
        }
        with self._lock:
            for key, value in payload.items():
                if key in allowed:
                    if key == "delay_adjust_sec":
                        self._overrides[key] = self._clamp_delay_adjust(value)
                    else:
                        self._overrides[key] = value
            mode = self._overrides.get("mode")
            if isinstance(mode, str) and mode.strip().lower() in ("auto", "green", "yellow", "red"):
                self._forced_mode = mode.strip().lower()
            return dict(self._overrides)

    def observe_translate_route(
        self,
        *,
        primary_ok: bool,
        fallback_used: bool,
        fallback_ok: bool,
        used_fast_model: bool,
        now_mono_ms: int,
    ) -> None:
        now = now_mono_ms / 1000.0
        with self._lock:
            self._translate_routes.append(
                (now, int(primary_ok), int(fallback_used), int(fallback_ok), int(used_fast_model))
            )
            self._trim(now)

    def resolve_fast_model_usage(
        self,
        *,
        requested_fast: bool,
        queue_stats: Optional[Dict[str, Optional[int]]] = None,
    ) -> tuple[bool, bool]:
        with self._lock:
            overdue_unfinished_count = 0
            if queue_stats is not None:
                overdue_unfinished_count = int(queue_stats.get("overdue_unfinished_count") or 0)
            effective_delay_ms = max(1, self._effective_delay_ms_locked())
            latest_ready_remaining_ms = self._last_status.get("latest_ready_remaining_ms")
            allow_fast_if_overdue = bool(self._cfg.adaptive_fast.allow_fast_if_overdue)
            block_ratio = float(self._cfg.adaptive_fast.block_when_latest_ready_ratio)
            emergency_ratio = float(self._cfg.adaptive_fast.emergency_when_latest_ready_ratio)

            blocked_by_latest_ready = False
            if allow_fast_if_overdue and overdue_unfinished_count > 0:
                self._last_fast_blocked_by_latest_ready = False
                return True, False

            if isinstance(latest_ready_remaining_ms, (int, float)):
                block_threshold = int(block_ratio * effective_delay_ms)
                emergency_threshold = int(emergency_ratio * effective_delay_ms)
                if int(latest_ready_remaining_ms) >= block_threshold:
                    blocked_by_latest_ready = True
                    self._last_fast_blocked_by_latest_ready = True
                    return False, True
                if int(latest_ready_remaining_ms) < emergency_threshold:
                    self._last_fast_blocked_by_latest_ready = False
                    return True, False

            self._last_fast_blocked_by_latest_ready = False
            return bool(requested_fast), blocked_by_latest_ready

    def observe_asr_source_emit(
        self,
        *,
        emitted_count: int,
        fragment_count: int,
        canonical_hit_count: int,
        now_mono_ms: int,
    ) -> None:
        now = now_mono_ms / 1000.0
        with self._lock:
            self._asr_source_events.append(
                (
                    now,
                    max(0, int(emitted_count)),
                    max(0, int(fragment_count)),
                    max(0, int(canonical_hit_count)),
                )
            )
            self._trim(now)

    def observe_s1_skipped(self, *, count: int, now_mono_ms: int) -> None:
        if count <= 0:
            return
        now = now_mono_ms / 1000.0
        with self._lock:
            self._s1_skipped_events.append((now, int(count)))
            self._trim(now)

    def observe_s1_join(self, *, total_count: int, suppressed_count: int, now_mono_ms: int) -> None:
        if total_count <= 0:
            return
        now = now_mono_ms / 1000.0
        with self._lock:
            self._s1_join_events.append(
                (
                    now,
                    max(0, int(total_count)),
                    max(0, min(int(suppressed_count), int(total_count))),
                )
            )
            self._trim(now)

    def observe_render_frame(self, *, second_line_used: bool, now_mono_ms: int) -> None:
        now = now_mono_ms / 1000.0
        with self._lock:
            self._render_frames.append((now, int(bool(second_line_used))))
            self._trim(now)

    def set_mode(self, mode: str) -> str:
        normalized = str(mode).strip().lower()
        if normalized not in ("auto", "green", "yellow", "red"):
            raise ValueError(f"invalid mode: {mode}")
        with self._lock:
            self._forced_mode = normalized
            self._overrides["mode"] = normalized
            if normalized != "auto":
                self._mode = normalized
        return normalized

    def update_and_get_status(self, queue_stats: Dict[str, Optional[int]], now_mono_ms: int) -> Dict[str, Any]:
        now = now_mono_ms / 1000.0
        with self._lock:
            self._trim(now)
            arrival_rate = self._rate_from_pairs(self._arrivals, now)
            service_rate = self._rate_from_completion(self._completions, now)
            service_rate_floor = max(
                1e-3,
                float(self._overrides.get("service_rate_floor_lps", self._cfg.adaptive.service_rate_floor_lps)),
            )
            service_rate_effective = max(service_rate, service_rate_floor)

            new_count = int(queue_stats.get("new_count") or 0)
            inflight_count = int(queue_stats.get("inflight_count") or 0)
            unfinished = int(queue_stats.get("unfinished_count") or 0)
            batch_lines = self._effective_batch_lines_locked(self._mode)
            done_total = sum(total for _, total, _ in self._completions)
            predict_percentile = int(self._cfg.adaptive_predict.pipeline_service_percentile)
            pipeline_px = _percentile([v for _, v in self._stage_lat["pipeline"]], predict_percentile)
            tail_denoise_active = self._tail_denoise_active_locked(
                arrival_rate=arrival_rate,
                new_count=new_count,
                unfinished=unfinished,
            )
            warmup_active = done_total < max(0, int(self._cfg.adaptive_predict.warmup_min_completions))
            if tail_denoise_active:
                predicted_queue_wait_ms = int((new_count / service_rate_effective) * 1000) if new_count > 0 else 0
                outstanding = max(1, min(max(inflight_count, new_count, 1), batch_lines))
                predicted_service_ms = int(max(pipeline_px, (outstanding / service_rate_effective) * 1000))
            else:
                if bool(self._cfg.adaptive_predict.queue_exclude_current_batch):
                    effective_backlog = max(0, int(unfinished) - int(batch_lines))
                else:
                    effective_backlog = max(0, int(unfinished))
                predicted_queue_wait_ms = int((effective_backlog / service_rate_effective) * 1000) if effective_backlog > 0 else 0
                predicted_service_ms = int(max(pipeline_px, (batch_lines / service_rate_effective) * 1000))
            if warmup_active:
                predicted_service_ms = max(
                    int(predicted_service_ms),
                    int(self._cfg.adaptive_predict.warmup_nominal_pipeline_ms),
                )
            predicted_total_ms = predicted_queue_wait_ms + predicted_service_ms

            min_due = queue_stats.get("min_due_unfinished_mono_ms")
            slack_ms = None
            deadline_remaining_ms = None
            risk_ratio = None
            if min_due is not None:
                deadline_remaining_ms = int(min_due) - int(now_mono_ms)
                slack_ms = int(deadline_remaining_ms) - predicted_total_ms
                risk_ratio = float(predicted_total_ms / max(int(deadline_remaining_ms), 1))
            elif unfinished <= 0:
                risk_ratio = 0.0

            if self._is_adaptive_enabled_locked():
                self._update_mode_locked(
                    risk_ratio=risk_ratio,
                    unfinished=unfinished,
                    arrival_rate=arrival_rate,
                    new_count=new_count,
                    tail_denoise_active=tail_denoise_active,
                )

            late_total = sum(late for _, _, late in self._completions)
            late_rate = (late_total / done_total) if done_total > 0 else 0.0
            route_count_recent = len(self._translate_routes)
            primary_ok_recent = sum(primary_ok for _, primary_ok, _, _, _ in self._translate_routes)
            fallback_count_recent = sum(fallback_used for _, _, fallback_used, _, _ in self._translate_routes)
            fallback_success_count_recent = sum(fallback_ok for _, _, _, fallback_ok, _ in self._translate_routes)
            fallback_success_rate = (
                (fallback_success_count_recent / fallback_count_recent) if fallback_count_recent > 0 else 0.0
            )
            fallback_ratio_recent = (fallback_count_recent / route_count_recent) if route_count_recent > 0 else 0.0
            fast_count_recent = sum(fast_used for _, _, _, _, fast_used in self._translate_routes)
            fast_ratio_recent = (fast_count_recent / route_count_recent) if route_count_recent > 0 else 0.0
            asr_aggregate_emit_count_recent = sum(emitted for _, emitted, _, _ in self._asr_source_events)
            asr_aggregate_fragment_count_recent = sum(fragments for _, _, fragments, _ in self._asr_source_events)
            asr_canonical_hit_count_recent = sum(hits for _, _, _, hits in self._asr_source_events)
            s1_skipped_count_recent = sum(count for _, count in self._s1_skipped_events)
            s1_join_total_count_recent = sum(total for _, total, _ in self._s1_join_events)
            s1_join_suppressed_count_recent = sum(suppressed for _, _, suppressed in self._s1_join_events)
            s1_join_ratio_recent = (
                (s1_join_suppressed_count_recent / s1_join_total_count_recent)
                if s1_join_total_count_recent > 0
                else 0.0
            )
            render_frame_count_recent = len(self._render_frames)
            render_second_line_used_recent = sum(v for _, v in self._render_frames)
            line_utilization_ratio_recent = (
                (render_second_line_used_recent / render_frame_count_recent)
                if render_frame_count_recent > 0
                else 0.0
            )

            stage = {
                "correct_p50_ms": _percentile([v for _, v in self._stage_lat["correct"]], 50),
                "correct_p95_ms": _percentile([v for _, v in self._stage_lat["correct"]], 95),
                "translate_p50_ms": _percentile([v for _, v in self._stage_lat["translate"]], 50),
                "translate_p95_ms": _percentile([v for _, v in self._stage_lat["translate"]], 95),
                "pipeline_p50_ms": _percentile([v for _, v in self._stage_lat["pipeline"]], 50),
                "pipeline_p95_ms": _percentile([v for _, v in self._stage_lat["pipeline"]], 95),
            }
            latest_ready_source_key = queue_stats.get("latest_ready_source_key")
            latest_ready_due_effective_mono_ms = queue_stats.get("latest_ready_due_effective_mono_ms")
            latest_ready_remaining_ms = queue_stats.get("latest_ready_remaining_ms")
            latest_ready_anomaly = bool(queue_stats.get("latest_ready_anomaly"))
            effective_delay_sec = max(
                0.0,
                self.base_delay_sec() - self.asr_delay_sec() + self._delay_adjust_sec_locked(),
            )
            if latest_ready_anomaly:
                alert_level = "yellow"
            elif latest_ready_remaining_ms is None:
                alert_level = "green" if unfinished <= 0 else "yellow"
            else:
                remaining_ms = int(latest_ready_remaining_ms)
                yellow_threshold_ms = int(max(1000.0, effective_delay_sec * 1000.0 * 0.5))
                red_threshold_ms = int(max(500.0, effective_delay_sec * 1000.0 * 0.2))
                if remaining_ms < red_threshold_ms:
                    alert_level = "red"
                elif remaining_ms < yellow_threshold_ms:
                    alert_level = "yellow"
                else:
                    alert_level = "green"

            status = {
                "mode": self._mode,
                "forced_mode": self._forced_mode,
                "adaptive_enabled": self._is_adaptive_enabled_locked(),
                "base_delay_sec": self.base_delay_sec(),
                "asr_delay_sec": self.asr_delay_sec(),
                "delay_adjust_sec": self._delay_adjust_sec_locked(),
                "effective_delay_sec": effective_delay_sec,
                "arrival_rate_lps": arrival_rate,
                "service_rate_lps": service_rate,
                "service_rate_floor_lps": service_rate_floor,
                "service_rate_effective_lps": service_rate_effective,
                "tail_denoise_active": tail_denoise_active,
                "predicted_queue_wait_ms": predicted_queue_wait_ms,
                "predicted_service_ms": predicted_service_ms,
                "predicted_total_ms": predicted_total_ms,
                "slack_ms": slack_ms,
                "deadline_remaining_ms": deadline_remaining_ms,
                "risk_ratio": risk_ratio,
                "predict_percentile": predict_percentile,
                "warmup_active": warmup_active,
                "fast_blocked_by_latest_ready": bool(self._last_fast_blocked_by_latest_ready),
                "latest_ready_source_key": latest_ready_source_key,
                "latest_ready_due_effective_mono_ms": latest_ready_due_effective_mono_ms,
                "latest_ready_remaining_ms": latest_ready_remaining_ms,
                "latest_ready_anomaly": latest_ready_anomaly,
                "alert_level": alert_level,
                "late_rate_recent": late_rate,
                "translate_primary_ok_recent": primary_ok_recent,
                "translate_fallback_count_recent": fallback_count_recent,
                "translate_fallback_ratio_recent": fallback_ratio_recent,
                "translate_fallback_success_count_recent": fallback_success_count_recent,
                "translate_fallback_success_rate_recent": fallback_success_rate,
                "translate_fast_count_recent": fast_count_recent,
                "translate_fast_ratio_recent": fast_ratio_recent,
                "asr_aggregate_emit_count_recent": asr_aggregate_emit_count_recent,
                "asr_aggregate_fragment_count_recent": asr_aggregate_fragment_count_recent,
                "asr_canonical_hit_count_recent": asr_canonical_hit_count_recent,
                "s1_skipped_count_recent": s1_skipped_count_recent,
                "s1_join_suppressed_count_recent": s1_join_suppressed_count_recent,
                "s1_join_total_count_recent": s1_join_total_count_recent,
                "s1_join_ratio_recent": s1_join_ratio_recent,
                "render_second_line_used_recent": render_second_line_used_recent,
                "render_frame_count_recent": render_frame_count_recent,
                "line_utilization_ratio_recent": line_utilization_ratio_recent,
                "queue": queue_stats,
                "stage": stage,
                "stage_route": {
                    "correct_provider": self._cfg.llm.correct_provider,
                    "translate_provider": self._cfg.llm.translate_provider,
                    "translate_fallback_provider": self._cfg.llm.translate_fallback_provider,
                    "s1_mode": str(self._overrides.get("s1_mode", self._cfg.pipeline.s1_mode)),
                },
                "runtime_overrides": dict(self._overrides),
                "timestamp_mono_ms": int(now_mono_ms),
                **self._asr_state,
            }
            self._last_status = status
            return dict(status)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_status)

    def get_fast_model_catalog(self) -> Dict[str, Dict[str, str]]:
        with self._lock:
            catalog: Dict[str, Dict[str, str]] = {}
            for provider, runtime in self._cfg.llm.provider_settings.items():
                catalog[str(provider)] = {
                    "fast_correct_model": str(runtime.fast_correct_model or ""),
                    "fast_translate_model": str(runtime.fast_translate_model or ""),
                }
            return catalog

    def base_delay_sec(self) -> float:
        return float(self._cfg.align.delay_sec)

    def asr_delay_sec(self) -> float:
        return float(self._cfg.align.asr_delay_sec)

    def delay_adjust_sec(self) -> float:
        with self._lock:
            return self._delay_adjust_sec_locked()

    def delay_adjust_ms(self) -> int:
        with self._lock:
            return self._delay_adjust_ms_locked()

    def effective_delay_sec(self) -> float:
        with self._lock:
            return self._effective_delay_sec_locked()

    def effective_delay_ms(self) -> int:
        with self._lock:
            return self._effective_delay_ms_locked()

    def adjust_delay(self, delta_sec: float) -> float:
        with self._lock:
            current = float(self._overrides.get("delay_adjust_sec", 0.0))
            updated = self._clamp_delay_adjust(current + float(delta_sec))
            self._overrides["delay_adjust_sec"] = updated
            return float(updated)

    def reset_delay_adjust(self) -> float:
        with self._lock:
            self._overrides["delay_adjust_sec"] = 0.0
            return 0.0

    def set_asr_provider(self, provider: str) -> None:
        with self._lock:
            value = str(provider).strip() or "unknown"
            self._asr_state["asr_provider"] = value
            self._last_status["asr_provider"] = value

    def set_asr_connected(self, connected: bool) -> None:
        with self._lock:
            value = bool(connected)
            self._asr_state["asr_connected"] = value
            self._last_status["asr_connected"] = value
            if connected:
                self._asr_state["asr_last_error"] = None
                self._last_status["asr_last_error"] = None

    def set_asr_error(self, message: Optional[str]) -> None:
        with self._lock:
            value = None if not message else str(message)
            self._asr_state["asr_last_error"] = value
            self._last_status["asr_last_error"] = value

    def observe_asr_final(self, now_mono_ms: int) -> None:
        with self._lock:
            value = int(now_mono_ms)
            self._asr_state["asr_last_final_mono_ms"] = value
            self._last_status["asr_last_final_mono_ms"] = value
            self._asr_state["asr_connected"] = True
            self._asr_state["asr_last_error"] = None
            self._last_status["asr_connected"] = True
            self._last_status["asr_last_error"] = None

    def observe_asr_partial(self, text: str, now_mono_ms: int) -> None:
        _ = text
        _ = now_mono_ms

    def get_worker_plan(self, rescue: bool = False) -> WorkerPlan:
        with self._lock:
            mode = self._mode
            batch_wait = float(self._overrides.get("batch_max_wait_sec", self._cfg.pipeline.batch_max_wait_sec))
            batch_lines = int(self._overrides.get("batch_max_lines", self._cfg.pipeline.batch_max_lines))
            context_wait = float(
                self._overrides.get("context_wait_timeout_sec", self._cfg.pipeline.context_wait_timeout_sec)
            )
            retries = int(self._cfg.llm.max_retries)
            context_window = int(self._cfg.llm.translate_context_window)
            use_fast = False
            s1_mode = str(self._overrides.get("s1_mode", self._cfg.pipeline.s1_mode)).strip().lower()
            if s1_mode not in ("off", "lite", "full", "lite_ai_join"):
                s1_mode = "lite"
            s1_lite_punctuation_only = bool(self._cfg.pipeline.s1_lite_punctuation_only)

            if mode == "yellow":
                batch_wait = min(batch_wait, 12.0)
                batch_lines = max(4, batch_lines - 1)
                context_window = min(context_window, 1)
                retries = min(retries, 0)
            elif mode == "red":
                batch_wait = min(batch_wait, 5.0)
                batch_lines = max(2, batch_lines - 2)
                context_window = 0
                retries = 0
                use_fast = True
                if bool(self._cfg.adaptive.red_force_s1_off):
                    s1_mode = "off"

            if rescue:
                context_window = 0
                retries = 0
                use_fast = True
                s1_mode = "off"

            return WorkerPlan(
                mode=mode,
                batch_max_wait_sec=max(0.5, batch_wait),
                batch_max_lines=max(1, batch_lines),
                context_wait_timeout_sec=max(0.0, context_wait),
                context_window=max(0, context_window),
                use_fast_model=use_fast,
                contextless_only=rescue,
                max_retries=max(0, retries),
                s1_mode=s1_mode,
                s1_lite_punctuation_only=s1_lite_punctuation_only,
            )

    def rescue_parallelism(self) -> int:
        with self._lock:
            value = int(self._overrides.get("red_parallel_contextless", self._cfg.adaptive.red_parallel_contextless))
            return max(0, value)

    def is_red_mode(self) -> bool:
        with self._lock:
            return self._mode == "red"

    @classmethod
    def _clamp_delay_adjust(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        return float(min(cls.DELAY_ADJUST_MAX_SEC, max(cls.DELAY_ADJUST_MIN_SEC, parsed)))

    def _delay_adjust_sec_locked(self) -> float:
        value = self._overrides.get("delay_adjust_sec", 0.0)
        return float(self._clamp_delay_adjust(value))

    def _delay_adjust_ms_locked(self) -> int:
        return int(round(self._delay_adjust_sec_locked() * 1000.0))

    def _effective_delay_sec_locked(self) -> float:
        base_eff = float(self._cfg.align.delay_sec) - float(self._cfg.align.asr_delay_sec)
        return float(max(0.0, base_eff + self._delay_adjust_sec_locked()))

    def _effective_delay_ms_locked(self) -> int:
        return int(round(self._effective_delay_sec_locked() * 1000.0))

    def _is_adaptive_enabled_locked(self) -> bool:
        return bool(self._overrides.get("adaptive_enabled", self._cfg.adaptive.enabled))

    def _effective_batch_lines_locked(self, mode: str) -> int:
        base = int(self._overrides.get("batch_max_lines", self._cfg.pipeline.batch_max_lines))
        if mode == "yellow":
            return max(4, base - 1)
        if mode == "red":
            return max(2, base - 2)
        return max(1, base)

    def _update_mode_locked(
        self,
        *,
        risk_ratio: Optional[float],
        unfinished: int,
        arrival_rate: float,
        new_count: int,
        tail_denoise_active: bool,
    ) -> None:
        forced = self._forced_mode
        if forced in ("green", "yellow", "red"):
            self._mode = forced
            return
        tail_guard = bool(self._overrides.get("tail_red_guard", self._cfg.adaptive.tail_red_guard))
        tail_small_backlog = int(
            self._overrides.get("tail_small_backlog_threshold", self._cfg.adaptive.tail_small_backlog_threshold)
        )
        green_max_ratio = float(self._cfg.adaptive_risk.green_max_ratio)
        yellow_max_ratio = float(self._cfg.adaptive_risk.yellow_max_ratio)
        tail_guard_ratio = float(self._cfg.adaptive_risk.tail_guard_ratio)

        if risk_ratio is None:
            self._mode = "green" if unfinished <= 0 else "yellow"
            return

        # Tail denoise: no new arrivals + tiny backlog drain should not flip to RED on estimation noise.
        if (
            tail_guard
            and tail_denoise_active
            and new_count == 0
            and unfinished <= max(1, tail_small_backlog)
            and risk_ratio <= tail_guard_ratio
        ):
            self._mode = "yellow"
            return

        if risk_ratio <= green_max_ratio:
            self._mode = "green"
            return
        if risk_ratio <= yellow_max_ratio:
            self._mode = "yellow"
            return
        self._mode = "red"

    def _tail_denoise_active_locked(self, *, arrival_rate: float, new_count: int, unfinished: int) -> bool:
        if unfinished <= 0:
            return False
        quiet_arrival = max(
            0.0,
            float(self._overrides.get("tail_quiet_arrival_lps", self._cfg.adaptive.tail_quiet_arrival_lps)),
        )
        tail_small_backlog = int(
            self._overrides.get("tail_small_backlog_threshold", self._cfg.adaptive.tail_small_backlog_threshold)
        )
        return new_count == 0 and arrival_rate <= quiet_arrival and unfinished <= max(1, tail_small_backlog)

    def _trim(self, now_sec: float) -> None:
        cutoff = now_sec - self._window_sec
        while self._arrivals and self._arrivals[0][0] < cutoff:
            self._arrivals.popleft()
        while self._completions and self._completions[0][0] < cutoff:
            self._completions.popleft()
        for stage in self._stage_lat.values():
            while stage and stage[0][0] < cutoff:
                stage.popleft()
        while self._translate_routes and self._translate_routes[0][0] < cutoff:
            self._translate_routes.popleft()
        while self._asr_source_events and self._asr_source_events[0][0] < cutoff:
            self._asr_source_events.popleft()
        while self._s1_skipped_events and self._s1_skipped_events[0][0] < cutoff:
            self._s1_skipped_events.popleft()
        while self._s1_join_events and self._s1_join_events[0][0] < cutoff:
            self._s1_join_events.popleft()
        while self._render_frames and self._render_frames[0][0] < cutoff:
            self._render_frames.popleft()

    @staticmethod
    def _rate_from_pairs(pairs: Deque[Tuple[float, int]], now_sec: float) -> float:
        if not pairs:
            return 0.0
        total = sum(count for _, count in pairs)
        span = max(1e-6, now_sec - pairs[0][0])
        return float(total / span)

    @staticmethod
    def _rate_from_completion(items: Deque[Tuple[float, int, int]], now_sec: float) -> float:
        if not items:
            return 0.0
        total = sum(count for _, count, _ in items)
        span = max(1e-6, now_sec - items[0][0])
        return float(total / span)


def _percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return float(ordered[low])
    frac = rank - low
    return float((ordered[low] * (1.0 - frac)) + (ordered[high] * frac))
