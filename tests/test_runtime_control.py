from __future__ import annotations

import unittest
from pathlib import Path

from live_sub_daemon.config import (
    AdaptiveConfig,
    AlignConfig,
    AppConfig,
    ConsoleConfig,
    FallbackConfig,
    KnowledgeConfig,
    LLMConfig,
    OutputConfig,
    PipelineConfig,
    RenderConfig,
    SourceConfig,
    StateConfig,
)
from live_sub_daemon.runtime_control import RuntimeController


def _build_cfg() -> AppConfig:
    return AppConfig(
        source=SourceConfig(input_srt=Path("jp.srt"), input_txt=Path("jp.txt")),
        output=OutputConfig(zh_txt=Path("zh.txt"), zh_srt=Path("zh.srt")),
        llm=LLMConfig(
            api_key="dummy",
            correct_model="model-a",
            translate_model="model-b",
            fast_correct_model="model-fast-a",
            fast_translate_model="model-fast-b",
            translate_context_window=3,
            max_retries=1,
        ),
        align=AlignConfig(),
        render=RenderConfig(),
        pipeline=PipelineConfig(batch_max_wait_sec=20, batch_max_lines=6, context_wait_timeout_sec=10),
        adaptive=AdaptiveConfig(
            enabled=True,
            mode="auto",
            green_slack_ms=30000,
            red_slack_ms=10000,
            backlog_red_threshold=8,
            red_parallel_contextless=2,
        ),
        console=ConsoleConfig(),
        state=StateConfig(db_path=Path("state.sqlite3")),
        fallback=FallbackConfig(),
        knowledge=KnowledgeConfig(glossary_path=Path("glossary_ja_zh.tsv"), names_path=Path("names_whitelist_ja.txt")),
    )


class RuntimeControlTests(unittest.TestCase):
    def _prime_completion_history(self, ctl: RuntimeController, *, now_ms: int, pipeline_ms: int) -> None:
        # ~0.55 lps service rate over ~55s window, and enough completions to exit warmup.
        start_ms = now_ms - 55000
        for idx in range(10):
            ctl.observe_completion(
                total_count=3,
                late_count=0,
                stage1_ms=max(1000, pipeline_ms // 3),
                stage2_ms=max(1000, pipeline_ms // 2),
                pipeline_ms=pipeline_ms,
                now_mono_ms=start_ms + (idx * 5000),
            )

    def test_red_mode_triggered_by_backlog(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 100000
        ctl.observe_arrival(20, now_ms)
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 10,
                "inflight_count": 0,
                "unfinished_count": 10,
                "oldest_new_seen_mono_ms": now_ms - 1000,
                "oldest_new_wait_ms": 1000,
                "min_due_unfinished_mono_ms": now_ms + 50000,
            },
            now_mono_ms=now_ms,
        )
        self.assertEqual(status["mode"], "red")

    def test_worker_plan_red_contextless(self) -> None:
        ctl = RuntimeController(_build_cfg())
        ctl.set_mode("red")
        main_plan = ctl.get_worker_plan(rescue=False)
        rescue_plan = ctl.get_worker_plan(rescue=True)
        self.assertEqual(main_plan.context_window, 0)
        self.assertTrue(main_plan.use_fast_model)
        self.assertEqual(main_plan.s1_mode, "off")
        self.assertEqual(rescue_plan.context_window, 0)
        self.assertTrue(rescue_plan.contextless_only)
        self.assertEqual(rescue_plan.s1_mode, "off")

    def test_tail_denoise_prevents_false_red(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 500000
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 4,
                "unfinished_count": 4,
                "oldest_new_seen_mono_ms": None,
                "oldest_new_wait_ms": None,
                "min_due_unfinished_mono_ms": now_ms + 90000,
            },
            now_mono_ms=now_ms,
        )
        self.assertEqual(status["mode"], "yellow")
        self.assertEqual(status["tail_denoise_active"], True)
        self.assertGreaterEqual(status["service_rate_effective_lps"], 0.05)

    def test_risk_ratio_green_when_remaining_70_predicted_45(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 1_000_000
        self._prime_completion_history(ctl, now_ms=now_ms, pipeline_ms=17000)
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 2,
                "inflight_count": 1,
                "unfinished_count": 20,
                "oldest_new_seen_mono_ms": now_ms - 2000,
                "oldest_new_wait_ms": 2000,
                "min_due_unfinished_mono_ms": now_ms + 70000,
            },
            now_mono_ms=now_ms,
        )
        self.assertEqual(status["mode"], "green")
        self.assertIsNotNone(status["predicted_total_ms"])
        self.assertLess(float(status["predicted_total_ms"]), 70000.0)

    def test_risk_ratio_yellow_when_remaining_60_predicted_52(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 1_200_000
        self._prime_completion_history(ctl, now_ms=now_ms, pipeline_ms=24000)
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 2,
                "inflight_count": 1,
                "unfinished_count": 20,
                "oldest_new_seen_mono_ms": now_ms - 2000,
                "oldest_new_wait_ms": 2000,
                "min_due_unfinished_mono_ms": now_ms + 60000,
            },
            now_mono_ms=now_ms,
        )
        self.assertEqual(status["mode"], "yellow")
        self.assertIsNotNone(status["predicted_total_ms"])
        self.assertGreater(float(status["predicted_total_ms"]), 47000.0)
        self.assertLess(float(status["predicted_total_ms"]), 60000.0)

    def test_risk_ratio_red_when_remaining_60_predicted_65(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 1_400_000
        self._prime_completion_history(ctl, now_ms=now_ms, pipeline_ms=37000)
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 2,
                "inflight_count": 1,
                "unfinished_count": 20,
                "oldest_new_seen_mono_ms": now_ms - 2000,
                "oldest_new_wait_ms": 2000,
                "min_due_unfinished_mono_ms": now_ms + 60000,
            },
            now_mono_ms=now_ms,
        )
        self.assertEqual(status["mode"], "red")
        self.assertIsNotNone(status["predicted_total_ms"])
        self.assertGreater(float(status["predicted_total_ms"]), 60000.0)

    def test_fast_blocked_when_latest_ready_buffer_high(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 1_600_000
        buffer_ms = int(ctl.effective_delay_ms() * 0.6)
        ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 0,
                "unfinished_count": 0,
                "oldest_new_seen_mono_ms": None,
                "oldest_new_wait_ms": None,
                "min_due_unfinished_mono_ms": None,
                "latest_ready_remaining_ms": buffer_ms,
                "latest_ready_anomaly": False,
            },
            now_mono_ms=now_ms,
        )
        allowed, blocked = ctl.resolve_fast_model_usage(
            requested_fast=True,
            queue_stats={"overdue_unfinished_count": 0},
        )
        self.assertFalse(allowed)
        self.assertTrue(blocked)

    def test_fast_allowed_when_overdue_even_if_latest_ready_high(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 1_800_000
        buffer_ms = int(ctl.effective_delay_ms() * 0.6)
        ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 1,
                "unfinished_count": 1,
                "oldest_new_seen_mono_ms": now_ms - 2000,
                "oldest_new_wait_ms": 2000,
                "min_due_unfinished_mono_ms": now_ms + 30000,
                "latest_ready_remaining_ms": buffer_ms,
                "latest_ready_anomaly": False,
            },
            now_mono_ms=now_ms,
        )
        allowed, blocked = ctl.resolve_fast_model_usage(
            requested_fast=True,
            queue_stats={"overdue_unfinished_count": 2},
        )
        self.assertTrue(allowed)
        self.assertFalse(blocked)

    def test_delay_adjust_clamped_and_effective_delay_changes(self) -> None:
        ctl = RuntimeController(_build_cfg())
        self.assertEqual(ctl.delay_adjust_sec(), 0.0)
        base_eff = ctl.base_delay_sec() - ctl.asr_delay_sec()
        self.assertAlmostEqual(ctl.effective_delay_sec(), base_eff)

        ctl.adjust_delay(15)
        self.assertEqual(ctl.delay_adjust_sec(), 15.0)
        self.assertAlmostEqual(ctl.effective_delay_sec(), base_eff + 15.0)

        ctl.adjust_delay(1000)
        self.assertEqual(ctl.delay_adjust_sec(), 60.0)

        ctl.adjust_delay(-1000)
        self.assertEqual(ctl.delay_adjust_sec(), -60.0)

    def test_alert_level_prefers_latest_ready_and_handles_anomaly(self) -> None:
        ctl = RuntimeController(_build_cfg())
        now_ms = 100000
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 0,
                "unfinished_count": 0,
                "oldest_new_seen_mono_ms": None,
                "oldest_new_wait_ms": None,
                "min_due_unfinished_mono_ms": None,
                "latest_ready_remaining_ms": 120000,
                "latest_ready_anomaly": False,
            },
            now_mono_ms=now_ms,
        )
        self.assertEqual(status["alert_level"], "green")

        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 0,
                "unfinished_count": 0,
                "oldest_new_seen_mono_ms": None,
                "oldest_new_wait_ms": None,
                "min_due_unfinished_mono_ms": None,
                "latest_ready_remaining_ms": 1000,
                "latest_ready_anomaly": False,
            },
            now_mono_ms=now_ms + 1000,
        )
        self.assertEqual(status["alert_level"], "red")

        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 0,
                "unfinished_count": 0,
                "oldest_new_seen_mono_ms": None,
                "oldest_new_wait_ms": None,
                "min_due_unfinished_mono_ms": None,
                "latest_ready_remaining_ms": None,
                "latest_ready_anomaly": True,
            },
            now_mono_ms=now_ms + 2000,
        )
        self.assertEqual(status["alert_level"], "yellow")
        self.assertEqual(status["latest_ready_anomaly"], True)

    def test_fast_and_fallback_ratios_split(self) -> None:
        ctl = RuntimeController(_build_cfg())
        base_ms = 200000
        ctl.observe_translate_route(
            primary_ok=True,
            fallback_used=False,
            fallback_ok=False,
            used_fast_model=True,
            now_mono_ms=base_ms,
        )
        ctl.observe_translate_route(
            primary_ok=False,
            fallback_used=True,
            fallback_ok=True,
            used_fast_model=False,
            now_mono_ms=base_ms + 1000,
        )
        status = ctl.update_and_get_status(
            queue_stats={
                "new_count": 0,
                "inflight_count": 0,
                "unfinished_count": 0,
                "oldest_new_seen_mono_ms": None,
                "oldest_new_wait_ms": None,
                "min_due_unfinished_mono_ms": None,
                "latest_ready_remaining_ms": 90000,
                "latest_ready_anomaly": False,
            },
            now_mono_ms=base_ms + 2000,
        )
        self.assertEqual(status["translate_fast_count_recent"], 1)
        self.assertAlmostEqual(float(status["translate_fast_ratio_recent"]), 0.5, places=3)
        self.assertEqual(status["translate_fallback_count_recent"], 1)
        self.assertAlmostEqual(float(status["translate_fallback_ratio_recent"]), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
