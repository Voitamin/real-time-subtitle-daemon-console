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
    ProviderRuntimeConfig,
    RenderConfig,
    SourceConfig,
    StateConfig,
)
from live_sub_daemon.llm_client import StageResult
from live_sub_daemon.stage_router import StageRouter


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run_batch(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise RuntimeError("no fake responses left")
        return self.responses.pop(0)


def _build_config() -> AppConfig:
    provider_settings = {
        "qwen": ProviderRuntimeConfig(
            provider="qwen",
            base_url="https://qwen.local/v1",
            api_key="k-qwen",
            api_key_env="DASHSCOPE_API_KEY",
            correct_model="qwen3-max",
            translate_model="qwen3-max",
            fast_correct_model="qwen3-fast",
            fast_translate_model="qwen3-fast",
        ),
        "deepseek": ProviderRuntimeConfig(
            provider="deepseek",
            base_url="https://deepseek.local/v1",
            api_key="k-deepseek",
            api_key_env="DEEPSEEK_API_KEY",
            correct_model="deepseek-chat",
            translate_model="deepseek-reasoner",
            fast_correct_model="deepseek-chat",
            fast_translate_model="deepseek-chat",
        ),
    }
    return AppConfig(
        source=SourceConfig(input_srt=Path("jp.srt"), input_txt=Path("jp.txt")),
        output=OutputConfig(zh_txt=Path("zh.txt"), zh_srt=Path("zh.srt")),
        llm=LLMConfig(
            api_key="unused",
            provider="qwen",
            correct_provider="qwen",
            translate_provider="deepseek",
            translate_fallback_provider="qwen",
            translate_fallback_on_error=True,
            translate_fallback_timeout_sec=20.0,
            provider_settings=provider_settings,
            correct_timeout_sec=10.0,
            translate_timeout_sec=12.0,
            max_retries=1,
        ),
        align=AlignConfig(),
        render=RenderConfig(),
        pipeline=PipelineConfig(),
        adaptive=AdaptiveConfig(),
        console=ConsoleConfig(),
        state=StateConfig(db_path=Path("state.sqlite3")),
        fallback=FallbackConfig(),
        knowledge=KnowledgeConfig(glossary_path=Path("g.tsv"), names_path=Path("n.txt")),
    )


class StageRouterTests(unittest.TestCase):
    def test_correct_uses_stage_provider(self) -> None:
        cfg = _build_config()
        router = StageRouter(cfg)
        qwen_client = _FakeClient(
            [
                StageResult(outputs={"k1": "fix"}, latency_ms=100, ok=True, timed_out=False, error=None, cached_tokens=0),
            ]
        )
        router._clients = {"qwen": qwen_client}

        routed = router.run_correct_batch(
            stage_name="jp_correct",
            system_prompt="sp",
            items=[{"source_key": "k1", "text": "a"}],
            max_retries=1,
            retry_backoff_sec=0.1,
            temperature=0.0,
            use_fast_model=False,
        )

        self.assertEqual(routed.provider, "qwen")
        self.assertEqual(routed.model, "qwen3-max")
        self.assertTrue(routed.result.ok)
        self.assertEqual(qwen_client.calls[0]["model"], "qwen3-max")

    def test_translate_fallback_on_primary_error(self) -> None:
        cfg = _build_config()
        router = StageRouter(cfg)
        deepseek_client = _FakeClient(
            [
                StageResult(
                    outputs={},
                    latency_ms=300,
                    ok=False,
                    timed_out=True,
                    error="timeout",
                    cached_tokens=None,
                ),
            ]
        )
        qwen_client = _FakeClient(
            [
                StageResult(
                    outputs={"k1": "zh"},
                    latency_ms=150,
                    ok=True,
                    timed_out=False,
                    error=None,
                    cached_tokens=10,
                ),
            ]
        )
        router._clients = {"deepseek": deepseek_client, "qwen": qwen_client}

        routed = router.run_translate_batch(
            stage_name="jp_to_zh",
            system_prompt="sp",
            items=[{"source_key": "k1", "text": "a"}],
            max_retries=1,
            retry_backoff_sec=0.1,
            temperature=0.0,
            use_fast_model=False,
            translation_options=None,
        )

        self.assertTrue(routed.fallback_used)
        self.assertEqual(routed.primary.provider, "deepseek")
        self.assertEqual(routed.final.provider, "qwen")
        self.assertTrue(routed.final.result.ok)
        self.assertEqual(deepseek_client.calls[0]["model"], "deepseek-reasoner")
        self.assertEqual(qwen_client.calls[0]["timeout_sec"], 20.0)
        self.assertEqual(qwen_client.calls[0]["max_retries"], 0)


if __name__ == "__main__":
    unittest.main()
