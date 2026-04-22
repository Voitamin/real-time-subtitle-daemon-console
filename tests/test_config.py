from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from live_sub_daemon.config import load_config, parse_cli_args


class ConfigTests(unittest.TestCase):
    def test_source_mode_defaults_to_file(self) -> None:
        args = parse_cli_args(["--api-key", "dummy"])
        cfg = load_config(args)
        self.assertEqual(cfg.source.mode, "file")

    def test_speechmatics_mode_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
api_key = "dummy"

[source]
mode = "speechmatics"
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            with self.assertRaises(ValueError):
                load_config(args)

    def test_speechmatics_mode_reads_env_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
api_key = "dummy"

[source]
mode = "speechmatics"

[source.speechmatics]
api_key_env = "SPEECHMATICS_API_KEY"
sample_rate = 16000
chunk_size = 2048
""".strip(),
                encoding="utf-8",
            )
            old = os.environ.get("SPEECHMATICS_API_KEY")
            try:
                os.environ["SPEECHMATICS_API_KEY"] = "sm-key"
                args = parse_cli_args(["--config", str(config_path)])
                cfg = load_config(args)
            finally:
                if old is None:
                    os.environ.pop("SPEECHMATICS_API_KEY", None)
                else:
                    os.environ["SPEECHMATICS_API_KEY"] = old

            self.assertEqual(cfg.source.mode, "speechmatics")
            self.assertEqual(cfg.source.speechmatics.api_key, "sm-key")
            self.assertEqual(cfg.source.speechmatics.chunk_size, 2048)
            self.assertEqual(cfg.source.speechmatics.max_delay_mode, "flexible")
            self.assertEqual(cfg.source.speechmatics.final_aggregate_mode, "v2")

    def test_speechmatics_max_delay_mode_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
api_key = "dummy"

[source]
mode = "speechmatics"

[source.speechmatics]
api_key = "sm-key"
max_delay_mode = "invalid"
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            with self.assertRaises(ValueError):
                load_config(args)

    def test_pipeline_s1_mode_and_knowledge_phonetic_flags(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
api_key = "dummy"

[pipeline]
s1_mode = "lite_ai_join"
s1_lite_punctuation_only = false
s1_ai_join_marker = "<<JOIN_NEXT>>"
s1_ai_join_allow_light_compress = true
s1_ai_join_max_chain = 0
s1_ai_join_keep_min_chars = 0

[knowledge]
names_phonetic_canonicalize_enabled = false
names_phonetic_max_rules = 123
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            cfg = load_config(args)
            self.assertEqual(cfg.pipeline.s1_mode, "lite_ai_join")
            self.assertEqual(cfg.pipeline.s1_lite_punctuation_only, False)
            self.assertEqual(cfg.pipeline.s1_ai_join_marker, "<<JOIN_NEXT>>")
            self.assertEqual(cfg.pipeline.s1_ai_join_allow_light_compress, True)
            self.assertEqual(cfg.knowledge.names_phonetic_canonicalize_enabled, False)
            self.assertEqual(cfg.knowledge.names_phonetic_max_rules, 123)

    def test_speechmatics_vocab_source_alias_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
api_key = "dummy"

[source]
mode = "speechmatics"

[source.speechmatics]
api_key = "sm-key"
additional_vocab_from = "names_glossary"
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            cfg = load_config(args)
            self.assertEqual(cfg.source.speechmatics.additional_vocab_from, "names_whitelist_glossary")

    def test_cli_overrides_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
correct_timeout_sec = 20
translate_timeout_sec = 22
max_retries = 3
retry_backoff_sec = 3
translate_context_window = 4

[align]
delay_sec = 300
asr_delay_sec = 4
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(
                [
                    "--config",
                    str(config_path),
                    "--api-key",
                    "dummy",
                    "--delay-sec",
                    "180",
                    "--asr-delay-sec",
                    "2.5",
                    "--correct-timeout-sec",
                    "12",
                ]
            )
            cfg = load_config(args)

            self.assertEqual(cfg.align.delay_sec, 180)
            self.assertEqual(cfg.align.asr_delay_sec, 2.5)
            self.assertEqual(cfg.llm.correct_timeout_sec, 12)
            self.assertEqual(cfg.llm.translate_timeout_sec, 22)
            self.assertEqual(cfg.llm.max_retries, 3)
            self.assertEqual(cfg.llm.translate_context_window, 4)
            self.assertEqual(cfg.pipeline.batch_max_wait_sec, 20.0)

    def test_knowledge_limits_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[knowledge]
correct_glossary_limit = 111
translate_glossary_limit = 222
names_limit = 333
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(
                [
                    "--config",
                    str(config_path),
                    "--api-key",
                    "dummy",
                ]
            )
            cfg = load_config(args)

            self.assertEqual(cfg.knowledge.correct_glossary_limit, 111)
            self.assertEqual(cfg.knowledge.translate_glossary_limit, 222)
            self.assertEqual(cfg.knowledge.names_limit, 333)

    def test_deepseek_provider_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "deepseek"
""".strip(),
                encoding="utf-8",
            )
            old = os.environ.get("DEEPSEEK_API_KEY")
            try:
                os.environ["DEEPSEEK_API_KEY"] = "dummy"
                args = parse_cli_args(["--config", str(config_path)])
                cfg = load_config(args)
            finally:
                if old is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = old

            self.assertEqual(cfg.llm.provider, "deepseek")
            self.assertEqual(cfg.llm.base_url, "https://api.deepseek.com/v1")
            self.assertEqual(cfg.llm.correct_model, "deepseek-chat")
            self.assertEqual(cfg.llm.translate_model, "deepseek-chat")

    def test_provider_section_values_used(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"

[providers.xai]
base_url = "https://example.local/v1"
api_key = "provider_key"
correct_model = "model-a"
translate_model = "model-b"
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            cfg = load_config(args)

            self.assertEqual(cfg.llm.provider, "xai")
            self.assertEqual(cfg.llm.base_url, "https://example.local/v1")
            self.assertEqual(cfg.llm.api_key, "provider_key")
            self.assertEqual(cfg.llm.correct_model, "model-a")
            self.assertEqual(cfg.llm.translate_model, "model-b")

    def test_qwen_provider_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "qwen"
""".strip(),
                encoding="utf-8",
            )
            old = os.environ.get("DASHSCOPE_API_KEY")
            try:
                os.environ["DASHSCOPE_API_KEY"] = "dummy"
                args = parse_cli_args(["--config", str(config_path)])
                cfg = load_config(args)
            finally:
                if old is None:
                    os.environ.pop("DASHSCOPE_API_KEY", None)
                else:
                    os.environ["DASHSCOPE_API_KEY"] = old

            self.assertEqual(cfg.llm.provider, "qwen")
            self.assertEqual(cfg.llm.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(cfg.llm.correct_model, "qwen-plus")
            self.assertEqual(cfg.llm.translate_model, "qwen-plus")

    def test_zai_provider_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "zai"
""".strip(),
                encoding="utf-8",
            )
            old = os.environ.get("ZAI_API_KEY")
            try:
                os.environ["ZAI_API_KEY"] = "dummy"
                args = parse_cli_args(["--config", str(config_path)])
                cfg = load_config(args)
            finally:
                if old is None:
                    os.environ.pop("ZAI_API_KEY", None)
                else:
                    os.environ["ZAI_API_KEY"] = old

            self.assertEqual(cfg.llm.provider, "zai")
            self.assertEqual(cfg.llm.base_url, "https://api.z.ai/api/paas/v4")
            self.assertEqual(cfg.llm.correct_model, "glm-5")
            self.assertEqual(cfg.llm.translate_model, "glm-5")

    def test_zai_alias_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "z.ai"
""".strip(),
                encoding="utf-8",
            )
            old = os.environ.get("ZAI_API_KEY")
            try:
                os.environ["ZAI_API_KEY"] = "dummy"
                args = parse_cli_args(["--config", str(config_path)])
                cfg = load_config(args)
            finally:
                if old is None:
                    os.environ.pop("ZAI_API_KEY", None)
                else:
                    os.environ["ZAI_API_KEY"] = old

            self.assertEqual(cfg.llm.provider, "zai")

    def test_stage_provider_routing_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
correct_provider = "qwen"
translate_provider = "deepseek"
translate_fallback_provider = "qwen"
translate_fallback_on_error = true
translate_fallback_timeout_sec = 18

[providers.xai]
api_key = "k-xai"

[providers.qwen]
api_key = "k-qwen"
correct_model = "qwen3-max"
translate_model = "qwen3-max"

[providers.deepseek]
api_key = "k-deepseek"
correct_model = "deepseek-chat"
translate_model = "deepseek-reasoner"
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            cfg = load_config(args)

            self.assertEqual(cfg.llm.correct_provider, "qwen")
            self.assertEqual(cfg.llm.translate_provider, "deepseek")
            self.assertEqual(cfg.llm.translate_fallback_provider, "qwen")
            self.assertEqual(cfg.llm.translate_fallback_timeout_sec, 18.0)
            self.assertEqual(cfg.llm.provider_settings["qwen"].correct_model, "qwen3-max")
            self.assertEqual(cfg.llm.provider_settings["deepseek"].translate_model, "deepseek-reasoner")

    def test_missing_stage_provider_api_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
correct_provider = "qwen"
translate_provider = "deepseek"
translate_fallback_provider = "qwen"

[providers.xai]
api_key = "k-xai"
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            with self.assertRaises(ValueError):
                load_config(args)

    def test_pipeline_adaptive_console_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "config.toml"
            config_path.write_text(
                """
[llm]
provider = "xai"
api_key = "dummy"

[pipeline]
batch_max_wait_sec = 9
batch_max_lines = 4
context_wait_timeout_sec = 3

[adaptive]
enabled = true
mode = "yellow"
green_slack_ms = 45000
red_slack_ms = 15000
backlog_red_threshold = 99
red_parallel_contextless = 1
service_rate_floor_lps = 0.08
tail_quiet_arrival_lps = 0.01
tail_small_backlog_threshold = 5
tail_red_guard = false

[console]
enabled = false
host = "127.0.0.1"
port = 9988
""".strip(),
                encoding="utf-8",
            )
            args = parse_cli_args(["--config", str(config_path)])
            cfg = load_config(args)

            self.assertEqual(cfg.pipeline.batch_max_wait_sec, 9.0)
            self.assertEqual(cfg.pipeline.batch_max_lines, 4)
            self.assertEqual(cfg.pipeline.context_wait_timeout_sec, 3.0)
            self.assertEqual(cfg.adaptive.mode, "yellow")
            self.assertEqual(cfg.adaptive.green_slack_ms, 45000)
            self.assertEqual(cfg.adaptive.red_parallel_contextless, 1)
            self.assertEqual(cfg.adaptive.service_rate_floor_lps, 0.08)
            self.assertEqual(cfg.adaptive.tail_quiet_arrival_lps, 0.01)
            self.assertEqual(cfg.adaptive.tail_small_backlog_threshold, 5)
            self.assertEqual(cfg.adaptive.tail_red_guard, False)
            self.assertEqual(cfg.console.enabled, False)
            self.assertEqual(cfg.console.port, 9988)


if __name__ == "__main__":
    unittest.main()
