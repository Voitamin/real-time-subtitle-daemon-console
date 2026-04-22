from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProviderRuntimeConfig:
    provider: str
    base_url: str
    api_key: str
    api_key_env: str
    correct_model: str
    translate_model: str
    fast_correct_model: str = ""
    fast_translate_model: str = ""


@dataclass
class SpeechmaticsConfig:
    api_key: str = ""
    api_key_env: str = "SPEECHMATICS_API_KEY"
    language: str = "ja"
    operating_point: str = "enhanced"
    enable_partials: bool = True
    max_delay: float = 1.0
    max_delay_mode: str = "flexible"
    sample_rate: int = 16000
    chunk_size: int = 4096
    capture_backend: str = "ffmpeg_pipe"
    capture_cmd: str = (
        "ffmpeg -hide_banner -loglevel warning "
        "-f wasapi -i default -ac 1 -ar {sample_rate} -f s16le -"
    )
    additional_vocab_from: str = "names_whitelist_glossary"
    additional_vocab_limit: int = 1000
    final_aggregate_enabled: bool = True
    final_aggregate_mode: str = "v2"
    final_aggregate_gap_sec: float = 1.2
    final_aggregate_max_sec: float = 8.0
    final_aggregate_force_emit_sec: float = 12.0
    final_aggregate_min_chars: int = 8
    final_aggregate_max_chars: int = 120
    final_aggregate_overlap_dedup: bool = True
    final_aggregate_flush_on_punct: bool = True


@dataclass
class SourceConfig:
    input_srt: Path
    input_txt: Path
    mode: str = "file"  # file | speechmatics
    poll_interval_sec: float = 0.5
    speechmatics: SpeechmaticsConfig = field(default_factory=SpeechmaticsConfig)


@dataclass
class OutputConfig:
    zh_txt: Path
    zh_srt: Path


@dataclass
class LLMConfig:
    api_key: str
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    correct_model: str = "deepseek-chat"
    translate_model: str = "deepseek-chat"
    fast_correct_model: str = ""
    fast_translate_model: str = ""
    correct_provider: str = "deepseek"
    translate_provider: str = "deepseek"
    translate_fallback_provider: str = "deepseek"
    translate_fallback_on_error: bool = True
    translate_fallback_timeout_sec: float = 20.0
    correct_timeout_sec: float = 12.0
    translate_timeout_sec: float = 12.0
    max_retries: int = 1
    retry_backoff_sec: float = 1.5
    batch_size: int = 5
    translate_context_window: int = 3
    temperature: float = 0.0
    provider_settings: Dict[str, ProviderRuntimeConfig] = field(default_factory=dict)


@dataclass
class AlignConfig:
    delay_sec: float = 240.0
    asr_delay_sec: float = 0.0


@dataclass
class RenderConfig:
    tick_hz: float = 1.0
    char_threshold: int = 18
    max_total_chars: int = 42
    max_lines: int = 2
    two_line_roll_enabled: bool = True
    min_hold_sec: float = 1.6
    target_cps: float = 13.0
    max_hold_sec: float = 4.0
    backlog_relax_threshold: int = 10
    backlog_relaxed_min_hold_sec: float = 1.0


@dataclass
class PipelineConfig:
    batch_max_wait_sec: float = 20.0
    batch_max_lines: int = 6
    context_wait_timeout_sec: float = 10.0
    s1_mode: str = "lite"  # off | lite | full | lite_ai_join
    s1_lite_punctuation_only: bool = True
    s1_ai_join_marker: str = "<<JOIN_NEXT>>"
    s1_ai_join_allow_light_compress: bool = True
    s1_ai_join_max_chain: int = 0
    s1_ai_join_keep_min_chars: int = 0


@dataclass
class AdaptiveConfig:
    enabled: bool = True
    mode: str = "auto"  # auto|green|yellow|red
    green_slack_ms: int = 30000
    red_slack_ms: int = 10000
    backlog_red_threshold: int = 24
    red_parallel_contextless: int = 2
    service_rate_floor_lps: float = 0.05
    tail_quiet_arrival_lps: float = 0.02
    tail_small_backlog_threshold: int = 6
    tail_red_guard: bool = True
    red_force_s1_off: bool = True


@dataclass
class AdaptivePredictConfig:
    pipeline_service_percentile: int = 90  # 50 | 90 | 95
    queue_exclude_current_batch: bool = True
    warmup_min_completions: int = 8
    warmup_nominal_pipeline_ms: int = 22000


@dataclass
class AdaptiveRiskConfig:
    green_max_ratio: float = 0.80
    yellow_max_ratio: float = 1.00
    tail_guard_ratio: float = 1.05


@dataclass
class AdaptiveFastConfig:
    block_when_latest_ready_ratio: float = 0.50
    emergency_when_latest_ready_ratio: float = 0.18
    allow_fast_if_overdue: bool = True


@dataclass
class ConsoleConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8787


@dataclass
class RuntimeScopeConfig:
    enabled: bool = True
    only_active_source_kind: bool = True
    freshness_sec: float = 1800.0
    monotonic_guard_sec: float = 600.0
    auto_archive_stale_on_start: bool = True


@dataclass
class MetricsConfig:
    trace_enabled: bool = True
    trace_path: Path = Path("runtime_trace.jsonl")
    trace_interval_sec: float = 1.0


@dataclass
class StateConfig:
    db_path: Path


@dataclass
class FallbackConfig:
    mode: str = "jp_raw"


@dataclass
class KnowledgeConfig:
    glossary_path: Path
    names_path: Path
    reload_interval_sec: float = 2.0
    correct_glossary_limit: int = 200
    translate_glossary_limit: int = 500
    names_limit: int = 500
    names_phonetic_canonicalize_enabled: bool = True
    names_phonetic_max_rules: int = 500


@dataclass
class AppConfig:
    source: SourceConfig
    output: OutputConfig
    llm: LLMConfig
    align: AlignConfig
    render: RenderConfig
    pipeline: PipelineConfig
    adaptive: AdaptiveConfig
    console: ConsoleConfig
    state: StateConfig
    fallback: FallbackConfig
    knowledge: KnowledgeConfig
    adaptive_predict: AdaptivePredictConfig = field(default_factory=AdaptivePredictConfig)
    adaptive_risk: AdaptiveRiskConfig = field(default_factory=AdaptiveRiskConfig)
    adaptive_fast: AdaptiveFastConfig = field(default_factory=AdaptiveFastConfig)
    runtime_scope: RuntimeScopeConfig = field(default_factory=RuntimeScopeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
SUPPORTED_LLM_PROVIDERS = ("xai", "deepseek", "qwen", "zai")
PROVIDER_ALIASES = {
    "z.ai": "zai",
}
SUPPORTED_LLM_PROVIDER_INPUTS = tuple(sorted(set(SUPPORTED_LLM_PROVIDERS) | set(PROVIDER_ALIASES.keys())))
PROVIDER_DEFAULT_BASE_URL = {
    "xai": "https://api.x.ai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zai": "https://api.z.ai/api/paas/v4",
}
PROVIDER_DEFAULT_MODEL = {
    "xai": "grok-3-mini",
    "deepseek": "deepseek-chat",
    "qwen": "qwen-plus",
    "zai": "glm-5",
}
PROVIDER_DEFAULT_API_KEY_ENV = {
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "zai": "ZAI_API_KEY",
}

SUPPORTED_SPEECHMATICS_VOCAB_SOURCES = (
    "names_whitelist_glossary",
    "names_whitelist",
    "glossary",
    "none",
)
SUPPORTED_SPEECHMATICS_MAX_DELAY_MODES = ("fixed", "flexible")
SUPPORTED_SPEECHMATICS_AGGREGATE_MODES = ("v1", "v2")
SUPPORTED_S1_MODES = ("off", "lite", "full", "lite_ai_join")
SPEECHMATICS_VOCAB_SOURCE_ALIASES = {
    "names_glossary": "names_whitelist_glossary",
    "names": "names_whitelist",
}


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live subtitle translation daemon")
    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument("--input-srt", type=Path, default=None)
    parser.add_argument("--input-txt", type=Path, default=None)
    parser.add_argument("--source-mode", type=str, choices=["file", "speechmatics"], default=None)
    parser.add_argument("--speechmatics-api-key", type=str, default=None)
    parser.add_argument("--speechmatics-api-key-env", type=str, default=None)
    parser.add_argument("--speechmatics-language", type=str, default=None)
    parser.add_argument("--speechmatics-operating-point", type=str, default=None)
    parser.add_argument("--speechmatics-enable-partials", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--speechmatics-max-delay", type=float, default=None)
    parser.add_argument("--speechmatics-max-delay-mode", type=str, default=None)
    parser.add_argument("--speechmatics-sample-rate", type=int, default=None)
    parser.add_argument("--speechmatics-chunk-size", type=int, default=None)
    parser.add_argument("--speechmatics-capture-backend", type=str, default=None)
    parser.add_argument("--speechmatics-capture-cmd", type=str, default=None)
    parser.add_argument("--speechmatics-additional-vocab-from", type=str, default=None)
    parser.add_argument("--speechmatics-additional-vocab-limit", type=int, default=None)
    parser.add_argument("--speechmatics-final-aggregate-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--speechmatics-final-aggregate-mode", type=str, default=None)
    parser.add_argument("--speechmatics-final-aggregate-gap-sec", type=float, default=None)
    parser.add_argument("--speechmatics-final-aggregate-max-sec", type=float, default=None)
    parser.add_argument("--speechmatics-final-aggregate-force-emit-sec", type=float, default=None)
    parser.add_argument("--speechmatics-final-aggregate-min-chars", type=int, default=None)
    parser.add_argument("--speechmatics-final-aggregate-max-chars", type=int, default=None)
    parser.add_argument("--speechmatics-final-aggregate-overlap-dedup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--speechmatics-final-aggregate-flush-on-punct",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--zh-txt", type=Path, default=None)
    parser.add_argument("--zh-srt", type=Path, default=None)

    parser.add_argument("--delay-sec", type=float, default=None)
    parser.add_argument("--asr-delay-sec", "--whisper-delay-sec", dest="asr_delay_sec", type=float, default=None)

    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=SUPPORTED_LLM_PROVIDER_INPUTS, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model-correct", type=str, default=None)
    parser.add_argument("--model-translate", type=str, default=None)
    parser.add_argument("--correct-provider", type=str, choices=SUPPORTED_LLM_PROVIDER_INPUTS, default=None)
    parser.add_argument("--translate-provider", type=str, choices=SUPPORTED_LLM_PROVIDER_INPUTS, default=None)
    parser.add_argument("--translate-fallback-provider", type=str, choices=SUPPORTED_LLM_PROVIDER_INPUTS, default=None)
    parser.add_argument("--translate-fallback-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--translate-fallback-timeout-sec", type=float, default=None)
    parser.add_argument("--correct-timeout-sec", type=float, default=None)
    parser.add_argument("--translate-timeout-sec", type=float, default=None)
    parser.add_argument("--llm-max-retries", type=int, default=None)
    parser.add_argument("--llm-retry-backoff-sec", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--translate-context-window", type=int, default=None)
    parser.add_argument("--batch-max-wait-sec", type=float, default=None)
    parser.add_argument("--batch-max-lines", type=int, default=None)
    parser.add_argument("--context-wait-timeout-sec", type=float, default=None)
    parser.add_argument("--s1-mode", type=str, default=None)
    parser.add_argument("--s1-lite-punctuation-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--s1-ai-join-marker", type=str, default=None)
    parser.add_argument("--s1-ai-join-allow-light-compress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--s1-ai-join-max-chain", type=int, default=None)
    parser.add_argument("--s1-ai-join-keep-min-chars", type=int, default=None)
    parser.add_argument("--adaptive-mode", type=str, choices=["auto", "green", "yellow", "red"], default=None)
    parser.add_argument("--adaptive-service-rate-floor-lps", type=float, default=None)
    parser.add_argument("--adaptive-tail-quiet-arrival-lps", type=float, default=None)
    parser.add_argument("--adaptive-tail-small-backlog-threshold", type=int, default=None)
    parser.add_argument("--adaptive-tail-red-guard", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-red-force-s1-off", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-host", type=str, default=None)
    parser.add_argument("--console-port", type=int, default=None)
    parser.add_argument("--console-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metrics-trace-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metrics-trace-path", type=Path, default=None)
    parser.add_argument("--metrics-trace-interval-sec", type=float, default=None)

    parser.add_argument("--tick-hz", type=float, default=None)
    parser.add_argument("--char-threshold", type=int, default=None)
    parser.add_argument("--render-max-total-chars", type=int, default=None)
    parser.add_argument("--render-max-lines", type=int, default=None)
    parser.add_argument("--render-two-line-roll-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render-min-hold-sec", type=float, default=None)
    parser.add_argument("--render-target-cps", type=float, default=None)
    parser.add_argument("--render-max-hold-sec", type=float, default=None)
    parser.add_argument("--render-backlog-relax-threshold", type=int, default=None)
    parser.add_argument("--render-backlog-relaxed-min-hold-sec", type=float, default=None)
    parser.add_argument("--fallback-mode", type=str, choices=["jp_raw", "empty"], default=None)

    parser.add_argument("--state-db", type=Path, default=None)
    parser.add_argument("--glossary", type=Path, default=None)
    parser.add_argument("--names-whitelist", type=Path, default=None)
    parser.add_argument("--knowledge-reload-sec", type=float, default=None)
    parser.add_argument("--knowledge-correct-glossary-limit", type=int, default=None)
    parser.add_argument("--knowledge-translate-glossary-limit", type=int, default=None)
    parser.add_argument("--knowledge-names-limit", type=int, default=None)
    parser.add_argument("--knowledge-names-phonetic-canonicalize-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--knowledge-names-phonetic-max-rules", type=int, default=None)
    parser.add_argument("--source-poll-sec", type=float, default=None)
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> AppConfig:
    defaults = _default_config()
    config_data: Dict[str, Any] = {}
    if args.config:
        config_data = _parse_toml_file(args.config)

    def get_value(path: str, cli_value: Any, default_value: Any) -> Any:
        if cli_value is not None:
            return cli_value
        from_file = _nested_get(config_data, path)
        if from_file is not None:
            return from_file
        return default_value

    source_mode = str(get_value("source.mode", args.source_mode, defaults.source.mode)).strip().lower()
    if source_mode not in ("file", "speechmatics"):
        raise ValueError(f"Unsupported source mode: {source_mode}")
    speechmatics_api_key_env = str(
        get_value(
            "source.speechmatics.api_key_env",
            args.speechmatics_api_key_env,
            defaults.source.speechmatics.api_key_env,
        )
    ).strip()
    speechmatics_api_key = str(
        get_value(
            "source.speechmatics.api_key",
            args.speechmatics_api_key,
            defaults.source.speechmatics.api_key,
        )
    ).strip()
    if not speechmatics_api_key and speechmatics_api_key_env:
        speechmatics_api_key = os.getenv(speechmatics_api_key_env, "").strip()
    speechmatics = SpeechmaticsConfig(
        api_key=speechmatics_api_key,
        api_key_env=speechmatics_api_key_env or defaults.source.speechmatics.api_key_env,
        language=str(
            get_value(
                "source.speechmatics.language",
                args.speechmatics_language,
                defaults.source.speechmatics.language,
            )
        ),
        operating_point=str(
            get_value(
                "source.speechmatics.operating_point",
                args.speechmatics_operating_point,
                defaults.source.speechmatics.operating_point,
            )
        ),
        enable_partials=bool(
            get_value(
                "source.speechmatics.enable_partials",
                args.speechmatics_enable_partials,
                defaults.source.speechmatics.enable_partials,
            )
        ),
        max_delay=float(
            get_value(
                "source.speechmatics.max_delay",
                args.speechmatics_max_delay,
                defaults.source.speechmatics.max_delay,
            )
        ),
        max_delay_mode=normalize_speechmatics_max_delay_mode(
            str(
                get_value(
                    "source.speechmatics.max_delay_mode",
                    args.speechmatics_max_delay_mode,
                    defaults.source.speechmatics.max_delay_mode,
                )
            )
        ),
        sample_rate=int(
            get_value(
                "source.speechmatics.sample_rate",
                args.speechmatics_sample_rate,
                defaults.source.speechmatics.sample_rate,
            )
        ),
        chunk_size=int(
            get_value(
                "source.speechmatics.chunk_size",
                args.speechmatics_chunk_size,
                defaults.source.speechmatics.chunk_size,
            )
        ),
        capture_backend=str(
            get_value(
                "source.speechmatics.capture_backend",
                args.speechmatics_capture_backend,
                defaults.source.speechmatics.capture_backend,
            )
        ),
        capture_cmd=str(
            get_value(
                "source.speechmatics.capture_cmd",
                args.speechmatics_capture_cmd,
                defaults.source.speechmatics.capture_cmd,
            )
        ),
        additional_vocab_from=normalize_speechmatics_vocab_source(
            str(
                get_value(
                    "source.speechmatics.additional_vocab_from",
                    args.speechmatics_additional_vocab_from,
                    defaults.source.speechmatics.additional_vocab_from,
                )
            )
        ),
        additional_vocab_limit=int(
            get_value(
                "source.speechmatics.additional_vocab_limit",
                args.speechmatics_additional_vocab_limit,
                defaults.source.speechmatics.additional_vocab_limit,
            )
        ),
        final_aggregate_enabled=bool(
            get_value(
                "source.speechmatics.final_aggregate_enabled",
                args.speechmatics_final_aggregate_enabled,
                defaults.source.speechmatics.final_aggregate_enabled,
            )
        ),
        final_aggregate_mode=normalize_speechmatics_aggregate_mode(
            str(
                get_value(
                    "source.speechmatics.final_aggregate_mode",
                    args.speechmatics_final_aggregate_mode,
                    defaults.source.speechmatics.final_aggregate_mode,
                )
            )
        ),
        final_aggregate_gap_sec=float(
            get_value(
                "source.speechmatics.final_aggregate_gap_sec",
                args.speechmatics_final_aggregate_gap_sec,
                defaults.source.speechmatics.final_aggregate_gap_sec,
            )
        ),
        final_aggregate_max_sec=float(
            get_value(
                "source.speechmatics.final_aggregate_max_sec",
                args.speechmatics_final_aggregate_max_sec,
                defaults.source.speechmatics.final_aggregate_max_sec,
            )
        ),
        final_aggregate_force_emit_sec=float(
            get_value(
                "source.speechmatics.final_aggregate_force_emit_sec",
                args.speechmatics_final_aggregate_force_emit_sec,
                defaults.source.speechmatics.final_aggregate_force_emit_sec,
            )
        ),
        final_aggregate_min_chars=int(
            get_value(
                "source.speechmatics.final_aggregate_min_chars",
                args.speechmatics_final_aggregate_min_chars,
                defaults.source.speechmatics.final_aggregate_min_chars,
            )
        ),
        final_aggregate_max_chars=int(
            get_value(
                "source.speechmatics.final_aggregate_max_chars",
                args.speechmatics_final_aggregate_max_chars,
                defaults.source.speechmatics.final_aggregate_max_chars,
            )
        ),
        final_aggregate_overlap_dedup=bool(
            get_value(
                "source.speechmatics.final_aggregate_overlap_dedup",
                args.speechmatics_final_aggregate_overlap_dedup,
                defaults.source.speechmatics.final_aggregate_overlap_dedup,
            )
        ),
        final_aggregate_flush_on_punct=bool(
            get_value(
                "source.speechmatics.final_aggregate_flush_on_punct",
                args.speechmatics_final_aggregate_flush_on_punct,
                defaults.source.speechmatics.final_aggregate_flush_on_punct,
            )
        ),
    )
    if source_mode == "speechmatics" and not speechmatics.api_key:
        raise ValueError(
            "Missing Speechmatics API key: set source.speechmatics.api_key or env "
            f"{speechmatics.api_key_env}"
        )

    source = SourceConfig(
        input_srt=Path(get_value("source.input_srt", args.input_srt, defaults.source.input_srt)),
        input_txt=Path(get_value("source.input_txt", args.input_txt, defaults.source.input_txt)),
        mode=source_mode,
        poll_interval_sec=float(get_value("source.poll_interval_sec", args.source_poll_sec, defaults.source.poll_interval_sec)),
        speechmatics=speechmatics,
    )
    output = OutputConfig(
        zh_txt=Path(get_value("output.zh_txt", args.zh_txt, defaults.output.zh_txt)),
        zh_srt=Path(get_value("output.zh_srt", args.zh_srt, defaults.output.zh_srt)),
    )

    provider_raw = str(get_value("llm.provider", args.provider, defaults.llm.provider)).strip().lower()
    provider = normalize_provider_name(provider_raw)
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            "Unsupported llm provider: "
            f"{provider_raw}. Supported: {', '.join(SUPPORTED_LLM_PROVIDERS)}; aliases: {', '.join(PROVIDER_ALIASES.keys())}"
        )

    api_key_env = _resolve_api_key_env(args, config_data, provider)
    api_key = (
        args.api_key
        or str(_nested_get(config_data, "llm.api_key") or "")
        or str(_provider_get(config_data, provider, "api_key") or "")
    )
    if not api_key:
        api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ValueError(f"Missing API key: pass --api-key or set env {api_key_env}")

    default_base_url = _default_base_url_for_provider(provider)
    default_model = _default_model_for_provider(provider)
    resolved_base_url = str(
        _resolve_llm_value(
            args.base_url,
            _nested_get(config_data, "llm.base_url"),
            _provider_get(config_data, provider, "base_url"),
            default_base_url,
        )
    )
    resolved_correct_model = str(
        _resolve_llm_value(
            args.model_correct,
            _nested_get(config_data, "llm.correct_model"),
            _provider_get(config_data, provider, "correct_model"),
            _provider_get(config_data, provider, "default_model") or default_model,
        )
    )
    resolved_translate_model = str(
        _resolve_llm_value(
            args.model_translate,
            _nested_get(config_data, "llm.translate_model"),
            _provider_get(config_data, provider, "translate_model"),
            _provider_get(config_data, provider, "default_model") or default_model,
        )
    )
    resolved_fast_correct_model = str(
        _resolve_llm_value(
            None,
            _nested_get(config_data, "llm.fast_correct_model"),
            _provider_get(config_data, provider, "fast_correct_model"),
            "",
        )
    )
    resolved_fast_translate_model = str(
        _resolve_llm_value(
            None,
            _nested_get(config_data, "llm.fast_translate_model"),
            _provider_get(config_data, provider, "fast_translate_model"),
            "",
        )
    )

    provider_settings = _build_provider_settings(
        args=args,
        config_data=config_data,
        selected_provider=provider,
        selected_api_key=api_key,
        selected_base_url=resolved_base_url,
        selected_correct_model=resolved_correct_model,
        selected_translate_model=resolved_translate_model,
        selected_fast_correct_model=resolved_fast_correct_model,
        selected_fast_translate_model=resolved_fast_translate_model,
    )

    correct_provider_raw = str(
        get_value("llm.correct_provider", args.correct_provider, provider)
    ).strip().lower()
    translate_provider_raw = str(
        get_value("llm.translate_provider", args.translate_provider, provider)
    ).strip().lower()
    fallback_provider_default = normalize_provider_name(translate_provider_raw)
    fallback_provider_raw = str(
        get_value("llm.translate_fallback_provider", args.translate_fallback_provider, fallback_provider_default)
    ).strip().lower()

    correct_provider = normalize_provider_name(correct_provider_raw)
    translate_provider = normalize_provider_name(translate_provider_raw)
    translate_fallback_provider = normalize_provider_name(fallback_provider_raw)
    for stage_provider_raw, stage_provider in (
        (correct_provider_raw, correct_provider),
        (translate_provider_raw, translate_provider),
        (fallback_provider_raw, translate_fallback_provider),
    ):
        if stage_provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"Unsupported stage provider: {stage_provider_raw}")

    translate_fallback_on_error = bool(
        get_value(
            "llm.translate_fallback_on_error",
            args.translate_fallback_on_error,
            defaults.llm.translate_fallback_on_error,
        )
    )
    translate_fallback_timeout_sec = float(
        get_value(
            "llm.translate_fallback_timeout_sec",
            args.translate_fallback_timeout_sec,
            defaults.llm.translate_fallback_timeout_sec,
        )
    )

    required_stage_providers = {correct_provider, translate_provider}
    if translate_fallback_on_error:
        required_stage_providers.add(translate_fallback_provider)
    for required_provider in sorted(required_stage_providers):
        runtime = provider_settings.get(required_provider)
        if runtime is None:
            raise ValueError(f"Missing provider runtime config for stage provider: {required_provider}")
        if not runtime.api_key:
            raise ValueError(
                f"Missing API key for provider '{required_provider}': set providers.{required_provider}.api_key "
                f"or env {runtime.api_key_env}"
            )

    llm = LLMConfig(
        api_key=api_key,
        provider=provider,
        base_url=resolved_base_url,
        correct_model=resolved_correct_model,
        translate_model=resolved_translate_model,
        fast_correct_model=resolved_fast_correct_model,
        fast_translate_model=resolved_fast_translate_model,
        correct_provider=correct_provider,
        translate_provider=translate_provider,
        translate_fallback_provider=translate_fallback_provider,
        translate_fallback_on_error=translate_fallback_on_error,
        translate_fallback_timeout_sec=translate_fallback_timeout_sec,
        correct_timeout_sec=float(get_value("llm.correct_timeout_sec", args.correct_timeout_sec, defaults.llm.correct_timeout_sec)),
        translate_timeout_sec=float(get_value("llm.translate_timeout_sec", args.translate_timeout_sec, defaults.llm.translate_timeout_sec)),
        max_retries=int(get_value("llm.max_retries", args.llm_max_retries, defaults.llm.max_retries)),
        retry_backoff_sec=float(get_value("llm.retry_backoff_sec", args.llm_retry_backoff_sec, defaults.llm.retry_backoff_sec)),
        batch_size=int(get_value("llm.batch_size", args.batch_size, defaults.llm.batch_size)),
        translate_context_window=int(
            get_value("llm.translate_context_window", args.translate_context_window, defaults.llm.translate_context_window)
        ),
        temperature=float(get_value("llm.temperature", None, defaults.llm.temperature)),
        provider_settings=provider_settings,
    )

    align = AlignConfig(
        delay_sec=float(get_value("align.delay_sec", args.delay_sec, defaults.align.delay_sec)),
        asr_delay_sec=float(get_value("align.asr_delay_sec", args.asr_delay_sec, defaults.align.asr_delay_sec)),
    )
    render = RenderConfig(
        tick_hz=float(get_value("render.tick_hz", args.tick_hz, defaults.render.tick_hz)),
        char_threshold=int(get_value("render.char_threshold", args.char_threshold, defaults.render.char_threshold)),
        max_total_chars=int(
            get_value(
                "render.max_total_chars",
                args.render_max_total_chars,
                defaults.render.max_total_chars,
            )
        ),
        max_lines=int(
            get_value(
                "render.max_lines",
                args.render_max_lines,
                defaults.render.max_lines,
            )
        ),
        two_line_roll_enabled=bool(
            get_value(
                "render.two_line_roll_enabled",
                args.render_two_line_roll_enabled,
                defaults.render.two_line_roll_enabled,
            )
        ),
        min_hold_sec=float(
            get_value(
                "render.min_hold_sec",
                args.render_min_hold_sec,
                defaults.render.min_hold_sec,
            )
        ),
        target_cps=float(
            get_value(
                "render.target_cps",
                args.render_target_cps,
                defaults.render.target_cps,
            )
        ),
        max_hold_sec=float(
            get_value(
                "render.max_hold_sec",
                args.render_max_hold_sec,
                defaults.render.max_hold_sec,
            )
        ),
        backlog_relax_threshold=int(
            get_value(
                "render.backlog_relax_threshold",
                args.render_backlog_relax_threshold,
                defaults.render.backlog_relax_threshold,
            )
        ),
        backlog_relaxed_min_hold_sec=float(
            get_value(
                "render.backlog_relaxed_min_hold_sec",
                args.render_backlog_relaxed_min_hold_sec,
                defaults.render.backlog_relaxed_min_hold_sec,
            )
        ),
    )
    render.max_total_chars = max(0, int(render.max_total_chars))
    render.max_lines = max(1, int(render.max_lines))
    render.min_hold_sec = max(0.1, float(render.min_hold_sec))
    render.target_cps = max(1.0, float(render.target_cps))
    render.max_hold_sec = max(render.min_hold_sec, float(render.max_hold_sec))
    render.backlog_relax_threshold = max(0, int(render.backlog_relax_threshold))
    render.backlog_relaxed_min_hold_sec = max(0.1, min(render.min_hold_sec, float(render.backlog_relaxed_min_hold_sec)))
    pipeline = PipelineConfig(
        batch_max_wait_sec=float(
            get_value("pipeline.batch_max_wait_sec", args.batch_max_wait_sec, defaults.pipeline.batch_max_wait_sec)
        ),
        batch_max_lines=int(get_value("pipeline.batch_max_lines", args.batch_max_lines, defaults.pipeline.batch_max_lines)),
        context_wait_timeout_sec=float(
            get_value(
                "pipeline.context_wait_timeout_sec",
                args.context_wait_timeout_sec,
                defaults.pipeline.context_wait_timeout_sec,
            )
        ),
        s1_mode=normalize_s1_mode(
            str(
                get_value(
                    "pipeline.s1_mode",
                    args.s1_mode,
                    defaults.pipeline.s1_mode,
                )
            )
        ),
        s1_lite_punctuation_only=bool(
            get_value(
                "pipeline.s1_lite_punctuation_only",
                args.s1_lite_punctuation_only,
                defaults.pipeline.s1_lite_punctuation_only,
            )
        ),
        s1_ai_join_marker=str(
            get_value(
                "pipeline.s1_ai_join_marker",
                args.s1_ai_join_marker,
                defaults.pipeline.s1_ai_join_marker,
            )
        ),
        s1_ai_join_allow_light_compress=bool(
            get_value(
                "pipeline.s1_ai_join_allow_light_compress",
                args.s1_ai_join_allow_light_compress,
                defaults.pipeline.s1_ai_join_allow_light_compress,
            )
        ),
        s1_ai_join_max_chain=int(
            get_value(
                "pipeline.s1_ai_join_max_chain",
                args.s1_ai_join_max_chain,
                defaults.pipeline.s1_ai_join_max_chain,
            )
        ),
        s1_ai_join_keep_min_chars=int(
            get_value(
                "pipeline.s1_ai_join_keep_min_chars",
                args.s1_ai_join_keep_min_chars,
                defaults.pipeline.s1_ai_join_keep_min_chars,
            )
        ),
    )
    if not pipeline.s1_ai_join_marker.strip():
        pipeline.s1_ai_join_marker = defaults.pipeline.s1_ai_join_marker
    pipeline.s1_ai_join_max_chain = max(0, int(pipeline.s1_ai_join_max_chain))
    pipeline.s1_ai_join_keep_min_chars = max(0, int(pipeline.s1_ai_join_keep_min_chars))
    adaptive = AdaptiveConfig(
        enabled=bool(get_value("adaptive.enabled", None, defaults.adaptive.enabled)),
        mode=str(get_value("adaptive.mode", args.adaptive_mode, defaults.adaptive.mode)).strip().lower(),
        green_slack_ms=int(get_value("adaptive.green_slack_ms", None, defaults.adaptive.green_slack_ms)),
        red_slack_ms=int(get_value("adaptive.red_slack_ms", None, defaults.adaptive.red_slack_ms)),
        backlog_red_threshold=int(
            get_value("adaptive.backlog_red_threshold", None, defaults.adaptive.backlog_red_threshold)
        ),
        red_parallel_contextless=int(
            get_value("adaptive.red_parallel_contextless", None, defaults.adaptive.red_parallel_contextless)
        ),
        service_rate_floor_lps=float(
            get_value(
                "adaptive.service_rate_floor_lps",
                args.adaptive_service_rate_floor_lps,
                defaults.adaptive.service_rate_floor_lps,
            )
        ),
        tail_quiet_arrival_lps=float(
            get_value(
                "adaptive.tail_quiet_arrival_lps",
                args.adaptive_tail_quiet_arrival_lps,
                defaults.adaptive.tail_quiet_arrival_lps,
            )
        ),
        tail_small_backlog_threshold=int(
            get_value(
                "adaptive.tail_small_backlog_threshold",
                args.adaptive_tail_small_backlog_threshold,
                defaults.adaptive.tail_small_backlog_threshold,
            )
        ),
        tail_red_guard=bool(
            get_value(
                "adaptive.tail_red_guard",
                args.adaptive_tail_red_guard,
                defaults.adaptive.tail_red_guard,
            )
        ),
        red_force_s1_off=bool(
            get_value(
                "adaptive.red_force_s1_off",
                args.adaptive_red_force_s1_off,
                defaults.adaptive.red_force_s1_off,
            )
        ),
    )
    if adaptive.mode not in ("auto", "green", "yellow", "red"):
        raise ValueError(f"Unsupported adaptive mode: {adaptive.mode}")
    adaptive_predict = AdaptivePredictConfig(
        pipeline_service_percentile=int(
            get_value(
                "adaptive_predict.pipeline_service_percentile",
                None,
                defaults.adaptive_predict.pipeline_service_percentile,
            )
        ),
        queue_exclude_current_batch=bool(
            get_value(
                "adaptive_predict.queue_exclude_current_batch",
                None,
                defaults.adaptive_predict.queue_exclude_current_batch,
            )
        ),
        warmup_min_completions=int(
            get_value(
                "adaptive_predict.warmup_min_completions",
                None,
                defaults.adaptive_predict.warmup_min_completions,
            )
        ),
        warmup_nominal_pipeline_ms=int(
            get_value(
                "adaptive_predict.warmup_nominal_pipeline_ms",
                None,
                defaults.adaptive_predict.warmup_nominal_pipeline_ms,
            )
        ),
    )
    if adaptive_predict.pipeline_service_percentile not in (50, 90, 95):
        raise ValueError(
            "adaptive_predict.pipeline_service_percentile must be one of 50/90/95"
        )
    adaptive_predict.warmup_min_completions = max(0, int(adaptive_predict.warmup_min_completions))
    adaptive_predict.warmup_nominal_pipeline_ms = max(1000, int(adaptive_predict.warmup_nominal_pipeline_ms))
    adaptive_risk = AdaptiveRiskConfig(
        green_max_ratio=float(
            get_value(
                "adaptive_risk.green_max_ratio",
                None,
                defaults.adaptive_risk.green_max_ratio,
            )
        ),
        yellow_max_ratio=float(
            get_value(
                "adaptive_risk.yellow_max_ratio",
                None,
                defaults.adaptive_risk.yellow_max_ratio,
            )
        ),
        tail_guard_ratio=float(
            get_value(
                "adaptive_risk.tail_guard_ratio",
                None,
                defaults.adaptive_risk.tail_guard_ratio,
            )
        ),
    )
    adaptive_risk.green_max_ratio = max(0.1, float(adaptive_risk.green_max_ratio))
    adaptive_risk.yellow_max_ratio = max(adaptive_risk.green_max_ratio, float(adaptive_risk.yellow_max_ratio))
    adaptive_risk.tail_guard_ratio = max(adaptive_risk.yellow_max_ratio, float(adaptive_risk.tail_guard_ratio))
    adaptive_fast = AdaptiveFastConfig(
        block_when_latest_ready_ratio=float(
            get_value(
                "adaptive_fast.block_when_latest_ready_ratio",
                None,
                defaults.adaptive_fast.block_when_latest_ready_ratio,
            )
        ),
        emergency_when_latest_ready_ratio=float(
            get_value(
                "adaptive_fast.emergency_when_latest_ready_ratio",
                None,
                defaults.adaptive_fast.emergency_when_latest_ready_ratio,
            )
        ),
        allow_fast_if_overdue=bool(
            get_value(
                "adaptive_fast.allow_fast_if_overdue",
                None,
                defaults.adaptive_fast.allow_fast_if_overdue,
            )
        ),
    )
    adaptive_fast.block_when_latest_ready_ratio = max(
        0.0, float(adaptive_fast.block_when_latest_ready_ratio)
    )
    adaptive_fast.emergency_when_latest_ready_ratio = max(
        0.0, float(adaptive_fast.emergency_when_latest_ready_ratio)
    )
    if adaptive_fast.emergency_when_latest_ready_ratio > adaptive_fast.block_when_latest_ready_ratio:
        adaptive_fast.emergency_when_latest_ready_ratio = adaptive_fast.block_when_latest_ready_ratio
    console = ConsoleConfig(
        enabled=bool(get_value("console.enabled", args.console_enabled, defaults.console.enabled)),
        host=str(get_value("console.host", args.console_host, defaults.console.host)),
        port=int(get_value("console.port", args.console_port, defaults.console.port)),
    )
    runtime_scope = RuntimeScopeConfig(
        enabled=bool(get_value("runtime_scope.enabled", None, defaults.runtime_scope.enabled)),
        only_active_source_kind=bool(
            get_value(
                "runtime_scope.only_active_source_kind",
                None,
                defaults.runtime_scope.only_active_source_kind,
            )
        ),
        freshness_sec=float(get_value("runtime_scope.freshness_sec", None, defaults.runtime_scope.freshness_sec)),
        monotonic_guard_sec=float(
            get_value("runtime_scope.monotonic_guard_sec", None, defaults.runtime_scope.monotonic_guard_sec)
        ),
        auto_archive_stale_on_start=bool(
            get_value(
                "runtime_scope.auto_archive_stale_on_start",
                None,
                defaults.runtime_scope.auto_archive_stale_on_start,
            )
        ),
    )
    runtime_scope.freshness_sec = max(0.0, float(runtime_scope.freshness_sec))
    runtime_scope.monotonic_guard_sec = max(0.0, float(runtime_scope.monotonic_guard_sec))
    state = StateConfig(db_path=Path(get_value("state.db_path", args.state_db, defaults.state.db_path)))
    fallback = FallbackConfig(mode=str(get_value("fallback.mode", args.fallback_mode, defaults.fallback.mode)))
    knowledge = KnowledgeConfig(
        glossary_path=Path(get_value("knowledge.glossary_path", args.glossary, defaults.knowledge.glossary_path)),
        names_path=Path(get_value("knowledge.names_path", args.names_whitelist, defaults.knowledge.names_path)),
        reload_interval_sec=float(
            get_value("knowledge.reload_interval_sec", args.knowledge_reload_sec, defaults.knowledge.reload_interval_sec)
        ),
        correct_glossary_limit=int(
            get_value(
                "knowledge.correct_glossary_limit",
                args.knowledge_correct_glossary_limit,
                defaults.knowledge.correct_glossary_limit,
            )
        ),
        translate_glossary_limit=int(
            get_value(
                "knowledge.translate_glossary_limit",
                args.knowledge_translate_glossary_limit,
                defaults.knowledge.translate_glossary_limit,
            )
        ),
        names_limit=int(
            get_value(
                "knowledge.names_limit",
                args.knowledge_names_limit,
                defaults.knowledge.names_limit,
            )
        ),
        names_phonetic_canonicalize_enabled=bool(
            get_value(
                "knowledge.names_phonetic_canonicalize_enabled",
                args.knowledge_names_phonetic_canonicalize_enabled,
                defaults.knowledge.names_phonetic_canonicalize_enabled,
            )
        ),
        names_phonetic_max_rules=int(
            get_value(
                "knowledge.names_phonetic_max_rules",
                args.knowledge_names_phonetic_max_rules,
                defaults.knowledge.names_phonetic_max_rules,
            )
        ),
    )
    metrics = MetricsConfig(
        trace_enabled=bool(
            get_value(
                "metrics.trace_enabled",
                args.metrics_trace_enabled,
                defaults.metrics.trace_enabled,
            )
        ),
        trace_path=Path(
            get_value(
                "metrics.trace_path",
                args.metrics_trace_path,
                defaults.metrics.trace_path,
            )
        ),
        trace_interval_sec=float(
            get_value(
                "metrics.trace_interval_sec",
                args.metrics_trace_interval_sec,
                defaults.metrics.trace_interval_sec,
            )
        ),
    )
    metrics.trace_interval_sec = max(0.2, float(metrics.trace_interval_sec))

    return AppConfig(
        source=source,
        output=output,
        llm=llm,
        align=align,
        render=render,
        pipeline=pipeline,
        adaptive=adaptive,
        adaptive_predict=adaptive_predict,
        adaptive_risk=adaptive_risk,
        adaptive_fast=adaptive_fast,
        console=console,
        state=state,
        fallback=fallback,
        knowledge=knowledge,
        runtime_scope=runtime_scope,
        metrics=metrics,
    )


def _default_config() -> AppConfig:
    return AppConfig(
        source=SourceConfig(input_srt=Path("jp.srt"), input_txt=Path("jp.txt")),
        output=OutputConfig(zh_txt=Path("zh.txt"), zh_srt=Path("zh.srt")),
        llm=LLMConfig(api_key=""),
        align=AlignConfig(),
        render=RenderConfig(),
        pipeline=PipelineConfig(),
        adaptive=AdaptiveConfig(),
        adaptive_predict=AdaptivePredictConfig(),
        adaptive_risk=AdaptiveRiskConfig(),
        adaptive_fast=AdaptiveFastConfig(),
        console=ConsoleConfig(),
        state=StateConfig(db_path=Path("state.sqlite3")),
        fallback=FallbackConfig(),
        knowledge=KnowledgeConfig(
            glossary_path=Path("glossary_ja_zh.tsv"),
            names_path=Path("names_whitelist_ja.txt"),
        ),
        metrics=MetricsConfig(),
    )


def _nested_get(data: Dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _parse_toml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return _parse_toml_minimal(text)


def _parse_toml_minimal(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    current: Dict[str, Any] = root

    for raw_line in text.splitlines():
        if raw_line.startswith("\ufeff"):
            raw_line = raw_line.lstrip("\ufeff")
        line = _strip_comments(raw_line).strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            section_path = line[1:-1].strip()
            if not section_path:
                raise ValueError("Empty TOML section is invalid")
            current = root
            for part in section_path.split("."):
                current = current.setdefault(part.strip(), {})
            continue

        if "=" not in line:
            raise ValueError(f"Invalid TOML line: {raw_line}")
        key, raw_value = line.split("=", 1)
        current[key.strip()] = _parse_scalar(raw_value.strip())

    return root


def _strip_comments(line: str) -> str:
    in_string = False
    escaped = False
    for i, ch in enumerate(line):
        if ch == "\\" and in_string and not escaped:
            escaped = True
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
        if ch == "#" and not in_string:
            return line[:i]
        escaped = False
    return line


def _parse_scalar(raw: str) -> Any:
    if raw.startswith('"') and raw.endswith('"'):
        inner = raw[1:-1]
        inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _default_base_url_for_provider(provider: str) -> str:
    return PROVIDER_DEFAULT_BASE_URL.get(provider, PROVIDER_DEFAULT_BASE_URL["xai"])


def _default_model_for_provider(provider: str) -> str:
    return PROVIDER_DEFAULT_MODEL.get(provider, PROVIDER_DEFAULT_MODEL["xai"])


def _resolve_api_key_env(args: argparse.Namespace, config_data: Dict[str, Any], provider: str) -> str:
    if args.api_key_env:
        return args.api_key_env
    env_from_config = _nested_get(config_data, "llm.api_key_env")
    if isinstance(env_from_config, str) and env_from_config.strip():
        return env_from_config.strip()
    provider_env = _provider_get(config_data, provider, "api_key_env")
    if isinstance(provider_env, str) and provider_env.strip():
        return provider_env.strip()
    return PROVIDER_DEFAULT_API_KEY_ENV.get(provider, DEFAULT_API_KEY_ENV)


def _resolve_llm_value(cli_value: Any, llm_value: Any, provider_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if llm_value is not None:
        return llm_value
    if provider_value is not None:
        return provider_value
    return default_value


def _build_provider_settings(
    args: argparse.Namespace,
    config_data: Dict[str, Any],
    selected_provider: str,
    selected_api_key: str,
    selected_base_url: str,
    selected_correct_model: str,
    selected_translate_model: str,
    selected_fast_correct_model: str,
    selected_fast_translate_model: str,
) -> Dict[str, ProviderRuntimeConfig]:
    settings: Dict[str, ProviderRuntimeConfig] = {}

    for provider in SUPPORTED_LLM_PROVIDERS:
        default_base_url = _default_base_url_for_provider(provider)
        default_model = _default_model_for_provider(provider)

        provider_base_url = str(_provider_get(config_data, provider, "base_url") or default_base_url)
        provider_correct_model = str(
            _provider_get(config_data, provider, "correct_model")
            or _provider_get(config_data, provider, "default_model")
            or default_model
        )
        provider_translate_model = str(
            _provider_get(config_data, provider, "translate_model")
            or _provider_get(config_data, provider, "default_model")
            or default_model
        )
        provider_fast_correct_model = str(_provider_get(config_data, provider, "fast_correct_model") or "")
        provider_fast_translate_model = str(_provider_get(config_data, provider, "fast_translate_model") or "")

        provider_api_key_env_raw = _provider_get(config_data, provider, "api_key_env")
        provider_api_key_env = (
            str(provider_api_key_env_raw).strip()
            if isinstance(provider_api_key_env_raw, str) and provider_api_key_env_raw.strip()
            else PROVIDER_DEFAULT_API_KEY_ENV.get(provider, DEFAULT_API_KEY_ENV)
        )

        provider_api_key = str(_provider_get(config_data, provider, "api_key") or "")
        if not provider_api_key:
            provider_api_key = os.getenv(provider_api_key_env, "")

        settings[provider] = ProviderRuntimeConfig(
            provider=provider,
            base_url=provider_base_url,
            api_key=provider_api_key,
            api_key_env=provider_api_key_env,
            correct_model=provider_correct_model,
            translate_model=provider_translate_model,
            fast_correct_model=provider_fast_correct_model,
            fast_translate_model=provider_fast_translate_model,
        )

    selected = settings[selected_provider]
    selected.api_key = selected_api_key
    selected.base_url = selected_base_url
    selected.correct_model = selected_correct_model
    selected.translate_model = selected_translate_model
    selected.fast_correct_model = selected_fast_correct_model
    selected.fast_translate_model = selected_fast_translate_model
    selected.api_key_env = _resolve_api_key_env(args, config_data, selected_provider)

    return settings


def _provider_get(config_data: Dict[str, Any], provider: str, key: str) -> Any:
    provider_cfg = _nested_get(config_data, f"providers.{provider}")
    if isinstance(provider_cfg, dict):
        return provider_cfg.get(key)
    return None


def normalize_provider_name(name: str) -> str:
    raw = str(name).strip().lower()
    return PROVIDER_ALIASES.get(raw, raw)


def normalize_speechmatics_vocab_source(name: str) -> str:
    raw = str(name).strip().lower()
    normalized = SPEECHMATICS_VOCAB_SOURCE_ALIASES.get(raw, raw)
    if normalized not in SUPPORTED_SPEECHMATICS_VOCAB_SOURCES:
        allowed = ", ".join(SUPPORTED_SPEECHMATICS_VOCAB_SOURCES)
        aliases = ", ".join(sorted(SPEECHMATICS_VOCAB_SOURCE_ALIASES.keys()))
        raise ValueError(
            f"Unsupported source.speechmatics.additional_vocab_from: {name}. "
            f"Supported: {allowed}; aliases: {aliases}"
        )
    return normalized


def normalize_speechmatics_max_delay_mode(name: str) -> str:
    raw = str(name).strip().lower()
    if raw not in SUPPORTED_SPEECHMATICS_MAX_DELAY_MODES:
        allowed = ", ".join(SUPPORTED_SPEECHMATICS_MAX_DELAY_MODES)
        raise ValueError(
            f"Unsupported source.speechmatics.max_delay_mode: {name}. Supported: {allowed}"
        )
    return raw


def normalize_speechmatics_aggregate_mode(name: str) -> str:
    raw = str(name).strip().lower()
    if raw not in SUPPORTED_SPEECHMATICS_AGGREGATE_MODES:
        allowed = ", ".join(SUPPORTED_SPEECHMATICS_AGGREGATE_MODES)
        raise ValueError(
            f"Unsupported source.speechmatics.final_aggregate_mode: {name}. Supported: {allowed}"
        )
    return raw


def normalize_s1_mode(name: str) -> str:
    raw = str(name).strip().lower()
    if raw not in SUPPORTED_S1_MODES:
        allowed = ", ".join(SUPPORTED_S1_MODES)
        raise ValueError(
            f"Unsupported pipeline.s1_mode: {name}. Supported: {allowed}"
        )
    return raw
