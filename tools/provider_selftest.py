from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_sub_daemon.config import load_config, normalize_provider_name, parse_cli_args
from live_sub_daemon.knowledge import KnowledgeStore
from live_sub_daemon.llm_client import XAIClient, is_qwen_mt_model
from live_sub_daemon.pipeline import (
    _compose_translate_input,
    build_qwen_mt_translation_options,
    build_correct_prompt,
    build_translate_prompt,
)
from live_sub_daemon.source_reader import SourceReader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local provider connectivity + latency self-test")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--srt", default="jp.srt")
    parser.add_argument("--providers", default="deepseek,qwen")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model-override", default=None, help="Override correct+translate model for all providers")
    parser.add_argument(
        "--probe-text",
        default="と思います。注目していきましょう。はい。",
        help="Used for connectivity probe",
    )
    return parser.parse_args()


def _load_first_cues(srt_path: Path, count: int):
    reader = SourceReader(srt_path=srt_path, txt_path=Path("__unused__.txt"))
    cues = [c for c in reader.poll() if c.source_kind == "srt"]
    return cues[: max(1, count)]


def _print_header(title: str) -> None:
    print("")
    print("=" * 78)
    print(title)
    print("=" * 78)


def _build_cfg(config_path: str, provider: str, args: argparse.Namespace):
    cli_args: List[str] = [
        "--config",
        config_path,
        "--provider",
        provider,
        "--correct-timeout-sec",
        str(args.timeout_sec),
        "--translate-timeout-sec",
        str(args.timeout_sec),
        "--llm-max-retries",
        str(args.max_retries),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.model_override:
        cli_args.extend(
            [
                "--model-correct",
                args.model_override,
                "--model-translate",
                args.model_override,
            ]
        )
    return load_config(parse_cli_args(cli_args))


def _run_provider(provider: str, args: argparse.Namespace, cues) -> None:
    _print_header(f"Provider: {provider}")
    try:
        cfg = _build_cfg(args.config, provider, args)
    except Exception as exc:  # noqa: BLE001
        print(f"[config] failed: {exc}")
        return

    print(f"base_url: {cfg.llm.base_url}")
    print(f"correct_model: {cfg.llm.correct_model}")
    print(f"translate_model: {cfg.llm.translate_model}")
    print(f"timeout_sec: {cfg.llm.correct_timeout_sec}, max_retries: {cfg.llm.max_retries}")

    client = XAIClient(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)

    # Probe 1: minimal connectivity test.
    print("")
    print("[probe-1] connectivity")
    probe_t0 = time.monotonic()
    probe_result = client.run_batch(
        model=cfg.llm.translate_model,
        stage_name="connectivity_probe",
        system_prompt="Return strict JSON array of {source_key,text}.",
        items=[{"source_key": "probe-1", "text": args.probe_text}],
        timeout_sec=cfg.llm.translate_timeout_sec,
        max_retries=cfg.llm.max_retries,
        retry_backoff_sec=cfg.llm.retry_backoff_sec,
        temperature=cfg.llm.temperature,
    )
    probe_t1 = time.monotonic()
    print(f"ok={probe_result.ok} timed_out={probe_result.timed_out} latency_ms={probe_result.latency_ms}")
    print(f"wall_ms={int((probe_t1 - probe_t0) * 1000)} error={probe_result.error}")
    print(f"input: {args.probe_text}")
    print(f"output: {probe_result.outputs.get('probe-1', '')}")

    if not cues:
        print("")
        print("[probe-2] skipped: no cues found in SRT")
        return

    # Probe 2: real pipeline mini-batch (Stage1 + Stage2).
    print("")
    print("[probe-2] mini pipeline batch")
    knowledge = KnowledgeStore(
        glossary_path=cfg.knowledge.glossary_path,
        names_path=cfg.knowledge.names_path,
        reload_interval_sec=cfg.knowledge.reload_interval_sec,
    )
    snapshot = knowledge.get_snapshot()
    correct_prompt = build_correct_prompt(
        snapshot.glossary,
        snapshot.names_whitelist,
        glossary_limit=cfg.knowledge.correct_glossary_limit,
        names_limit=cfg.knowledge.names_limit,
    )
    translate_prompt = build_translate_prompt(
        snapshot.glossary,
        snapshot.names_whitelist,
        glossary_limit=cfg.knowledge.translate_glossary_limit,
        names_limit=cfg.knowledge.names_limit,
    )
    use_qwen_mt = is_qwen_mt_model(cfg.llm.translate_model)
    translation_options = None
    if use_qwen_mt:
        translate_prompt = ""
        translation_options = build_qwen_mt_translation_options(
            snapshot.glossary,
            snapshot.names_whitelist,
            glossary_limit=cfg.knowledge.translate_glossary_limit,
            names_limit=cfg.knowledge.names_limit,
        )

    batch = cues[: cfg.llm.batch_size]
    stage1_items = [{"source_key": c.source_key, "text": c.jp_raw} for c in batch]

    s1_t0 = time.monotonic()
    stage1 = client.run_batch(
        model=cfg.llm.correct_model,
        stage_name="jp_correct",
        system_prompt=correct_prompt,
        items=stage1_items,
        timeout_sec=cfg.llm.correct_timeout_sec,
        max_retries=cfg.llm.max_retries,
        retry_backoff_sec=cfg.llm.retry_backoff_sec,
        temperature=cfg.llm.temperature,
    )
    s1_t1 = time.monotonic()
    print(f"stage1_ok={stage1.ok} stage1_ms={stage1.latency_ms} wall_ms={int((s1_t1 - s1_t0) * 1000)}")
    if stage1.error:
        print(f"stage1_error={stage1.error}")

    corrected = {}
    for cue in batch:
        corrected[cue.source_key] = stage1.outputs.get(cue.source_key, cue.jp_raw)

    context_window = 0 if use_qwen_mt else max(0, cfg.llm.translate_context_window)
    rolling = []
    stage2_items = []
    for cue in batch:
        curr = corrected[cue.source_key]
        ctx = rolling[-context_window:] if context_window > 0 else []
        translated_input = curr if use_qwen_mt else _compose_translate_input(curr, ctx)
        stage2_items.append({"source_key": cue.source_key, "text": translated_input})
        rolling.append(curr)

    s2_t0 = time.monotonic()
    stage2 = client.run_batch(
        model=cfg.llm.translate_model,
        stage_name="jp_to_zh",
        system_prompt=translate_prompt,
        items=stage2_items,
        timeout_sec=cfg.llm.translate_timeout_sec,
        max_retries=cfg.llm.max_retries,
        retry_backoff_sec=cfg.llm.retry_backoff_sec,
        temperature=cfg.llm.temperature,
        translation_options=translation_options,
    )
    s2_t1 = time.monotonic()
    print(f"stage2_ok={stage2.ok} stage2_ms={stage2.latency_ms} wall_ms={int((s2_t1 - s2_t0) * 1000)}")
    if stage2.error:
        print(f"stage2_error={stage2.error}")

    print("")
    print("sample outputs (first 3 cues):")
    for cue in batch[:3]:
        sk = cue.source_key
        print(f"- {sk}")
        print(f"  jp_raw:   {cue.jp_raw}")
        print(f"  jp_fixed: {corrected.get(sk, '')}")
        print(f"  zh_out:   {stage2.outputs.get(sk, '')}")


def main() -> int:
    args = _parse_args()
    providers = [normalize_provider_name(p.strip()) for p in args.providers.split(",") if p.strip()]
    providers = list(dict.fromkeys(providers))
    if not providers:
        raise ValueError("--providers is empty")

    cues = _load_first_cues(Path(args.srt), args.batch_size)
    print(f"SRT: {args.srt}, loaded cues: {len(cues)}")

    for provider in providers:
        _run_provider(provider, args, cues)

    print("")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
