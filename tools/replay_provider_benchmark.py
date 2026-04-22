from __future__ import annotations

import argparse
import threading
import json
import math
import subprocess
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence
from collections import deque

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_sub_daemon.config import SUPPORTED_LLM_PROVIDERS, load_config, normalize_provider_name, parse_cli_args
from live_sub_daemon.knowledge import KnowledgeStore
from live_sub_daemon.llm_client import XAIClient, is_qwen_mt_model
from live_sub_daemon.pipeline import (
    _compose_translate_input,
    build_qwen_mt_translation_options,
    build_correct_prompt,
    build_translate_prompt,
)
from live_sub_daemon.source_reader import SourceReader


@dataclass
class ReplayCue:
    cue_index: int
    source_key: str
    srt_start_ms: int
    srt_end_ms: int
    jp_raw: str


@dataclass
class CueResult:
    provider: str
    cue_index: int
    source_key: str
    srt_start_ms: int
    srt_end_ms: int
    seen_mono_ms: int
    batch_id: int
    stage1_start_ms: int
    stage1_end_ms: int
    stage1_latency_ms: int
    stage2_start_ms: int
    stage2_end_ms: int
    stage2_latency_ms: int
    queue_wait_ms: int
    emit_jitter_ms: int
    due_mono_ms: int
    eligible_mono_ms: int
    late: bool
    late_by_ms: int
    jp_raw: str
    jp_fixed: str
    zh_out: str
    status: str
    error: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    providers = [normalize_provider_name(x.strip()) for x in args.providers.split(",") if x.strip()]
    providers = list(dict.fromkeys(providers))
    if not providers:
        raise ValueError("--providers is empty")
    unsupported = [p for p in providers if p not in SUPPORTED_LLM_PROVIDERS]
    if unsupported and args.fail_on_missing_key and args.engine == "legacy":
        raise ValueError(f"Unsupported providers: {unsupported}. Supported: {list(SUPPORTED_LLM_PROVIDERS)}")

    srt_path = Path(args.srt)
    cues = _load_replay_cues(srt_path)
    if not cues:
        raise ValueError(f"No cues parsed from {srt_path}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("benchmark_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.engine == "daemon":
        return _run_daemon_engine(args=args, providers=providers, out_dir=out_dir, unsupported=unsupported)

    print(f"[bench] cues={len(cues)} providers={providers} speed={args.speed} out_dir={out_dir}")

    provider_results: Dict[str, List[CueResult]] = {}
    skipped: Dict[str, str] = {}

    for provider in unsupported:
        message = f"unsupported provider, supported={list(SUPPORTED_LLM_PROVIDERS)}"
        print(f"[bench] skip {provider}: {message}")
        skipped[provider] = message

    providers = [p for p in providers if p in SUPPORTED_LLM_PROVIDERS]

    for provider in providers:
        try:
            cfg = _load_provider_config(
                config_path=Path(args.config),
                provider=provider,
                batch_size=args.batch_size,
                context_window=args.translate_context_window,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"config load failed: {exc}"
            if args.fail_on_missing_key:
                raise RuntimeError(f"Provider {provider}: {message}") from exc
            print(f"[bench] skip {provider}: {message}")
            skipped[provider] = message
            continue

        try:
            results = _run_provider_replay(provider=provider, cfg=cfg, cues=cues, speed=args.speed, out_dir=out_dir)
            provider_results[provider] = results
        except Exception as exc:  # noqa: BLE001
            if args.fail_on_missing_key:
                raise
            message = f"runtime failed: {exc}"
            print(f"[bench] skip {provider}: {message}")
            skipped[provider] = message

    _write_summary(out_dir=out_dir, provider_results=provider_results, skipped=skipped)
    print(f"[bench] done summary={out_dir / 'summary.md'}")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay benchmark across providers with real-time SRT pacing")
    parser.add_argument("--config", type=str, default="config.toml")
    parser.add_argument("--srt", type=str, default="jp.srt")
    parser.add_argument("--providers", type=str, default="xai,deepseek,qwen")
    parser.add_argument("--engine", type=str, choices=["legacy", "daemon"], default="daemon")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--translate-context-window", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--daemon-log-interval-sec", type=float, default=1.0)
    parser.add_argument("--console-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-host", type=str, default=None)
    parser.add_argument("--console-port", type=int, default=None)
    parser.add_argument("--hold-after-complete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--fail-on-missing-key", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _run_daemon_engine(
    *,
    args: argparse.Namespace,
    providers: List[str],
    out_dir: Path,
    unsupported: List[str],
) -> int:
    if unsupported and args.fail_on_missing_key:
        raise ValueError(f"Unsupported providers: {unsupported}. Supported: {list(SUPPORTED_LLM_PROVIDERS)}")

    skipped: Dict[str, str] = {}
    provider_metrics: Dict[str, Dict[str, float]] = {}

    for provider in unsupported:
        skipped[provider] = "unsupported provider"

    runnable = [p for p in providers if p in SUPPORTED_LLM_PROVIDERS]
    if not runnable:
        raise ValueError("No runnable providers")

    print(f"[bench-daemon] providers={runnable} speed={args.speed} out_dir={out_dir}")
    for idx, provider in enumerate(runnable):
        provider_dir = out_dir / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "daemon_replay_runner.py"),
            "--config",
            str(args.config),
            "--srt",
            str(args.srt),
            "--speed",
            str(args.speed),
            "--out-dir",
            str(provider_dir),
            "--provider",
            provider,
            "--correct-provider",
            provider,
            "--translate-provider",
            provider,
            "--translate-fallback-provider",
            provider,
            "--translate-fallback-on-error",
            "--log-interval-sec",
            str(args.daemon_log_interval_sec),
        ]
        if args.duration_sec is not None and args.duration_sec > 0:
            cmd.extend(["--duration-sec", str(args.duration_sec)])

        if args.console_enabled is not None:
            cmd.append("--console-enabled" if args.console_enabled else "--no-console-enabled")
        if args.console_host:
            cmd.extend(["--console-host", str(args.console_host)])
        if args.console_port is not None:
            # Optional per-provider offset to avoid reuse issues during rapid reruns.
            cmd.extend(["--console-port", str(int(args.console_port) + idx)])

        if args.hold_after_complete is not None:
            cmd.append("--hold-after-complete" if args.hold_after_complete else "--no-hold-after-complete")

        print(f"[bench-daemon] run {provider}: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (provider_dir / "daemon.stdout.log").write_text(proc.stdout, encoding="utf-8")
        (provider_dir / "daemon.stderr.log").write_text(proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            message = f"daemon_replay_runner failed rc={proc.returncode}"
            if args.fail_on_missing_key:
                raise RuntimeError(f"{provider}: {message}\n{proc.stderr}")
            skipped[provider] = message
            continue

        summary_path = provider_dir / "summary.md"
        provider_metrics[provider] = _parse_daemon_summary(summary_path) if summary_path.exists() else {}

    _write_daemon_engine_summary(out_dir=out_dir, metrics=provider_metrics, skipped=skipped)
    print(f"[bench-daemon] done summary={out_dir / 'summary.md'}")
    return 0


def _parse_daemon_summary(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text.startswith("- total:"):
            metrics["total"] = _parse_float_tail(text)
        elif text.startswith("- ok:"):
            metrics["ok"] = _parse_float_tail(text)
        elif text.startswith("- late:"):
            metrics["late"] = _parse_float_tail(text)
        elif text.startswith("- pipeline_p95_ms:"):
            metrics["pipeline_p95_ms"] = _parse_float_tail(text)
        elif text.startswith("- stage1_p95_ms:"):
            metrics["stage1_p95_ms"] = _parse_float_tail(text)
        elif text.startswith("- stage2_p95_ms:"):
            metrics["stage2_p95_ms"] = _parse_float_tail(text)
    return metrics


def _parse_float_tail(line: str) -> float:
    try:
        return float(line.split(":", 1)[1].strip())
    except Exception:  # noqa: BLE001
        return 0.0


def _write_daemon_engine_summary(out_dir: Path, metrics: Dict[str, Dict[str, float]], skipped: Dict[str, str]) -> None:
    summary_path = out_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write("# Replay Benchmark Summary (Daemon Engine)\n\n")
        fp.write("Each provider is executed through `tools/daemon_replay_runner.py` so it uses the same runtime/state/console pipeline as production.\n\n")
        fp.write("## Providers\n")
        for provider, row in metrics.items():
            total = int(row.get("total", 0))
            ok = int(row.get("ok", 0))
            late = int(row.get("late", 0))
            p95 = row.get("pipeline_p95_ms", 0.0)
            fp.write(f"- `{provider}` total={total} ok={ok} late={late} pipeline_p95_ms={p95:.1f} ")
            fp.write(f"(detail: `{provider}/summary.md`)\n")
        if not metrics:
            fp.write("- (no successful provider run)\n")
        fp.write("\n")
        if skipped:
            fp.write("## Skipped/Failed\n")
            for provider, reason in skipped.items():
                fp.write(f"- `{provider}`: {reason}\n")


def _load_replay_cues(srt_path: Path) -> List[ReplayCue]:
    reader = SourceReader(srt_path=srt_path, txt_path=Path("__unused__.txt"))
    source_cues = [c for c in reader.poll() if c.source_kind == "srt" and c.start_ms is not None and c.end_ms is not None]
    replay: List[ReplayCue] = []
    for idx, cue in enumerate(source_cues, start=1):
        replay.append(
            ReplayCue(
                cue_index=idx,
                source_key=cue.source_key,
                srt_start_ms=int(cue.start_ms),
                srt_end_ms=int(cue.end_ms),
                jp_raw=cue.jp_raw,
            )
        )
    return replay


def _load_provider_config(config_path: Path, provider: str, batch_size: Optional[int], context_window: Optional[int]):
    cli = ["--config", str(config_path), "--provider", provider]
    if batch_size is not None:
        cli.extend(["--batch-size", str(batch_size)])
    if context_window is not None:
        cli.extend(["--translate-context-window", str(context_window)])
    return load_config(parse_cli_args(cli))


def _run_provider_replay(provider: str, cfg, cues: List[ReplayCue], speed: float, out_dir: Path) -> List[CueResult]:
    if speed <= 0:
        raise ValueError("speed must be > 0")

    provider_dir = out_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    live_path = provider_dir / f"{provider}.live.txt"
    full_path = provider_dir / f"{provider}.full.jsonl"

    knowledge = KnowledgeStore(
        glossary_path=cfg.knowledge.glossary_path,
        names_path=cfg.knowledge.names_path,
        reload_interval_sec=cfg.knowledge.reload_interval_sec,
    )
    snapshot = knowledge.get_snapshot()

    client = XAIClient(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url)
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

    effective_delay_ms = int(max(0.0, cfg.align.delay_sec - cfg.align.asr_delay_sec) * 1000)
    batch_size = max(1, int(cfg.llm.batch_size))
    context_window = 0 if use_qwen_mt else max(0, int(cfg.llm.translate_context_window))

    results: List[CueResult] = []
    context_history: List[str] = []

    s0_ms = cues[0].srt_start_ms
    t0 = time.monotonic()
    pending_queue: Deque[tuple[ReplayCue, int, int]] = deque()
    queue_lock = threading.Lock()
    emitter_done = threading.Event()
    batch_id = 0

    with live_path.open("w", encoding="utf-8") as live_fp, full_path.open("w", encoding="utf-8") as full_fp:
        _write_live_header(live_fp, provider, cfg, effective_delay_ms, speed, len(cues))

        emitter = threading.Thread(
            target=_emit_cues_realtime,
            args=(cues, t0, s0_ms, speed, pending_queue, queue_lock, emitter_done),
            daemon=True,
            name=f"emitter-{provider}",
        )
        emitter.start()

        while not emitter_done.is_set() or _queue_size(pending_queue, queue_lock) > 0:
            batch = _pop_batch(pending_queue, queue_lock, batch_size)
            if not batch:
                time.sleep(0.01)
                continue

            batch_id += 1
            batch_results = _process_batch(
                provider=provider,
                cfg=cfg,
                client=client,
                correct_prompt=correct_prompt,
                translate_prompt=translate_prompt,
                batch=batch,
                batch_id=batch_id,
                context_history=context_history,
                context_window=context_window,
                use_qwen_mt=use_qwen_mt,
                translation_options=translation_options,
                effective_delay_ms=effective_delay_ms,
            )
            for item in batch_results:
                results.append(item)
                _append_live(live_fp, item)
                full_fp.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
                full_fp.flush()

        emitter.join(timeout=1.0)

    return results


def _process_batch(
    provider: str,
    cfg,
    client: XAIClient,
    correct_prompt: str,
    translate_prompt: str,
    batch: List[tuple[ReplayCue, int, int]],
    batch_id: int,
    context_history: List[str],
    context_window: int,
    use_qwen_mt: bool,
    translation_options: Optional[dict[str, object]],
    effective_delay_ms: int,
) -> List[CueResult]:
    stage1_start = int(time.monotonic() * 1000)
    correct_items = [{"source_key": cue.source_key, "text": cue.jp_raw} for cue, _, _ in batch]
    stage1 = client.run_batch(
        model=cfg.llm.correct_model,
        stage_name="jp_correct",
        system_prompt=correct_prompt,
        items=correct_items,
        timeout_sec=cfg.llm.correct_timeout_sec,
        max_retries=cfg.llm.max_retries,
        retry_backoff_sec=cfg.llm.retry_backoff_sec,
        temperature=cfg.llm.temperature,
    )
    stage1_end = int(time.monotonic() * 1000)

    corrected: Dict[str, str] = {}
    for cue, _, _ in batch:
        corrected[cue.source_key] = stage1.outputs.get(cue.source_key, cue.jp_raw)

    rolling_context = list(context_history)
    translate_items: List[dict[str, str]] = []
    for cue, _, _ in batch:
        current = corrected[cue.source_key]
        context_lines = rolling_context[-context_window:] if context_window > 0 else []
        translated_input = current if use_qwen_mt else _compose_translate_input(current, context_lines)
        translate_items.append({"source_key": cue.source_key, "text": translated_input})
        rolling_context.append(current)
    if not use_qwen_mt and context_window > 0:
        context_history[:] = rolling_context[-context_window:]

    stage2_start = int(time.monotonic() * 1000)
    stage2 = client.run_batch(
        model=cfg.llm.translate_model,
        stage_name="jp_to_zh",
        system_prompt=translate_prompt,
        items=translate_items,
        timeout_sec=cfg.llm.translate_timeout_sec,
        max_retries=cfg.llm.max_retries,
        retry_backoff_sec=cfg.llm.retry_backoff_sec,
        temperature=cfg.llm.temperature,
        translation_options=translation_options,
    )
    stage2_end = int(time.monotonic() * 1000)

    results: List[CueResult] = []
    for cue, seen_mono_ms, emit_jitter_ms in batch:
        jp_fixed = corrected[cue.source_key]
        if stage2.outputs.get(cue.source_key) is not None:
            zh_out = stage2.outputs[cue.source_key]
        else:
            if cfg.fallback.mode == "empty":
                zh_out = ""
            else:
                zh_out = cue.jp_raw

        status = "ok"
        error_parts: List[str] = []
        if not stage1.ok:
            status = "stage1_failed"
            if stage1.error:
                error_parts.append(f"stage1:{stage1.error}")
        if not stage2.ok:
            status = "stage2_failed" if status == "ok" else status
            if stage2.error:
                error_parts.append(f"stage2:{stage2.error}")

        queue_wait_ms = max(0, stage1_start - seen_mono_ms)
        due_mono_ms = seen_mono_ms + effective_delay_ms
        eligible_mono_ms = stage2_end
        late = eligible_mono_ms > due_mono_ms
        late_by_ms = max(0, eligible_mono_ms - due_mono_ms)

        results.append(
            CueResult(
                provider=provider,
                cue_index=cue.cue_index,
                source_key=cue.source_key,
                srt_start_ms=cue.srt_start_ms,
                srt_end_ms=cue.srt_end_ms,
                seen_mono_ms=seen_mono_ms,
                batch_id=batch_id,
                stage1_start_ms=stage1_start,
                stage1_end_ms=stage1_end,
                stage1_latency_ms=max(0, stage1_end - stage1_start),
                stage2_start_ms=stage2_start,
                stage2_end_ms=stage2_end,
                stage2_latency_ms=max(0, stage2_end - stage2_start),
                queue_wait_ms=queue_wait_ms,
                emit_jitter_ms=emit_jitter_ms,
                due_mono_ms=due_mono_ms,
                eligible_mono_ms=eligible_mono_ms,
                late=late,
                late_by_ms=late_by_ms,
                jp_raw=cue.jp_raw,
                jp_fixed=jp_fixed,
                zh_out=zh_out,
                status=status,
                error="; ".join(error_parts),
            )
        )

    return results


def _write_live_header(fp, provider: str, cfg, effective_delay_ms: int, speed: float, cue_count: int) -> None:
    fp.write(f"provider={provider}\n")
    fp.write(f"base_url={cfg.llm.base_url}\n")
    fp.write(f"correct_model={cfg.llm.correct_model}\n")
    fp.write(f"translate_model={cfg.llm.translate_model}\n")
    fp.write(f"cue_count={cue_count} speed={speed} batch_size={cfg.llm.batch_size} context_window={cfg.llm.translate_context_window}\n")
    fp.write(
        "timing_model=due_mono_ms=seen_mono_ms+(delay_sec-asr_delay_sec)*1000; "
        f"effective_delay_ms={effective_delay_ms}\n"
    )
    fp.write("---\n")
    fp.flush()


def _append_live(fp, item: CueResult) -> None:
    fp.write(
        f"[{item.cue_index}] {item.source_key} status={item.status} "
        f"emit_jitter={item.emit_jitter_ms}ms wait={item.queue_wait_ms}ms s1={item.stage1_latency_ms}ms s2={item.stage2_latency_ms}ms "
        f"late={item.late} late_by={item.late_by_ms}ms\n"
    )
    fp.write(f"JP_RAW: {item.jp_raw}\n")
    fp.write(f"JP_FIX: {item.jp_fixed}\n")
    fp.write(f"ZH   : {item.zh_out}\n")
    if item.error:
        fp.write(f"ERROR: {item.error}\n")
    fp.write("\n")
    fp.flush()


def _write_summary(out_dir: Path, provider_results: Dict[str, List[CueResult]], skipped: Dict[str, str]) -> None:
    summary_path = out_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as fp:
        fp.write("# Replay Benchmark Summary\n\n")
        fp.write("## Timing model\n")
        fp.write("- Main chain uses `seen_mono_ms` when cue is first observed.\n")
        fp.write("- Due time model: `due_mono_ms = seen_mono_ms + (delay_sec - asr_delay_sec) * 1000`.\n")
        fp.write("- This differs from absolute SRT timestamp anchoring to wall clock.\n\n")

        if skipped:
            fp.write("## Skipped Providers\n")
            for provider, reason in skipped.items():
                fp.write(f"- `{provider}`: {reason}\n")
            fp.write("\n")

        for provider, rows in provider_results.items():
            fp.write(f"## {provider}\n")
            if not rows:
                fp.write("No rows.\n\n")
                continue

            ok_count = sum(1 for r in rows if r.status == "ok")
            s1_fail = sum(1 for r in rows if r.status == "stage1_failed")
            s2_fail = sum(1 for r in rows if r.status == "stage2_failed")
            pipeline_lat = [r.stage1_latency_ms + r.stage2_latency_ms for r in rows]
            late_count = sum(1 for r in rows if r.late)
            late_by = [r.late_by_ms for r in rows if r.late_by_ms > 0]
            emit_jitters = [abs(r.emit_jitter_ms) for r in rows]

            fp.write(f"- total: {len(rows)}\n")
            fp.write(f"- ok: {ok_count}\n")
            fp.write(f"- stage1_failed: {s1_fail}\n")
            fp.write(f"- stage2_failed: {s2_fail}\n")
            fp.write(f"- late: {late_count}\n")
            fp.write(f"- pipeline_avg_ms: {statistics.mean(pipeline_lat):.1f}\n")
            fp.write(f"- pipeline_p90_ms: {_percentile(pipeline_lat, 90):.1f}\n")
            fp.write(f"- pipeline_p95_ms: {_percentile(pipeline_lat, 95):.1f}\n")
            fp.write(f"- emit_jitter_p95_ms: {_percentile(emit_jitters, 95):.1f}\n")
            if late_by:
                fp.write(f"- late_avg_ms: {statistics.mean(late_by):.1f}\n")
                fp.write(f"- late_p95_ms: {_percentile(late_by, 95):.1f}\n")

            failed_idx = [r.cue_index for r in rows if r.status != "ok"]
            fp.write(f"- failed_indices: {failed_idx}\n\n")


def _percentile(values: Sequence[int], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _emit_cues_realtime(
    cues: List[ReplayCue],
    t0: float,
    s0_ms: int,
    speed: float,
    pending_queue: Deque[tuple[ReplayCue, int, int]],
    queue_lock: threading.Lock,
    done_event: threading.Event,
) -> None:
    try:
        for cue in cues:
            target_mono = t0 + ((cue.srt_start_ms - s0_ms) / 1000.0) / speed
            while True:
                now = time.monotonic()
                delta = target_mono - now
                if delta <= 0:
                    break
                time.sleep(min(0.01, max(0.001, delta)))
            seen_mono_ms = int(time.monotonic() * 1000)
            target_mono_ms = int(target_mono * 1000)
            emit_jitter_ms = seen_mono_ms - target_mono_ms
            with queue_lock:
                pending_queue.append((cue, seen_mono_ms, emit_jitter_ms))
    finally:
        done_event.set()


def _pop_batch(
    pending_queue: Deque[tuple[ReplayCue, int, int]],
    queue_lock: threading.Lock,
    batch_size: int,
) -> List[tuple[ReplayCue, int, int]]:
    out: List[tuple[ReplayCue, int, int]] = []
    with queue_lock:
        while pending_queue and len(out) < batch_size:
            out.append(pending_queue.popleft())
    return out


def _queue_size(
    pending_queue: Deque[tuple[ReplayCue, int, int]],
    queue_lock: threading.Lock,
) -> int:
    with queue_lock:
        return len(pending_queue)


if __name__ == "__main__":
    raise SystemExit(main())
