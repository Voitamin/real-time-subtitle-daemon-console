from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from live_sub_daemon.config import load_config, parse_cli_args
from live_sub_daemon.console_server import ConsoleServer
from live_sub_daemon.knowledge import KnowledgeStore
from live_sub_daemon.models import CueRecord, SourceCue
from live_sub_daemon.pipeline import PipelineWorker
from live_sub_daemon.renderer import Renderer
from live_sub_daemon.runtime_control import RuntimeController
from live_sub_daemon.source_reader import SourceReader
from live_sub_daemon.stage_router import StageRouter
from live_sub_daemon.state_store import StateStore


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay jp.srt through the real daemon pipeline")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--srt", default="jp.srt")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--log-interval-sec", type=float, default=1.0)
    parser.add_argument("--console-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-host", default=None)
    parser.add_argument("--console-port", type=int, default=None)
    parser.add_argument("--hold-after-complete", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fail-on-missing-key", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--correct-provider", default=None)
    parser.add_argument("--translate-provider", default=None)
    parser.add_argument("--translate-fallback-provider", default=None)
    parser.add_argument("--translate-fallback-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-correct", default=None)
    parser.add_argument("--model-translate", default=None)
    return parser.parse_args(argv)


def _load_srt_cues(path: Path) -> List[SourceCue]:
    reader = SourceReader(srt_path=path, txt_path=Path("__unused__.txt"))
    return [cue for cue in reader.poll() if cue.source_kind == "srt" and cue.start_ms is not None and cue.end_ms is not None]


def _emit_cues_realtime(
    cues: List[SourceCue],
    speed: float,
    state: StateStore,
    runtime: RuntimeController,
    delay_ms: int,
    done_event: threading.Event,
) -> None:
    try:
        if not cues:
            return
        base_start_ms = int(cues[0].start_ms or 0)
        t0 = time.monotonic()
        for cue in cues:
            cue_start_ms = int(cue.start_ms or base_start_ms)
            target = t0 + ((cue_start_ms - base_start_ms) / 1000.0) / speed
            while True:
                now = time.monotonic()
                remain = target - now
                if remain <= 0:
                    break
                time.sleep(min(0.01, max(0.001, remain)))
            now_mono_ms = int(time.monotonic() * 1000)
            inserted = state.upsert_source_cues([cue], now_mono_ms=now_mono_ms, delay_ms=delay_ms)
            if inserted:
                runtime.observe_arrival(inserted, now_mono_ms=now_mono_ms)
    finally:
        done_event.set()


def _record_from_cue(cue: CueRecord, cue_index_map: Dict[str, int]) -> Dict[str, object]:
    eligible_mono_ms = cue.translated_mono_ms if cue.translated_mono_ms is not None else 0
    due_mono_ms = cue.due_mono_ms
    late = bool(eligible_mono_ms > due_mono_ms) if eligible_mono_ms else False
    late_by_ms = max(0, eligible_mono_ms - due_mono_ms) if eligible_mono_ms else 0
    logical_status = "ok" if cue.status == "TRANSLATED" else "stage2_failed"

    return {
        "cue_index": cue_index_map.get(cue.source_key),
        "source_key": cue.source_key,
        "srt_start_ms": cue.start_ms,
        "srt_end_ms": cue.end_ms,
        "seen_mono_ms": cue.t_seen_mono_ms,
        "due_mono_ms": cue.due_mono_ms,
        "eligible_mono_ms": cue.translated_mono_ms,
        "late": late,
        "late_by_ms": late_by_ms,
        "dropped_late": bool(cue.dropped_late),
        "jp_raw": cue.jp_raw,
        "jp_fixed": cue.jp_corrected if cue.jp_corrected is not None else cue.jp_raw,
        "zh_out": cue.zh_text if cue.zh_text is not None else "",
        "status": logical_status,
        "raw_status": cue.status,
        "error": cue.last_error or "",
        "stage1_provider": cue.stage1_provider,
        "stage1_model": cue.stage1_model,
        "stage2_provider": cue.stage2_provider,
        "stage2_model": cue.stage2_model,
        "fallback_used": bool(cue.fallback_used),
        "stage1_ms": cue.stage1_latency_ms,
        "stage2_ms": cue.stage2_latency_ms,
        "pipeline_ms": cue.llm_latency_ms,
        "context_miss": bool(cue.context_miss),
    }


def _append_live(fp, record: Dict[str, object]) -> None:
    fp.write(
        f"[{record.get('cue_index')}] {record.get('source_key')} status={record.get('status')} "
        f"s1={record.get('stage1_ms')}ms s2={record.get('stage2_ms')}ms p={record.get('pipeline_ms')}ms "
        f"fallback={record.get('fallback_used')} late={record.get('late')} late_by={record.get('late_by_ms')}ms\n"
    )
    fp.write(f"JP_RAW: {record.get('jp_raw')}\n")
    fp.write(f"JP_FIX: {record.get('jp_fixed')}\n")
    fp.write(f"ZH   : {record.get('zh_out')}\n")
    if record.get("error"):
        fp.write(f"ERROR: {record.get('error')}\n")
    fp.write("\n")
    fp.flush()


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


def _write_summary(path: Path, records: List[Dict[str, object]], config, route_status: Dict[str, object]) -> None:
    ok_rows = [r for r in records if r.get("status") == "ok"]
    fallback_rows = [r for r in records if r.get("fallback_used")]
    fallback_success_rows = [r for r in fallback_rows if r.get("status") == "ok"]
    late_rows = [r for r in records if r.get("late")]
    s1 = [int(r["stage1_ms"]) for r in records if isinstance(r.get("stage1_ms"), int)]
    s2 = [int(r["stage2_ms"]) for r in records if isinstance(r.get("stage2_ms"), int)]
    p = [int(r["pipeline_ms"]) for r in records if isinstance(r.get("pipeline_ms"), int)]

    with path.open("w", encoding="utf-8") as fp:
        fp.write("# Daemon Replay Summary\n\n")
        fp.write("## Route\n")
        fp.write(f"- stage1 provider: `{config.llm.correct_provider}`\n")
        fp.write(f"- stage2 provider: `{config.llm.translate_provider}`\n")
        fp.write(f"- stage2 fallback provider: `{config.llm.translate_fallback_provider}`\n")
        fp.write(f"- fallback_on_error: `{config.llm.translate_fallback_on_error}`\n")
        fp.write("\n")
        fp.write("## Timing model\n")
        fp.write("- due_mono_ms = seen_mono_ms + (delay_sec - asr_delay_sec) * 1000\n")
        fp.write("- Scheduler is t_seen-based (monotonic), not SRT absolute wall-clock anchoring.\n\n")
        fp.write("## Totals\n")
        fp.write(f"- total: {len(records)}\n")
        fp.write(f"- ok: {len(ok_rows)}\n")
        fp.write(f"- fallback_used: {len(fallback_rows)}\n")
        fp.write(f"- fallback_success: {len(fallback_success_rows)}\n")
        fp.write(f"- late: {len(late_rows)}\n")
        fp.write(f"- runtime_translate_primary_ok_recent: {route_status.get('translate_primary_ok_recent', 0)}\n")
        fp.write(f"- runtime_translate_fallback_count_recent: {route_status.get('translate_fallback_count_recent', 0)}\n")
        fp.write(
            f"- runtime_translate_fallback_success_rate_recent: "
            f"{float(route_status.get('translate_fallback_success_rate_recent', 0.0)):.3f}\n"
        )
        fp.write("\n")
        if p:
            fp.write("## Latency\n")
            fp.write(f"- pipeline_avg_ms: {statistics.mean(p):.1f}\n")
            fp.write(f"- pipeline_p50_ms: {_percentile(p, 50):.1f}\n")
            fp.write(f"- pipeline_p95_ms: {_percentile(p, 95):.1f}\n")
            fp.write(f"- pipeline_p99_ms: {_percentile(p, 99):.1f}\n")
            fp.write(f"- stage1_p95_ms: {_percentile(s1, 95):.1f}\n")
            fp.write(f"- stage2_p95_ms: {_percentile(s2, 95):.1f}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.speed <= 0:
        raise ValueError("--speed must be > 0")

    load_cli: List[str] = ["--config", str(args.config)]
    if args.provider:
        load_cli.extend(["--provider", str(args.provider)])
    if args.correct_provider:
        load_cli.extend(["--correct-provider", str(args.correct_provider)])
    if args.translate_provider:
        load_cli.extend(["--translate-provider", str(args.translate_provider)])
    if args.translate_fallback_provider:
        load_cli.extend(["--translate-fallback-provider", str(args.translate_fallback_provider)])
    if args.translate_fallback_on_error is not None:
        load_cli.append("--translate-fallback-on-error" if args.translate_fallback_on_error else "--no-translate-fallback-on-error")
    if args.model_correct:
        load_cli.extend(["--model-correct", str(args.model_correct)])
    if args.model_translate:
        load_cli.extend(["--model-translate", str(args.model_translate)])

    try:
        cfg = load_config(parse_cli_args(load_cli))
    except Exception:
        if args.fail_on_missing_key:
            raise
        raise

    srt_path = Path(args.srt)
    cues = _load_srt_cues(srt_path)
    if not cues:
        raise ValueError(f"No SRT cues found in {srt_path}")

    if args.duration_sec is not None and args.duration_sec > 0:
        base = int(cues[0].start_ms or 0)
        cutoff = int(args.duration_sec * 1000)
        cues = [cue for cue in cues if int(cue.start_ms or base) - base <= cutoff]
        if not cues:
            raise ValueError("No cues left after --duration-sec filter")

    out_dir = Path(args.out_dir) if args.out_dir else Path("benchmark_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    live_path = out_dir / "runner.live.txt"
    full_path = out_dir / "runner.full.jsonl"
    summary_path = out_dir / "summary.md"

    cfg.state.db_path = out_dir / "runner_state.sqlite3"
    cfg.output.zh_txt = out_dir / "runner.zh.txt"
    cfg.output.zh_srt = out_dir / "runner.zh.srt"
    cfg.source.input_srt = srt_path
    if args.console_enabled is not None:
        cfg.console.enabled = bool(args.console_enabled)
    if args.console_host is not None:
        cfg.console.host = str(args.console_host)
    if args.console_port is not None:
        cfg.console.port = int(args.console_port)
    hold_after_complete = bool(args.hold_after_complete) if args.hold_after_complete is not None else bool(cfg.console.enabled)

    state = StateStore(cfg.state.db_path)
    knowledge = KnowledgeStore(
        glossary_path=cfg.knowledge.glossary_path,
        names_path=cfg.knowledge.names_path,
        reload_interval_sec=cfg.knowledge.reload_interval_sec,
    )
    renderer = Renderer(
        state=state,
        zh_txt_path=cfg.output.zh_txt,
        zh_srt_path=cfg.output.zh_srt,
        char_threshold=cfg.render.char_threshold,
    )
    runtime = RuntimeController(cfg)
    router = StageRouter(cfg)
    stop_event = threading.Event()
    console_server: Optional[ConsoleServer] = None

    workers: List[PipelineWorker] = []
    workers.append(
        PipelineWorker(
            config=cfg,
            state=state,
            knowledge=knowledge,
            router=router,
            runtime=runtime,
            stop_event=stop_event,
            worker_id="main",
            rescue=False,
        )
    )
    for idx in range(max(0, int(cfg.adaptive.red_parallel_contextless))):
        workers.append(
            PipelineWorker(
                config=cfg,
                state=state,
                knowledge=knowledge,
                router=router,
                runtime=runtime,
                stop_event=stop_event,
                worker_id=f"rescue-{idx + 1}",
                rescue=True,
                rescue_index=idx,
            )
        )
    for worker in workers:
        worker.start()

    if cfg.console.enabled:
        console_server = ConsoleServer(
            host=cfg.console.host,
            port=cfg.console.port,
            state=state,
            runtime=runtime,
            glossary_path=cfg.knowledge.glossary_path,
            names_path=cfg.knowledge.names_path,
            delay_sec=cfg.align.delay_sec,
            asr_delay_sec=cfg.align.asr_delay_sec,
            char_threshold=cfg.render.char_threshold,
        )
        console_server.start()

    effective_delay_sec = max(0.0, float(cfg.align.delay_sec) - float(cfg.align.asr_delay_sec))
    delay_ms = int(effective_delay_sec * 1000)
    emitter_done = threading.Event()
    emitter = threading.Thread(
        target=_emit_cues_realtime,
        args=(cues, args.speed, state, runtime, delay_ms, emitter_done),
        daemon=True,
        name="replay-emitter",
    )
    emitter.start()

    cue_index_map = {cue.source_key: idx + 1 for idx, cue in enumerate(cues)}
    emitted_keys = set()
    records: List[Dict[str, object]] = []

    next_render = 0.0
    next_runtime = 0.0
    next_log = 0.0
    tick_interval = 1.0 / max(0.1, float(cfg.render.tick_hz))
    last_status: Dict[str, object] = {}
    last_queue_stats: Dict[str, object] = {}

    with live_path.open("w", encoding="utf-8") as live_fp, full_path.open("w", encoding="utf-8") as full_fp:
        live_fp.write(f"srt={srt_path}\n")
        live_fp.write(f"speed={args.speed}\n")
        live_fp.write(f"delay_sec={cfg.align.delay_sec} asr_delay_sec={cfg.align.asr_delay_sec} effective_delay_sec={effective_delay_sec}\n")
        live_fp.write(
            f"route={cfg.llm.correct_provider}->{cfg.llm.translate_provider} "
            f"fallback={cfg.llm.translate_fallback_provider} enabled={cfg.llm.translate_fallback_on_error}\n"
        )
        if cfg.console.enabled:
            live_fp.write(f"dashboard=http://{cfg.console.host}:{cfg.console.port}\n")
        live_fp.write(f"hold_after_complete={hold_after_complete}\n")
        live_fp.write("---\n")
        live_fp.flush()

        try:
            replay_completed = False
            while True:
                now = time.monotonic()
                now_mono_ms = int(now * 1000)

                if now >= next_render:
                    next_render = now + tick_interval
                    try:
                        renderer.tick(now_mono_ms=now_mono_ms, delay_adjust_ms=runtime.delay_adjust_ms())
                    except Exception as exc:  # noqa: BLE001
                        live_fp.write(f"[render_error] {exc}\n")
                        live_fp.flush()

                if now >= next_runtime:
                    next_runtime = now + 1.0
                    last_queue_stats = state.fetch_new_queue_stats(
                        now_mono_ms=now_mono_ms,
                        delay_adjust_ms=runtime.delay_adjust_ms(),
                    )
                    last_queue_stats["wal_size_bytes"] = state.get_wal_size_bytes()
                    last_status = runtime.update_and_get_status(queue_stats=last_queue_stats, now_mono_ms=now_mono_ms)

                for cue in state.fetch_translated_cues():
                    if cue.source_key in emitted_keys:
                        continue
                    emitted_keys.add(cue.source_key)
                    row = _record_from_cue(cue, cue_index_map)
                    records.append(row)
                    full_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                    full_fp.flush()
                    _append_live(live_fp, row)

                if now >= next_log:
                    next_log = now + max(0.2, float(args.log_interval_sec))
                    live_fp.write(
                        "[status] "
                        f"mode={last_status.get('mode')} "
                        f"backlog={last_queue_stats.get('unfinished_count')} "
                        f"slack={last_status.get('slack_ms')} "
                        f"arrival={float(last_status.get('arrival_rate_lps', 0.0)):.2f}/s "
                        f"service={float(last_status.get('service_rate_lps', 0.0)):.2f}/s\n"
                    )
                    live_fp.flush()

                unfinished_count = int(last_queue_stats.get("unfinished_count") or 0)
                if emitter_done.is_set() and len(emitted_keys) >= len(cues) and unfinished_count == 0:
                    replay_completed = True
                    if not hold_after_complete:
                        break
                    live_fp.write("[runner] replay completed; holding dashboard, press Ctrl+C to stop\n")
                    live_fp.flush()
                    break

                time.sleep(0.03)
            while replay_completed and hold_after_complete:
                now = time.monotonic()
                now_mono_ms = int(now * 1000)
                if now >= next_render:
                    next_render = now + tick_interval
                    try:
                        renderer.tick(now_mono_ms=now_mono_ms, delay_adjust_ms=runtime.delay_adjust_ms())
                    except Exception as exc:  # noqa: BLE001
                        live_fp.write(f"[render_error] {exc}\n")
                        live_fp.flush()
                if now >= next_runtime:
                    next_runtime = now + 1.0
                    last_queue_stats = state.fetch_new_queue_stats(
                        now_mono_ms=now_mono_ms,
                        delay_adjust_ms=runtime.delay_adjust_ms(),
                    )
                    last_queue_stats["wal_size_bytes"] = state.get_wal_size_bytes()
                    last_status = runtime.update_and_get_status(queue_stats=last_queue_stats, now_mono_ms=now_mono_ms)
                if now >= next_log:
                    next_log = now + max(0.2, float(args.log_interval_sec))
                    live_fp.write(
                        "[status-hold] "
                        f"mode={last_status.get('mode')} "
                        f"backlog={last_queue_stats.get('unfinished_count')} "
                        f"slack={last_status.get('slack_ms')}\n"
                    )
                    live_fp.flush()
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            emitter.join(timeout=2.0)
            for worker in workers:
                worker.join(timeout=5.0)
            if console_server:
                console_server.stop()
                console_server.join(timeout=3.0)
            state.close()

    _write_summary(summary_path, records, cfg, last_status)
    print(f"[runner] out_dir={out_dir}")
    print(f"[runner] live={live_path}")
    print(f"[runner] full={full_path}")
    print(f"[runner] summary={summary_path}")
    if cfg.console.enabled:
        print(f"[runner] dashboard=http://{cfg.console.host}:{cfg.console.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
