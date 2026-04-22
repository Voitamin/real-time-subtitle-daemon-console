from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .config import load_config, parse_cli_args
from .console_server import ConsoleServer
from .knowledge import KnowledgeStore
from .pipeline import PipelineWorker
from .renderer import Renderer
from .runtime_control import RuntimeController
from .runtime_scope import build_query_scope, normalize_scope_kinds_for_json
from .stage_router import StageRouter
from .source_base import BaseSourceReader
from .source_reader import FileSourceReader
from .state_store import StateStore


def run_daemon(argv: Optional[list[str]] = None) -> int:
    args = parse_cli_args(argv)
    config = load_config(args)

    state = StateStore(config.state.db_path)
    knowledge = KnowledgeStore(
        glossary_path=config.knowledge.glossary_path,
        names_path=config.knowledge.names_path,
        reload_interval_sec=config.knowledge.reload_interval_sec,
    )
    router = StageRouter(config)
    runtime = RuntimeController(config)
    scope_seed = build_query_scope(config, effective_delay_sec=runtime.effective_delay_sec(), now_unix=time.time())
    scope_allowed_source_kinds = scope_seed.allowed_source_kinds
    scope_freshness_sec = scope_seed.freshness_sec if config.runtime_scope.enabled else 0.0
    renderer = Renderer(
        state=state,
        zh_txt_path=config.output.zh_txt,
        zh_srt_path=config.output.zh_srt,
        char_threshold=config.render.char_threshold,
        max_total_chars=config.render.max_total_chars,
        max_lines=config.render.max_lines,
        two_line_roll_enabled=config.render.two_line_roll_enabled,
        min_hold_sec=config.render.min_hold_sec,
        target_cps=config.render.target_cps,
        max_hold_sec=config.render.max_hold_sec,
        backlog_relax_threshold=config.render.backlog_relax_threshold,
        backlog_relaxed_min_hold_sec=config.render.backlog_relaxed_min_hold_sec,
        allowed_source_kinds=scope_allowed_source_kinds,
        scope_freshness_sec=scope_freshness_sec,
    )
    source_reader: BaseSourceReader = _build_source_reader(config=config, runtime=runtime)

    if config.runtime_scope.enabled and config.runtime_scope.auto_archive_stale_on_start:
        cleanup = state.cleanup_runtime_scope(
            now_mono_ms=int(time.monotonic() * 1000),
            allowed_source_kinds=scope_allowed_source_kinds,
            stale_unfinished_sec=scope_freshness_sec,
            monotonic_guard_sec=config.runtime_scope.monotonic_guard_sec,
            anomaly_remaining_max_sec=max(300.0, runtime.effective_delay_sec() * 3.0),
        )
        archived_count = int(cleanup.get("archived_count", 0))
        if archived_count > 0:
            print(
                "[runtime_scope] startup archive "
                f"archived={archived_count} excluded_source={cleanup.get('excluded_source_count', 0)} "
                f"stale_unfinished={cleanup.get('stale_unfinished_count', 0)} "
                f"anomaly={cleanup.get('anomaly_archived_count', 0)}"
            )

    stop_event = threading.Event()
    workers: list[PipelineWorker] = []
    main_worker = PipelineWorker(
        config=config,
        state=state,
        knowledge=knowledge,
        router=router,
        runtime=runtime,
        stop_event=stop_event,
        worker_id="main",
        rescue=False,
        allowed_source_kinds=scope_allowed_source_kinds,
        scope_freshness_sec=scope_freshness_sec,
    )
    workers.append(main_worker)

    rescue_workers = max(0, int(config.adaptive.red_parallel_contextless))
    for idx in range(rescue_workers):
        workers.append(
            PipelineWorker(
                config=config,
                state=state,
                knowledge=knowledge,
                router=router,
                runtime=runtime,
                stop_event=stop_event,
                worker_id=f"rescue-{idx + 1}",
                rescue=True,
                rescue_index=idx,
                allowed_source_kinds=scope_allowed_source_kinds,
                scope_freshness_sec=scope_freshness_sec,
            )
        )

    for worker in workers:
        worker.start()

    console_server: Optional[ConsoleServer] = None
    if config.console.enabled:
        console_server = ConsoleServer(
            host=config.console.host,
            port=config.console.port,
            state=state,
            runtime=runtime,
            glossary_path=config.knowledge.glossary_path,
            names_path=config.knowledge.names_path,
            delay_sec=config.align.delay_sec,
            asr_delay_sec=config.align.asr_delay_sec,
            char_threshold=config.render.char_threshold,
            max_total_chars=config.render.max_total_chars,
            max_lines=config.render.max_lines,
            allowed_source_kinds=scope_allowed_source_kinds,
            scope_freshness_sec=scope_freshness_sec,
            runtime_scope_monotonic_guard_sec=config.runtime_scope.monotonic_guard_sec,
        )
        console_server.start()

    print(
        "[live-sub] started "
        f"provider={config.llm.provider} route={config.llm.correct_provider}->{config.llm.translate_provider}"
        f"(fallback:{config.llm.translate_fallback_provider}) "
        f"source_mode={config.source.mode} input={config.source.input_srt} delay={config.align.delay_sec}s "
        f"s1_mode={config.pipeline.s1_mode} "
        f"asr_delay={config.align.asr_delay_sec}s workers={len(workers)} "
        f"microbatch(wait={config.pipeline.batch_max_wait_sec}s, lines={config.pipeline.batch_max_lines})"
    )
    if console_server and console_server.start_error:
        print(f"[console] {console_server.start_error}", file=sys.stderr)
    elif console_server:
        print(f"[console] http://{config.console.host}:{config.console.port}")

    tick_interval = 1.0 / max(config.render.tick_hz, 0.1)
    next_poll = 0.0
    next_render = 0.0
    next_runtime = 0.0
    next_metrics_log = 0.0
    next_checkpoint = 0.0
    next_trace = 0.0
    trace_fp = None

    effective_delay_sec = max(0.0, config.align.delay_sec - config.align.asr_delay_sec)
    delay_ms = int(effective_delay_sec * 1000)
    print(f"[align] effective_delay={effective_delay_sec:.3f}s (D={config.align.delay_sec}s - asr={config.align.asr_delay_sec}s)")
    if config.metrics.trace_enabled:
        try:
            trace_path = Path(config.metrics.trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_fp = trace_path.open("a", encoding="utf-8")
            print(f"[metrics] trace enabled -> {trace_path} interval={config.metrics.trace_interval_sec:.2f}s")
        except Exception as exc:  # noqa: BLE001
            trace_fp = None
            print(f"[metrics] trace disabled (open failed): {exc}", file=sys.stderr)

    try:
        while True:
            now = time.monotonic()
            now_mono_ms = int(now * 1000)
            scope_now = build_query_scope(config, effective_delay_sec=runtime.effective_delay_sec(), now_unix=time.time())
            scope_allowed = scope_now.allowed_source_kinds
            scope_updated_after = scope_now.updated_after_unix

            if now >= next_poll:
                next_poll = now + max(config.source.poll_interval_sec, 0.1)
                try:
                    cues = source_reader.poll()
                    inserted = state.upsert_source_cues(cues=cues, now_mono_ms=now_mono_ms, delay_ms=delay_ms)
                    if inserted:
                        runtime.observe_arrival(inserted, now_mono_ms=now_mono_ms)
                except Exception as exc:  # noqa: BLE001
                    print(f"[source] poll error: {exc}", file=sys.stderr)

            if now >= next_render:
                next_render = now + tick_interval
                try:
                    decision = renderer.tick(now_mono_ms=now_mono_ms, delay_adjust_ms=runtime.delay_adjust_ms())
                    runtime.observe_render_frame(
                        second_line_used=("\n" in decision.text),
                        now_mono_ms=now_mono_ms,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[render] tick error: {exc}", file=sys.stderr)

            if now >= next_runtime:
                next_runtime = now + 1.0
                queue_stats = state.fetch_new_queue_stats(
                    now_mono_ms=now_mono_ms,
                    delay_adjust_ms=runtime.delay_adjust_ms(),
                    allowed_source_kinds=scope_allowed,
                    updated_after_unix=scope_updated_after,
                )
                queue_stats["wal_size_bytes"] = state.get_wal_size_bytes()
                queue_stats["scope_allowed_source_kinds"] = normalize_scope_kinds_for_json(scope_allowed)
                queue_stats["scope_updated_after_unix"] = scope_updated_after
                latest_ready = state.fetch_latest_ready_unshown(
                    delay_adjust_ms=runtime.delay_adjust_ms(),
                    allowed_source_kinds=scope_allowed,
                    updated_after_unix=scope_updated_after,
                )
                if latest_ready is not None:
                    due_effective = int(latest_ready["due_effective_mono_ms"])
                    remaining_ms = due_effective - now_mono_ms
                    max_reasonable_ms = int(max(300000.0, runtime.effective_delay_ms() * 3.0))
                    latest_ready_anomaly = remaining_ms > max_reasonable_ms
                    queue_stats["latest_ready_source_key"] = latest_ready["source_key"]
                    queue_stats["latest_ready_due_effective_mono_ms"] = due_effective
                    queue_stats["latest_ready_remaining_ms"] = None if latest_ready_anomaly else remaining_ms
                    queue_stats["latest_ready_remaining_raw_ms"] = remaining_ms
                    queue_stats["latest_ready_anomaly"] = bool(latest_ready_anomaly)
                else:
                    queue_stats["latest_ready_source_key"] = None
                    queue_stats["latest_ready_due_effective_mono_ms"] = None
                    queue_stats["latest_ready_remaining_ms"] = None
                    queue_stats["latest_ready_remaining_raw_ms"] = None
                    queue_stats["latest_ready_anomaly"] = False
                status = runtime.update_and_get_status(queue_stats=queue_stats, now_mono_ms=now_mono_ms)
                if now >= next_metrics_log:
                    next_metrics_log = now + 10.0
                    stage = status.get("stage", {})
                    latest_ready_ms = status.get("latest_ready_remaining_ms")
                    latest_ready_text = (
                        "-"
                        if latest_ready_ms is None
                        else f"{int(latest_ready_ms)}ms"
                    )
                    if status.get("latest_ready_anomaly"):
                        latest_ready_text = f"anomaly(raw={queue_stats.get('latest_ready_remaining_raw_ms')})"
                    print(
                        "[runtime] "
                        f"mode={status.get('mode')} alert={status.get('alert_level')} slack={status.get('slack_ms')}ms "
                        f"queue={queue_stats.get('unfinished_count')} arrival={status.get('arrival_rate_lps', 0):.2f}/s "
                        f"service={status.get('service_rate_lps', 0):.2f}/s "
                        f"p95_pipeline={stage.get('pipeline_p95_ms', 0):.0f}ms "
                        f"latest_ready_remaining={latest_ready_text} "
                        f"fast_ratio={status.get('translate_fast_ratio_recent', 0.0):.2f} "
                        f"fallback_ratio={status.get('translate_fallback_ratio_recent', 0.0):.2f}"
                    )
                if trace_fp is not None and now >= next_trace:
                    next_trace = now + max(0.2, float(config.metrics.trace_interval_sec))
                    trace_row = {
                        "wall_time_unix": time.time(),
                        "mono_ms": now_mono_ms,
                        "mode": status.get("mode"),
                        "slack_ms": status.get("slack_ms"),
                        "latest_ready_remaining_ms": status.get("latest_ready_remaining_ms"),
                        "latest_ready_anomaly": status.get("latest_ready_anomaly"),
                        "latest_ready_source_key": status.get("latest_ready_source_key"),
                        "latest_ready_due_effective_mono_ms": status.get("latest_ready_due_effective_mono_ms"),
                        "overdue_unfinished_count": queue_stats.get("overdue_unfinished_count"),
                        "unfinished_count": queue_stats.get("unfinished_count"),
                        "new_count": queue_stats.get("new_count"),
                        "inflight_count": queue_stats.get("inflight_count"),
                        "arrival_rate_lps": status.get("arrival_rate_lps"),
                        "service_rate_lps": status.get("service_rate_lps"),
                        "late_rate_recent": status.get("late_rate_recent"),
                        "alert_level": status.get("alert_level"),
                        "translate_fast_count_recent": status.get("translate_fast_count_recent"),
                        "translate_fast_ratio_recent": status.get("translate_fast_ratio_recent"),
                        "translate_fallback_count_recent": status.get("translate_fallback_count_recent"),
                        "translate_fallback_ratio_recent": status.get("translate_fallback_ratio_recent"),
                        "predicted_total_ms": status.get("predicted_total_ms"),
                        "predicted_queue_wait_ms": status.get("predicted_queue_wait_ms"),
                        "predicted_service_ms": status.get("predicted_service_ms"),
                        "stage": status.get("stage"),
                    }
                    trace_fp.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                    trace_fp.flush()

            if now >= next_checkpoint:
                next_checkpoint = now + 15.0
                ck = state.run_wal_checkpoint("PASSIVE")
                if ck.get("busy", 0):
                    print(f"[wal] checkpoint busy={ck.get('busy')} log={ck.get('log')} ckpt={ck.get('checkpointed')}")

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("[live-sub] stopping")
    finally:
        stop_event.set()
        if console_server:
            console_server.stop()
        for worker in workers:
            worker.join(timeout=5.0)
        if console_server:
            console_server.join(timeout=3.0)
        source_reader.close()
        if trace_fp is not None:
            trace_fp.close()
        state.close()
        _write_back_delay_to_config(
            path=Path("config.toml"),
            final_delay_sec=max(0.0, runtime.base_delay_sec() + runtime.delay_adjust_sec()),
        )

    return 0


def _write_back_delay_to_config(path: Path, final_delay_sec: float) -> None:
    try:
        if not path.exists():
            print(f"[align] skip writeback: config not found at {path}")
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: list[str] = []
        in_align = False
        wrote = False
        align_seen = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if in_align and not wrote:
                    out.append(f"delay_sec = {final_delay_sec:.3f}")
                    wrote = True
                section = stripped[1:-1].strip().lower()
                in_align = section == "align"
                align_seen = align_seen or in_align
                out.append(line)
                continue

            if in_align and stripped.startswith("delay_sec") and "=" in stripped and not wrote:
                prefix = line.split("=", 1)[0]
                out.append(f"{prefix}= {final_delay_sec:.3f}")
                wrote = True
                continue
            out.append(line)

        if align_seen and not wrote:
            out.append(f"delay_sec = {final_delay_sec:.3f}")
            wrote = True

        if not align_seen:
            if out and out[-1].strip():
                out.append("")
            out.append("[align]")
            out.append(f"delay_sec = {final_delay_sec:.3f}")
            wrote = True

        if wrote:
            new_text = "\n".join(out).rstrip() + "\n"
            path.write_text(new_text, encoding="utf-8")
            print(f"[align] wrote back align.delay_sec={final_delay_sec:.3f} to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[align] writeback failed: {exc}", file=sys.stderr)


def _build_source_reader(config, runtime: RuntimeController) -> BaseSourceReader:
    source_mode = str(config.source.mode).strip().lower()
    if source_mode == "speechmatics":
        runtime.set_asr_provider("speechmatics")
        runtime.set_asr_connected(False)
        from .asr_speechmatics import SpeechmaticsSourceReader

        return SpeechmaticsSourceReader(
            config=config.source.speechmatics,
            glossary_path=config.knowledge.glossary_path,
            names_path=config.knowledge.names_path,
            names_phonetic_canonicalize_enabled=config.knowledge.names_phonetic_canonicalize_enabled,
            names_phonetic_max_rules=config.knowledge.names_phonetic_max_rules,
            runtime=runtime,
        )

    runtime.set_asr_provider("file")
    runtime.set_asr_connected(True)
    return FileSourceReader(config.source.input_srt, config.source.input_txt)
