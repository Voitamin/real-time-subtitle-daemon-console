from __future__ import annotations

import argparse
import shutil
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from live_sub_daemon.console_server import ConsoleServer
from live_sub_daemon.models import CueRecord, SourceCue
from live_sub_daemon.runtime_control import RuntimeController
from live_sub_daemon.state_store import StateStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seeded Console V3 demo for screenshots.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--workspace", default=str(REPO_ROOT / "demo_runs" / "console_demo"))
    parser.add_argument("--preserve", action="store_true", help="Preserve existing workspace files if present.")
    return parser.parse_args()


def _build_cfg(workspace: Path, host: str, port: int) -> AppConfig:
    provider_settings = {
        "deepseek": ProviderRuntimeConfig(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key="dummy",
            api_key_env="DEEPSEEK_API_KEY",
            correct_model="deepseek-chat",
            translate_model="deepseek-chat",
            fast_translate_model="deepseek-chat",
        ),
        "qwen": ProviderRuntimeConfig(
            provider="qwen",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="dummy",
            api_key_env="DASHSCOPE_API_KEY",
            correct_model="qwen-plus",
            translate_model="qwen-plus",
            fast_translate_model="qwen-turbo",
        ),
        "xai": ProviderRuntimeConfig(
            provider="xai",
            base_url="https://api.x.ai/v1",
            api_key="dummy",
            api_key_env="XAI_API_KEY",
            correct_model="grok-3-mini",
            translate_model="grok-3-mini",
            fast_translate_model="grok-3-mini-fast",
        ),
    }
    return AppConfig(
        source=SourceConfig(input_srt=workspace / "demo.srt", input_txt=workspace / "demo.txt"),
        output=OutputConfig(zh_txt=workspace / "demo.zh.txt", zh_srt=workspace / "demo.zh.srt"),
        llm=LLMConfig(
            api_key="dummy",
            provider="deepseek",
            correct_provider="deepseek",
            translate_provider="qwen",
            translate_fallback_provider="xai",
            provider_settings=provider_settings,
        ),
        align=AlignConfig(delay_sec=16.0, asr_delay_sec=2.0),
        render=RenderConfig(char_threshold=18, max_total_chars=42, max_lines=2),
        pipeline=PipelineConfig(batch_max_wait_sec=8.0, batch_max_lines=4, context_wait_timeout_sec=4.0),
        adaptive=AdaptiveConfig(enabled=True, mode="auto"),
        console=ConsoleConfig(enabled=True, host=host, port=port),
        state=StateConfig(db_path=workspace / "demo_state.sqlite3"),
        fallback=FallbackConfig(mode="jp_raw"),
        knowledge=KnowledgeConfig(
            glossary_path=workspace / "demo_glossary.tsv",
            names_path=workspace / "demo_names.txt",
        ),
    )


def _reset_workspace(workspace: Path, preserve: bool) -> None:
    if workspace.exists() and not preserve:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)


def _write_terms(cfg: AppConfig) -> None:
    cfg.knowledge.glossary_path.write_text(
        "\n".join(
            [
                "ライブ\t直播",
                "予選\t预选",
                "決勝\t决赛",
                "王者\t王者",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg.knowledge.names_path.write_text(
        "\n".join(
            [
                "Voitamin",
                "KOP7th",
                "ロイヤルブレッド\tろいやるぶれっど,ろいやる",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _insert_cue(
    state: StateStore,
    runtime: RuntimeController,
    cfg: AppConfig,
    *,
    source_key: str,
    seen_offset_ms: int,
    jp_raw: str,
    srt_index: int,
    start_ms: int,
    end_ms: int,
) -> CueRecord:
    now_mono_ms = int(time.monotonic() * 1000)
    effective_delay_ms = int(max(0.0, cfg.align.delay_sec - cfg.align.asr_delay_sec) * 1000)
    cue = SourceCue(
        source_key=source_key,
        source_kind="speechmatics",
        srt_index=srt_index,
        start_ms=start_ms,
        end_ms=end_ms,
        jp_raw=jp_raw,
    )
    seen_mono_ms = now_mono_ms + int(seen_offset_ms)
    state.upsert_source_cues([cue], now_mono_ms=seen_mono_ms, delay_ms=effective_delay_ms)
    runtime.observe_arrival(1, now_mono_ms=max(seen_mono_ms, now_mono_ms - 15000))
    stored = state.fetch_cue_by_key(source_key)
    if stored is None:
        raise RuntimeError(f"failed to insert cue: {source_key}")
    return stored


def _save_result(
    state: StateStore,
    runtime: RuntimeController,
    cue: CueRecord,
    *,
    corrected: str,
    translated: str | None,
    translated_after_due_ms: int,
    stage1_provider: str,
    stage1_model: str,
    stage2_provider: str,
    stage2_model: str,
    stage1_ms: int,
    stage2_ms: int,
    fallback_used: bool = False,
    display_suppressed: bool = False,
    join_target_source_key: str | None = None,
) -> None:
    translated_at_mono_ms = int(cue.due_mono_ms + translated_after_due_ms)
    pipeline_ms = stage1_ms + stage2_ms
    state.save_pipeline_results(
        cues=[cue],
        corrected_texts={cue.source_key: corrected},
        translated_texts=({cue.source_key: translated} if translated is not None else {}),
        translated_at_mono_ms=translated_at_mono_ms,
        fallback_mode="jp_raw",
        llm_latency_ms=pipeline_ms,
        stage1_provider=stage1_provider,
        stage1_model=stage1_model,
        stage2_provider=stage2_provider,
        stage2_model=stage2_model,
        fallback_used=fallback_used,
        stage1_latency_ms=stage1_ms,
        stage2_latency_ms=stage2_ms,
        display_suppressed_map={cue.source_key: display_suppressed},
        join_target_map=({cue.source_key: join_target_source_key} if join_target_source_key else None),
    )
    runtime.observe_completion(
        total_count=1,
        late_count=1 if translated_after_due_ms > 0 else 0,
        stage1_ms=stage1_ms,
        stage2_ms=stage2_ms,
        pipeline_ms=pipeline_ms,
        now_mono_ms=int(time.monotonic() * 1000),
    )
    runtime.observe_translate_route(
        primary_ok=translated is not None,
        fallback_used=fallback_used,
        fallback_ok=fallback_used,
        used_fast_model=stage2_model == "qwen-turbo",
        now_mono_ms=int(time.monotonic() * 1000),
    )


def _seed_demo_data(state: StateStore, runtime: RuntimeController, cfg: AppConfig) -> None:
    cues: dict[str, CueRecord] = {}
    cues["cue-001"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:001",
        seen_offset_ms=-22000,
        jp_raw="今日はKOP第7回、本戦の終盤に入っています。",
        srt_index=1,
        start_ms=0,
        end_ms=2400,
    )
    cues["cue-002"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:002",
        seen_offset_ms=-16000,
        jp_raw="ロイヤルブレッド選手、かなり攻めたルートを選びましたね。",
        srt_index=2,
        start_ms=2600,
        end_ms=5200,
    )
    cues["cue-003"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:003",
        seen_offset_ms=-11000,
        jp_raw="ここは一度仕切り直して、次の波を待つ判断です。",
        srt_index=3,
        start_ms=5400,
        end_ms=7600,
    )
    cues["cue-004"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:004",
        seen_offset_ms=-5000,
        jp_raw="実況のテンポが上がったので、fast model に切り替えます。",
        srt_index=4,
        start_ms=7800,
        end_ms=10200,
    )
    cues["cue-005"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:005",
        seen_offset_ms=-2000,
        jp_raw="この一文は次の字幕に繋げて、二行表示を避けます。",
        srt_index=5,
        start_ms=10400,
        end_ms=12800,
    )
    cues["cue-006"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:006",
        seen_offset_ms=1000,
        jp_raw="次の cue はまだ処理待ちで、新着キューに残しています。",
        srt_index=6,
        start_ms=13000,
        end_ms=15400,
    )
    cues["cue-007"] = _insert_cue(
        state,
        runtime,
        cfg,
        source_key="speechmatics:demo:007",
        seen_offset_ms=-500,
        jp_raw="这一条故意保持在处理中，方便截图时看到 inflight 状态。",
        srt_index=7,
        start_ms=15600,
        end_ms=18000,
    )

    _save_result(
        state,
        runtime,
        cues["cue-001"],
        corrected="今日はKOP第7回、本戦の終盤に入っています。",
        translated="今天已经进入 KOP 第 7 回正赛的后半段。",
        translated_after_due_ms=-3200,
        stage1_provider="deepseek",
        stage1_model="deepseek-chat",
        stage2_provider="deepseek",
        stage2_model="deepseek-chat",
        stage1_ms=180,
        stage2_ms=940,
    )
    _save_result(
        state,
        runtime,
        cues["cue-002"],
        corrected="ロイヤルブレッド選手、かなり攻めたルートを選びましたね。",
        translated="Royal Bread 这一把选了很激进的路线。",
        translated_after_due_ms=-1800,
        stage1_provider="deepseek",
        stage1_model="deepseek-chat",
        stage2_provider="qwen",
        stage2_model="qwen-plus",
        stage1_ms=220,
        stage2_ms=1180,
    )
    _save_result(
        state,
        runtime,
        cues["cue-003"],
        corrected="ここは一度仕切り直して、次の波を待つ判断です。",
        translated="这里先稳一下，等下一波机会再进攻。",
        translated_after_due_ms=-1200,
        stage1_provider="xai",
        stage1_model="grok-3-mini",
        stage2_provider="xai",
        stage2_model="grok-3-mini",
        stage1_ms=260,
        stage2_ms=1320,
    )
    _save_result(
        state,
        runtime,
        cues["cue-004"],
        corrected="実況のテンポが上がったので、fast model に切り替えます。",
        translated="解说节奏上来了，这里切到 fast model 保证时效。",
        translated_after_due_ms=-600,
        stage1_provider="deepseek",
        stage1_model="deepseek-chat",
        stage2_provider="qwen",
        stage2_model="qwen-turbo",
        stage1_ms=210,
        stage2_ms=620,
        fallback_used=True,
    )
    _save_result(
        state,
        runtime,
        cues["cue-005"],
        corrected="この一文は次の字幕に繋げて、二行表示を避けます。",
        translated="这一句会并到下一条里，避免当前帧出现双行堆叠。",
        translated_after_due_ms=-400,
        stage1_provider="deepseek",
        stage1_model="deepseek-chat",
        stage2_provider="deepseek",
        stage2_model="deepseek-chat",
        stage1_ms=190,
        stage2_ms=880,
        display_suppressed=True,
        join_target_source_key=cues["cue-006"].source_key,
    )

    state.mark_displayed([cues["cue-001"].source_key], displayed_at_mono_ms=int(time.monotonic() * 1000) - 9000)
    latest_for_edit = state.fetch_cue_by_key(cues["cue-003"].source_key)
    if latest_for_edit is not None:
        state.upsert_manual_translation(
            source_key=latest_for_edit.source_key,
            text="这里先稳一手，等下一波机会再推进。",
            now_mono_ms=int(time.monotonic() * 1000),
            delay_adjust_ms=runtime.delay_adjust_ms(),
            expected_revision=latest_for_edit.revision,
        )

    latest_for_delete = state.fetch_cue_by_key(cues["cue-004"].source_key)
    if latest_for_delete is not None:
        state.soft_delete_cue(
            source_key=latest_for_delete.source_key,
            now_mono_ms=int(time.monotonic() * 1000),
            delay_adjust_ms=runtime.delay_adjust_ms(),
            expected_revision=latest_for_delete.revision,
        )

    claimed = state.fetch_and_claim_batch(limit=1, owner="demo-worker", now_mono_ms=int(time.monotonic() * 1000))
    if not claimed:
        raise RuntimeError("failed to create an inflight demo cue")

    runtime.observe_asr_source_emit(
        emitted_count=7,
        fragment_count=19,
        canonical_hit_count=5,
        now_mono_ms=int(time.monotonic() * 1000),
    )
    runtime.observe_s1_skipped(count=2, now_mono_ms=int(time.monotonic() * 1000))
    runtime.observe_s1_join(total_count=3, suppressed_count=1, now_mono_ms=int(time.monotonic() * 1000))
    for second_line_used in (False, True, False, True, False):
        runtime.observe_render_frame(second_line_used=second_line_used, now_mono_ms=int(time.monotonic() * 1000))
    runtime.set_asr_provider("speechmatics")
    runtime.set_asr_connected(True)
    runtime.observe_asr_final(int(time.monotonic() * 1000) - 1200)
    runtime.adjust_delay(1.5)


def _status_loop(state: StateStore, runtime: RuntimeController, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        now_mono_ms = int(time.monotonic() * 1000)
        queue_stats = state.fetch_new_queue_stats(
            now_mono_ms=now_mono_ms,
            delay_adjust_ms=runtime.delay_adjust_ms(),
        )
        runtime.update_and_get_status(queue_stats, now_mono_ms)
        time.sleep(0.5)


def main() -> int:
    args = _parse_args()
    workspace = Path(args.workspace).resolve()
    _reset_workspace(workspace, preserve=args.preserve)
    cfg = _build_cfg(workspace, args.host, args.port)
    _write_terms(cfg)

    state = StateStore(cfg.state.db_path)
    runtime = RuntimeController(cfg)
    stop_event = threading.Event()
    server: ConsoleServer | None = None
    updater = threading.Thread(
        target=_status_loop,
        args=(state, runtime, stop_event),
        name="console-demo-status",
        daemon=True,
    )

    try:
        _seed_demo_data(state, runtime, cfg)
        updater.start()
        server = ConsoleServer(
            host=cfg.console.host,
            port=cfg.console.port,
            state=state,
            runtime=runtime,
            glossary_path=cfg.knowledge.glossary_path,
            names_path=cfg.knowledge.names_path,
            char_threshold=cfg.render.char_threshold,
            max_total_chars=cfg.render.max_total_chars,
            max_lines=cfg.render.max_lines,
        )
        server.start()
        time.sleep(1.0)
        if server.start_error:
            raise RuntimeError(server.start_error)
        dist_index = REPO_ROOT / "web_console" / "dist" / "index.html"
        print(f"[demo] workspace={workspace}")
        print(f"[demo] url=http://{cfg.console.host}:{cfg.console.port}")
        if not dist_index.exists():
            print("[demo] frontend dist not found. Run `cd web_console && npm install && npm run build` for the full Console V3 UI.")
        print("[demo] seeded states: displayed, manual edit, deleted, fallback, inflight, new cue, glossary and names.")
        print("[demo] press Ctrl+C when you are done taking screenshots.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        stop_event.set()
        if server is not None:
            server.stop()
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
