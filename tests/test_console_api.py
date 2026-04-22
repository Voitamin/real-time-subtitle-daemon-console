from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

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
from live_sub_daemon.console_server import build_console_app
from live_sub_daemon.models import SourceCue
from live_sub_daemon.runtime_control import RuntimeController
from live_sub_daemon.state_store import StateStore


def _build_cfg(root: Path) -> AppConfig:
    return AppConfig(
        source=SourceConfig(input_srt=root / "jp.srt", input_txt=root / "jp.txt"),
        output=OutputConfig(zh_txt=root / "zh.txt", zh_srt=root / "zh.srt"),
        llm=LLMConfig(api_key="dummy"),
        align=AlignConfig(delay_sec=180, asr_delay_sec=0),
        render=RenderConfig(),
        pipeline=PipelineConfig(),
        adaptive=AdaptiveConfig(),
        console=ConsoleConfig(enabled=True, host="127.0.0.1", port=8787),
        state=StateConfig(db_path=root / "state.sqlite3"),
        fallback=FallbackConfig(),
        knowledge=KnowledgeConfig(glossary_path=root / "glossary.tsv", names_path=root / "names.txt"),
    )


class ConsoleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        (self.root / "glossary.tsv").write_text("", encoding="utf-8")
        (self.root / "names.txt").write_text("", encoding="utf-8")

        cfg = _build_cfg(self.root)
        self.state = StateStore(cfg.state.db_path)
        self.runtime = RuntimeController(cfg)
        self.stop_event = threading.Event()
        app = build_console_app(
            state=self.state,
            runtime=self.runtime,
            stop_event=self.stop_event,
            glossary_path=cfg.knowledge.glossary_path,
            names_path=cfg.knowledge.names_path,
            char_threshold=18,
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.stop_event.set()
        self.state.close()
        self._temp.cleanup()

    def test_delay_adjust_accepts_json_body(self) -> None:
        resp = self.client.post("/api/control/delay_adjust", json={"delta_sec": 1})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("delay_adjust_sec", data)
        self.assertIn("effective_delay_sec", data)

    def test_delete_not_found_returns_404_not_500(self) -> None:
        resp = self.client.post("/api/cues/not-exists/delete", json={"revision": 1})
        self.assertEqual(resp.status_code, 404)
        self.assertIn("cue not found", str(resp.json().get("detail", "")))

    def test_delete_and_restore_allow_empty_body(self) -> None:
        now_mono_ms = int(time.monotonic() * 1000)
        cue = SourceCue(
            source_key="srt:1:0:1000",
            source_kind="srt",
            srt_index=1,
            start_ms=0,
            end_ms=1000,
            jp_raw="原文",
        )
        self.state.upsert_source_cues([cue], now_mono_ms=now_mono_ms, delay_ms=120000)
        batch = self.state.fetch_new_batch(5)
        self.state.save_pipeline_results(
            cues=batch,
            corrected_texts={cue.source_key: cue.jp_raw},
            translated_texts={cue.source_key: "翻译"},
            translated_at_mono_ms=now_mono_ms + 1000,
            fallback_mode="jp_raw",
            llm_latency_ms=120,
        )

        deleted = self.client.post(f"/api/cues/{cue.source_key}/delete")
        self.assertEqual(deleted.status_code, 200)

        restored = self.client.post(f"/api/cues/{cue.source_key}/restore")
        self.assertEqual(restored.status_code, 200)

    def test_jump_to_latest_skips_due_unshown(self) -> None:
        now_mono_ms = int(time.monotonic() * 1000)
        cue = SourceCue(
            source_key="srt:9:0:1000",
            source_kind="srt",
            srt_index=9,
            start_ms=0,
            end_ms=1000,
            jp_raw="原文",
        )
        self.state.upsert_source_cues([cue], now_mono_ms=now_mono_ms - 5000, delay_ms=1000)
        batch = self.state.fetch_new_batch(5)
        self.state.save_pipeline_results(
            cues=batch,
            corrected_texts={cue.source_key: cue.jp_raw},
            translated_texts={cue.source_key: "翻译"},
            translated_at_mono_ms=now_mono_ms - 6000,
            fallback_mode="jp_raw",
            llm_latency_ms=120,
        )
        resp = self.client.post("/api/control/jump_to_latest")
        self.assertEqual(resp.status_code, 200)
        self.assertGreaterEqual(int(resp.json().get("skipped_count", 0)), 1)

    def test_runtime_and_mode_accept_json_body(self) -> None:
        runtime_resp = self.client.post("/api/control/runtime", json={"batch_max_wait_sec": 12})
        self.assertEqual(runtime_resp.status_code, 200)
        self.assertTrue(runtime_resp.json().get("ok"))

        mode_resp = self.client.post("/api/control/mode", json={"mode": "yellow"})
        self.assertEqual(mode_resp.status_code, 200)
        self.assertEqual(mode_resp.json().get("mode"), "yellow")

    def test_cleanup_stale_endpoint_returns_counts(self) -> None:
        resp = self.client.post("/api/control/cleanup_stale")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body.get("ok"))
        self.assertIn("archived_count", body)

    def test_flush_pending_clears_new_and_inflight_queue(self) -> None:
        now_mono_ms = int(time.monotonic() * 1000)
        cue = SourceCue(
            source_key="speechmatics:flush:1",
            source_kind="speechmatics",
            srt_index=None,
            start_ms=None,
            end_ms=None,
            jp_raw="flush_me",
        )
        self.state.upsert_source_cues([cue], now_mono_ms=now_mono_ms, delay_ms=120000)
        claimed = self.state.fetch_and_claim_batch(
            limit=1,
            owner="test",
            now_mono_ms=now_mono_ms,
        )
        self.assertEqual(len(claimed), 1)

        resp = self.client.post("/api/control/flush_pending")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body.get("ok"))
        self.assertGreaterEqual(int(body.get("flushed_count", 0)), 1)

        stats = self.state.fetch_new_queue_stats(now_mono_ms=now_mono_ms + 1000, delay_adjust_ms=0)
        self.assertEqual(int(stats.get("new_count") or 0), 0)
        self.assertEqual(int(stats.get("inflight_count") or 0), 0)

    def test_window_scope_filters_out_non_active_source_kind(self) -> None:
        app = build_console_app(
            state=self.state,
            runtime=self.runtime,
            stop_event=self.stop_event,
            glossary_path=self.root / "glossary.tsv",
            names_path=self.root / "names.txt",
            char_threshold=18,
            allowed_source_kinds=["speechmatics"],
            scope_freshness_sec=3600.0,
        )
        scoped_client = TestClient(app)

        now_mono_ms = int(time.monotonic() * 1000)
        srt = SourceCue(
            source_key="srt:1:0:1000",
            source_kind="srt",
            srt_index=1,
            start_ms=0,
            end_ms=1000,
            jp_raw="srt_row",
        )
        speech = SourceCue(
            source_key="speechmatics:1:1000",
            source_kind="speechmatics",
            srt_index=None,
            start_ms=None,
            end_ms=None,
            jp_raw="speech_row",
        )
        self.state.upsert_source_cues([srt, speech], now_mono_ms=now_mono_ms, delay_ms=1000)

        resp = scoped_client.get("/api/cues/window?window_sec=300&limit=200")
        self.assertEqual(resp.status_code, 200)
        items = resp.json().get("items", [])
        keys = [str(it.get("source_key")) for it in items]
        self.assertNotIn("srt:1:0:1000", keys)

    def test_glossary_upsert_writes_file(self) -> None:
        resp = self.client.post("/api/terms/glossary/upsert", json={"ja": "課題曲", "zh": "课题曲"})
        self.assertEqual(resp.status_code, 200)
        text = (self.root / "glossary.tsv").read_text(encoding="utf-8")
        self.assertIn("課題曲\t课题曲", text)

    def test_names_upsert_and_legacy_add(self) -> None:
        upsert = self.client.post(
            "/api/terms/names/upsert",
            json={"content": "ロイヤルブレッド", "sounds_like": ["ろいやるぶれっど", "ろいやる"]},
        )
        self.assertEqual(upsert.status_code, 200)
        listed = self.client.get("/api/terms/names")
        self.assertEqual(listed.status_code, 200)
        items = listed.json().get("items", [])
        self.assertTrue(any(it.get("content") == "ロイヤルブレッド" for it in items))

        legacy = self.client.post("/api/terms/names/add", json={"name": "KOP7th"})
        self.assertEqual(legacy.status_code, 200)
        text = (self.root / "names.txt").read_text(encoding="utf-8")
        self.assertIn("ロイヤルブレッド\tろいやるぶれっど,ろいやる", text)
        self.assertIn("KOP7th", text)


if __name__ == "__main__":
    unittest.main()
