from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from live_sub_daemon.asr_speechmatics import SpeechmaticsSourceReader
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
    SpeechmaticsConfig,
    StateConfig,
)
from live_sub_daemon.runtime_control import RuntimeController


def _build_runtime(root: Path) -> RuntimeController:
    cfg = AppConfig(
        source=SourceConfig(
            input_srt=root / "jp.srt",
            input_txt=root / "jp.txt",
            mode="speechmatics",
            speechmatics=SpeechmaticsConfig(api_key="dummy"),
        ),
        output=OutputConfig(zh_txt=root / "zh.txt", zh_srt=root / "zh.srt"),
        llm=LLMConfig(api_key="dummy"),
        align=AlignConfig(),
        render=RenderConfig(),
        pipeline=PipelineConfig(),
        adaptive=AdaptiveConfig(),
        console=ConsoleConfig(),
        state=StateConfig(db_path=root / "state.sqlite3"),
        fallback=FallbackConfig(),
        knowledge=KnowledgeConfig(glossary_path=root / "glossary.tsv", names_path=root / "names.txt"),
    )
    return RuntimeController(cfg)


class _FakeClient:
    def __init__(self) -> None:
        self.handlers = {}

    def on(self, event: str):
        def decorator(fn):
            self.handlers[event] = fn
            return fn

        return decorator


class _FakeTranscriptResult:
    @staticmethod
    def from_message(message):
        return SimpleNamespace(metadata=SimpleNamespace(transcript=message.get("text", "")))


class SpeechmaticsSourceTests(unittest.TestCase):
    def test_partial_not_enqueued_and_final_enqueued(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = _build_runtime(root)
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(api_key="dummy", final_aggregate_enabled=False),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=runtime,
                autostart=False,
            )
            fake_client = _FakeClient()
            sdk = {
                "ServerMessageType": SimpleNamespace(
                    ADD_PARTIAL_TRANSCRIPT="partial",
                    ADD_TRANSCRIPT="final",
                ),
                "TranscriptResult": _FakeTranscriptResult,
            }
            reader._bind_handlers(fake_client, sdk)  # type: ignore[arg-type]

            fake_client.handlers["partial"]({"text": "部分"})
            self.assertEqual(reader.poll(), [])

            fake_client.handlers["final"]({"text": "確定文"})
            cues = reader.poll()
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0].source_kind, "speechmatics")
            self.assertEqual(cues[0].jp_raw, "確定文")
            self.assertIsNotNone(runtime.get_status().get("asr_last_final_mono_ms"))

            reader.close()

    def test_final_aggregation_merges_and_flushes_on_gap(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(
                    api_key="dummy",
                    final_aggregate_enabled=True,
                    final_aggregate_mode="v2",
                    final_aggregate_gap_sec=0.5,
                    final_aggregate_max_sec=30.0,
                    final_aggregate_min_chars=1,
                    final_aggregate_max_chars=200,
                    final_aggregate_flush_on_punct=False,
                ),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=None,
                autostart=False,
            )
            reader._handle_final_text("こちらの", now_mono_ms=1000)  # type: ignore[attr-defined]
            reader._handle_final_text("お二人", now_mono_ms=1200)  # type: ignore[attr-defined]
            self.assertEqual(reader.poll(), [])

            reader._flush_pending_if_due(now_mono_ms=1800)  # type: ignore[attr-defined]
            cues = reader.poll()
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0].jp_raw, "こちらのお二人")
            reader.close()

    def test_additional_vocab_names_priority_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "names.txt").write_text("name_a\nname_b\n", encoding="utf-8")
            (root / "glossary.tsv").write_text("name_b\t名字B\nterm_x\t术语X\nterm_y\t术语Y\n", encoding="utf-8")
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(api_key="dummy"),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=None,
                autostart=False,
            )

            vocab = reader._build_additional_vocab(limit=3)  # type: ignore[attr-defined]
            self.assertEqual([it["content"] for it in vocab], ["name_a", "name_b", "term_x"])
            reader.close()

    def test_additional_vocab_supports_sounds_like_from_names_whitelist(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "names.txt").write_text(
                "よしき\tよしき,ヨシキ\nロイヤルブレッド|ろいやるぶれっど\nKOP7th\n",
                encoding="utf-8",
            )
            (root / "glossary.tsv").write_text("term_x\t术语X\n", encoding="utf-8")
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(api_key="dummy", additional_vocab_from="names_whitelist"),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=None,
                autostart=False,
            )

            vocab = reader._build_additional_vocab(limit=10)  # type: ignore[attr-defined]
            self.assertEqual([it["content"] for it in vocab], ["よしき", "ロイヤルブレッド", "KOP7th"])
            self.assertEqual(vocab[0].get("sounds_like"), ["よしき", "ヨシキ"])
            self.assertEqual(vocab[1].get("sounds_like"), ["ろいやるぶれっど"])
            self.assertNotIn("sounds_like", vocab[2])
            reader.close()

    def test_reconnect_sets_error_and_keeps_loop_alive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = _build_runtime(root)
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(api_key="dummy"),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=runtime,
                autostart=False,
            )

            calls = {"count": 0}

            async def fake_run_single_session() -> None:
                calls["count"] += 1
                if calls["count"] >= 2:
                    reader._stop_event.set()  # type: ignore[attr-defined]
                raise RuntimeError("boom")

            reader._run_single_session = fake_run_single_session  # type: ignore[method-assign]
            with patch("live_sub_daemon.asr_speechmatics.asyncio.sleep", new=AsyncMock(return_value=None)):
                asyncio.run(reader._run_forever())  # type: ignore[attr-defined]

            status = runtime.get_status()
            self.assertGreaterEqual(calls["count"], 2)
            self.assertEqual(status.get("asr_connected"), False)
            self.assertIn("boom", str(status.get("asr_last_error")))
            reader.close()

    def test_close_terminates_capture_proc_sync(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(api_key="dummy"),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=None,
                autostart=False,
            )
            fake_proc = Mock()
            fake_proc.returncode = None
            reader._capture_proc = fake_proc  # type: ignore[attr-defined]
            reader._terminate_capture_proc_sync()  # type: ignore[attr-defined]
            self.assertTrue(fake_proc.terminate.called or fake_proc.kill.called)
            reader.close()

    def test_name_canonicalizer_applies_sounds_like_before_queue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "names.txt").write_text("KOP7th\tkop seven\n", encoding="utf-8")
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(
                    api_key="dummy",
                    final_aggregate_enabled=False,
                ),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                names_phonetic_canonicalize_enabled=True,
                names_phonetic_max_rules=10,
                runtime=None,
                autostart=False,
            )
            reader._handle_final_text("we are kop seven", now_mono_ms=1000)  # type: ignore[attr-defined]
            cues = reader.poll()
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0].jp_raw, "we are kop seven")
            self.assertEqual(cues[0].jp_canonicalized, "we are KOP7th")
            reader.close()

    def test_v2_overlap_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reader = SpeechmaticsSourceReader(
                config=SpeechmaticsConfig(
                    api_key="dummy",
                    final_aggregate_enabled=True,
                    final_aggregate_mode="v2",
                    final_aggregate_gap_sec=2.0,
                    final_aggregate_max_sec=30.0,
                    final_aggregate_force_emit_sec=30.0,
                    final_aggregate_min_chars=1,
                    final_aggregate_max_chars=200,
                    final_aggregate_overlap_dedup=True,
                    final_aggregate_flush_on_punct=False,
                ),
                glossary_path=root / "glossary.tsv",
                names_path=root / "names.txt",
                runtime=None,
                autostart=False,
            )
            reader._handle_final_text("royal", now_mono_ms=1000)  # type: ignore[attr-defined]
            reader._handle_final_text("royal bread", now_mono_ms=1200)  # type: ignore[attr-defined]
            reader._flush_pending_if_due(now_mono_ms=4000)  # type: ignore[attr-defined]
            cues = reader.poll()
            self.assertEqual(len(cues), 1)
            self.assertEqual(cues[0].jp_canonicalized, "royal bread")
            reader.close()


if __name__ == "__main__":
    unittest.main()
