from __future__ import annotations

import unittest

from live_sub_daemon.models import CueRecord
from live_sub_daemon.pipeline import (
    _apply_ai_join_plan,
    _cue_stage1_input,
    build_correct_prompt_lite,
    build_correct_prompt_lite_ai_join,
)


class PipelineS1Tests(unittest.TestCase):
    def _cue(self, key: str, text: str) -> CueRecord:
        return CueRecord(
            source_key=key,
            source_kind="speechmatics",
            srt_index=None,
            start_ms=None,
            end_ms=None,
            jp_raw=text,
            jp_aggregated=text,
            jp_canonicalized=text,
            jp_corrected=None,
            zh_text=None,
            status="NEW",
            t_seen_mono_ms=0,
            due_mono_ms=0,
            translated_mono_ms=None,
            dropped_late=False,
            llm_latency_ms=None,
        )

    def test_stage1_input_prefers_canonicalized(self) -> None:
        cue = self._cue("k1", "raw")
        cue.jp_aggregated = "agg"
        cue.jp_canonicalized = "canon"
        self.assertEqual(_cue_stage1_input(cue), "canon")

    def test_lite_prompt_mentions_normalization_only(self) -> None:
        prompt = build_correct_prompt_lite(
            glossary={"譜面": "谱面"},
            names_whitelist=["KOP7th"],
            glossary_limit=100,
            names_limit=100,
            punctuation_only=True,
        )
        self.assertIn("normalization only", prompt)
        self.assertIn("JSON array", prompt)
        self.assertIn("KOP7th", prompt)

    def test_lite_ai_join_prompt_mentions_marker(self) -> None:
        prompt = build_correct_prompt_lite_ai_join(
            glossary={},
            names_whitelist=["KOP7th"],
            glossary_limit=10,
            names_limit=10,
            marker="<<JOIN_NEXT>>",
            allow_light_compress=True,
        )
        self.assertIn("<<JOIN_NEXT>>", prompt)
        self.assertIn("lightly compress", prompt)

    def test_ai_join_plan_suppresses_joined_line(self) -> None:
        cues = [
            self._cue("k1", "こちらの"),
            self._cue("k2", "お二人"),
            self._cue("k3", "です"),
        ]
        corrected = {
            "k1": "<<JOIN_NEXT>>こちらの",
            "k2": "<<JOIN_NEXT>>お二人",
            "k3": "です",
        }
        merged, suppressed, targets = _apply_ai_join_plan(
            cues=cues,
            corrected_texts=corrected,
            marker="<<JOIN_NEXT>>",
            max_chain=0,
            keep_min_chars=0,
        )
        self.assertEqual(suppressed["k1"], True)
        self.assertEqual(suppressed["k2"], True)
        self.assertEqual(targets["k1"], "k2")
        self.assertEqual(targets["k2"], "k3")
        self.assertEqual(merged["k3"], "こちらのお二人です")


if __name__ == "__main__":
    unittest.main()
