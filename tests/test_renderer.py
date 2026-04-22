from __future__ import annotations

import unittest

from live_sub_daemon.models import CueRecord
from live_sub_daemon.renderer import build_render_decision, paginate_overlay_text, split_text_to_lines


class RenderTests(unittest.TestCase):
    def _cue(self, key: str, text: str) -> CueRecord:
        return CueRecord(
            source_key=key,
            source_kind="srt",
            srt_index=1,
            start_ms=0,
            end_ms=1000,
            jp_raw=text,
            jp_aggregated=text,
            jp_canonicalized=text,
            jp_corrected=text,
            zh_text=text,
            status="TRANSLATED",
            t_seen_mono_ms=0,
            due_mono_ms=0,
            translated_mono_ms=0,
            dropped_late=False,
            llm_latency_ms=100,
        )

    def test_render_uses_latest_single_cue(self) -> None:
        latest = self._cue("k2", "短句")
        prev = self._cue("k1", "上一句")
        decision = build_render_decision([latest, prev], char_threshold=18)
        self.assertEqual(decision.text, "短句")

    def test_single_line_for_long_latest(self) -> None:
        latest = self._cue("k2", "这是一个超过阈值很多很多很多很多字的长句")
        prev = self._cue("k1", "上一句")
        decision = build_render_decision([latest, prev], char_threshold=18)
        self.assertEqual(decision.text, latest.zh_text)

    def test_overlay_text_paginate_two_lines(self) -> None:
        text = "采用了摄像机拍摄的方式。可能看起来会有些不便，但这也是为了选手着想，希望大家能理解。再多一点点。"
        pages = paginate_overlay_text(text, max_total_chars=42, max_lines=2)
        self.assertGreaterEqual(len(pages), 2)
        reconstructed = "".join(page.replace("\n", "") for page in pages)
        self.assertEqual(reconstructed, text)
        for page in pages:
            lines = page.split("\n")
            self.assertLessEqual(len(lines), 2)
            self.assertLessEqual(len("".join(lines)), 42)

    def test_split_text_to_lines_keeps_all_content(self) -> None:
        text = "第一句非常非常长需要被切开第二句也很长需要继续切开"
        lines = split_text_to_lines(text, line_char_cap=8)
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual("".join(lines), text)
        for line in lines:
            self.assertLessEqual(len(line), 8)


if __name__ == "__main__":
    unittest.main()
