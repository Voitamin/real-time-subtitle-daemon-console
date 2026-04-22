from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from live_sub_daemon.source_reader import SourceReader


class SourceReaderTests(unittest.TestCase):
    def test_parse_srt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            srt = root / "jp.srt"
            srt.write_text(
                """
1
00:00:01,000 --> 00:00:02,000
こんにちは

2
00:00:03,000 --> 00:00:04,500
世界
""".strip(),
                encoding="utf-8",
            )
            reader = SourceReader(srt_path=srt, txt_path=root / "jp.txt")
            cues = reader.poll()

            self.assertEqual(len(cues), 2)
            self.assertEqual(cues[0].source_key, "srt:1:1000:2000")
            self.assertEqual(cues[1].jp_raw, "世界")


if __name__ == "__main__":
    unittest.main()
