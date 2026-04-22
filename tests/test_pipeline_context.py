from __future__ import annotations

import unittest

from live_sub_daemon.pipeline import _compose_translate_input


class PipelineContextTests(unittest.TestCase):
    def test_no_context_returns_current(self) -> None:
        current = "現在の文です。"
        self.assertEqual(_compose_translate_input(current, []), current)

    def test_context_wrap_format(self) -> None:
        current = "現在の文です。"
        context = ["前文1。", "前文2。"]
        wrapped = _compose_translate_input(current, context)
        self.assertIn("[CONTEXT]", wrapped)
        self.assertIn("- 前文1。", wrapped)
        self.assertIn("- 前文2。", wrapped)
        self.assertIn("[CURRENT]", wrapped)
        self.assertTrue(wrapped.endswith(current))


if __name__ == "__main__":
    unittest.main()

