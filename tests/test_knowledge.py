from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from live_sub_daemon.knowledge import KnowledgeStore


class KnowledgeStoreTests(unittest.TestCase):
    def test_names_with_sounds_like_only_use_content_for_prompt_side(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            glossary = root / "glossary.tsv"
            names = root / "names.txt"
            glossary.write_text("譜面\t谱面\n", encoding="utf-8")
            names.write_text(
                "\n".join(
                    [
                        "よしき\tよしき,ヨシキ",
                        "ロイヤルブレッド|ろいやるぶれっど",
                        "KOP7th",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            store = KnowledgeStore(glossary_path=glossary, names_path=names, reload_interval_sec=0.0)
            snapshot = store.get_snapshot()

            self.assertEqual(snapshot.names_whitelist, ["よしき", "ロイヤルブレッド", "KOP7th"])


if __name__ == "__main__":
    unittest.main()

