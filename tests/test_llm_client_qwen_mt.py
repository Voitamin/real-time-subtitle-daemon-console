from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from live_sub_daemon.llm_client import XAIClient
from live_sub_daemon.pipeline import build_qwen_mt_translation_options


class _FakeHTTPResponse:
    def __init__(self, body: str):
        self._body = body

    def read(self) -> bytes:
        return self._body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class QwenMTClientTests(unittest.TestCase):
    def test_qwen_mt_uses_user_only_messages_and_plain_text_output(self) -> None:
        captured_bodies = []

        def fake_urlopen(req, timeout=0):  # noqa: ANN001
            captured_bodies.append(json.loads(req.data.decode("utf-8")))
            body = {
                "choices": [{"message": {"content": "翻译结果"}}],
                "usage": {"prompt_tokens_details": {"cached_tokens": 9}},
            }
            return _FakeHTTPResponse(json.dumps(body, ensure_ascii=False))

        client = XAIClient(api_key="dummy", base_url="https://example.test/v1")
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.run_batch(
                model="qwen-mt-plus",
                stage_name="jp_to_zh",
                system_prompt="should not be sent",
                items=[{"source_key": "k1", "text": "現在の文です。"}],
                timeout_sec=5.0,
                max_retries=0,
                retry_backoff_sec=0.0,
                temperature=0.0,
                translation_options={
                    "source_lang": "Japanese",
                    "target_lang": "Chinese",
                    "tm_list": [{"source": "課題曲", "target": "课题曲"}],
                },
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.outputs.get("k1"), "翻译结果")
        self.assertEqual(result.cached_tokens, 9)
        self.assertEqual(len(captured_bodies), 1)

        body = captured_bodies[0]
        self.assertEqual(body["model"], "qwen-mt-plus")
        self.assertEqual(body["messages"], [{"role": "user", "content": "現在の文です。"}])
        self.assertEqual(body["translation_options"]["source_lang"], "Japanese")
        self.assertEqual(body["translation_options"]["target_lang"], "Chinese")
        self.assertEqual(body["translation_options"]["tm_list"], [{"source": "課題曲", "target": "课题曲"}])

    def test_non_qwen_mt_keeps_json_contract(self) -> None:
        captured_bodies = []

        def fake_urlopen(req, timeout=0):  # noqa: ANN001
            captured_bodies.append(json.loads(req.data.decode("utf-8")))
            body = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                [{"source_key": "k1", "text": "修正结果"}],
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
            return _FakeHTTPResponse(json.dumps(body, ensure_ascii=False))

        client = XAIClient(api_key="dummy", base_url="https://example.test/v1")
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = client.run_batch(
                model="qwen-plus",
                stage_name="jp_correct",
                system_prompt="S1 prompt",
                items=[{"source_key": "k1", "text": "原文"}],
                timeout_sec=5.0,
                max_retries=0,
                retry_backoff_sec=0.0,
                temperature=0.0,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.outputs.get("k1"), "修正结果")
        self.assertEqual(len(captured_bodies), 1)

        body = captured_bodies[0]
        self.assertEqual(body["model"], "qwen-plus")
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][0]["content"], "S1 prompt")
        self.assertEqual(body["messages"][1]["role"], "user")
        user_payload = json.loads(body["messages"][1]["content"])
        self.assertEqual(user_payload["stage"], "jp_correct")
        self.assertEqual(user_payload["items"][0]["source_key"], "k1")


class QwenMTTranslationOptionTests(unittest.TestCase):
    def test_build_qwen_mt_translation_options(self) -> None:
        options = build_qwen_mt_translation_options(
            glossary={"課題曲": "课题曲", "安定感": "稳定性"},
            names_whitelist=["湯毛", "課題曲"],
            glossary_limit=1,
            names_limit=10,
        )
        self.assertEqual(options["source_lang"], "Japanese")
        self.assertEqual(options["target_lang"], "Chinese")
        self.assertEqual(
            options["tm_list"],
            [
                {"source": "課題曲", "target": "课题曲"},
                {"source": "湯毛", "target": "湯毛"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
