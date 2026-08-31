from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from media_knowledge.providers.llm import AnswerProvider
from media_knowledge.qa.models import AnswerRequest, AnswerResponse, TokenUsage
from media_knowledge.transcripts.deep_correction import CorrectionChunk, LLMCorrectionRequest
from media_knowledge.transcripts.runtime import (
    AnswerProviderCorrectionLLM,
    FileCorrectionCheckpointStore,
)


class _Provider(AnswerProvider):
    name = "fake"
    model = "fake-model"

    def __init__(self, body: str) -> None:
        self.body = body
        self.request: AnswerRequest | None = None

    def generate(self, request: AnswerRequest) -> AnswerResponse:
        self.request = request
        return AnswerResponse(self.body, self.model, self.name, TokenUsage())


def _request() -> LLMCorrectionRequest:
    return LLMCorrectionRequest(
        "job",
        CorrectionChunk("chunk", 0, ("s1",), ("s1",), 0, 1000, 0, 1000),
        ({"segment_id": "s1", "raw_text": "原文"},),
        (),
        {},
        {},
        None,
    )


class DeepCorrectionRuntimeTests(unittest.TestCase):
    def test_llm_adapter_accepts_only_one_json_object(self) -> None:
        provider = _Provider("```json\n{\"ok\":true}\n```")
        payload = AnswerProviderCorrectionLLM(provider).correct(_request())
        self.assertEqual(json.loads(payload), {
            "ok": True,
            "schema_version": "deep-correction-response-v2",
        })
        assert provider.request is not None
        self.assertIn("不得把输入转写", provider.request.system_prompt)
        self.assertEqual(provider.request.response_format, "json_object")
        self.assertEqual(provider.request.max_output_tokens, 16_384)

    def test_llm_adapter_disables_thinking_only_for_deepseek(self) -> None:
        deepseek = _Provider('{"ok":true}')
        deepseek.model = "deepseek-v4-flash"
        AnswerProviderCorrectionLLM(deepseek).correct(_request())
        assert deepseek.request is not None
        self.assertEqual(deepseek.request.thinking_mode, "disabled")

        kimi = _Provider('{"ok":true}')
        kimi.model = "kimi-k3"
        AnswerProviderCorrectionLLM(kimi).correct(_request())
        assert kimi.request is not None
        self.assertIsNone(kimi.request.thinking_mode)

    def test_llm_adapter_does_not_salvage_surrounding_prose(self) -> None:
        provider = _Provider("说明：{\"ok\":true}")
        with self.assertRaisesRegex(RuntimeError, "完整 JSON"):
            AnswerProviderCorrectionLLM(provider).correct(_request())

    def test_llm_adapter_restores_only_constant_schema_version(self) -> None:
        provider = _Provider('{"chunk_id":"chunk"}')
        payload = json.loads(AnswerProviderCorrectionLLM(provider).correct(_request()))
        self.assertEqual(payload, {
            "chunk_id": "chunk",
            "schema_version": "deep-correction-response-v2",
        })

    def test_checkpoint_is_atomic_private_and_key_is_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileCorrectionCheckpointStore(directory)
            store.save("../../unsafe:chunk", {"response": "{}", "request_hash": "x"})
            files = list(Path(directory).glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertNotIn("unsafe", files[0].name)
            self.assertEqual(store.load("../../unsafe:chunk"), {
                "request_hash": "x", "response": "{}",
            })
            self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
