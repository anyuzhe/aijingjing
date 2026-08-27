from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from media_knowledge.answer_models import available_answer_models, resolve_answer_model
from media_knowledge.config import AppConfig, CompatibleQAProviderConfig
from media_knowledge.providers import CodexAnswerProvider, ExtractiveGroundedProvider, OpenAICompatibleAnswerProvider
from media_knowledge.qa.models import AnswerRequest, ImageAttachment
from media_knowledge.runtime import build_answer_provider


class AnswerModelTests(unittest.TestCase):
    def config(self, **overrides) -> AppConfig:
        values = {"database_path": Path(tempfile.gettempdir()) / "model-test.db"}
        values.update(overrides)
        return AppConfig(**values)

    def test_codex_catalog_exposes_current_and_compatible_models(self) -> None:
        config = self.config(qa_provider="codex")
        models = available_answer_models(config, codex_available=True)
        ids = [model.id for model in models]
        self.assertEqual(ids[:6], ["auto", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.5", "gpt-5.4"])
        self.assertIn("local-extractive", ids)
        self.assertEqual(next(model.id for model in models if model.default), "auto")

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value="/fake/codex")
    def test_auto_uses_luna_normally_and_terra_for_deep_analysis(self, _which) -> None:
        config = self.config(qa_provider="codex")
        regular = resolve_answer_model(config, "auto")
        deep = resolve_answer_model(config, "auto", deep_analysis=True)
        self.assertEqual((regular.model, regular.reasoning_effort), ("gpt-5.6-luna", "low"))
        self.assertEqual((deep.model, deep.reasoning_effort), ("gpt-5.6-terra", "medium"))

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value="/fake/codex")
    def test_runtime_builds_the_explicit_codex_model(self, _which) -> None:
        provider = build_answer_provider(
            self.config(qa_provider="codex"),
            model_id="gpt-5.6-sol",
            deep_analysis=True,
        )
        self.assertIsInstance(provider, CodexAnswerProvider)
        self.assertEqual(provider.model, "gpt-5.6-sol")
        self.assertEqual(provider.reasoning_effort, "high")

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value=None)
    def test_empty_selection_preserves_local_cli_default(self, _which) -> None:
        provider = build_answer_provider(self.config(qa_provider="extractive"))
        self.assertIsInstance(provider, ExtractiveGroundedProvider)

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value=None)
    def test_compatible_models_are_allowlisted_from_configuration(self, _which) -> None:
        config = self.config(
            qa_provider="openai-compatible",
            qa_model="deepseek-chat",
            qa_models=("deepseek-chat", "qwen-max"),
            qa_base_url="https://provider.example/v1",
            qa_api_key="test-key",
        )
        models = available_answer_models(config)
        self.assertEqual(
            [model.id for model in models],
            ["local-extractive", "compatible::default::deepseek-chat", "compatible::default::qwen-max"],
        )
        self.assertEqual(resolve_answer_model(config, "compatible::default::qwen-max").model, "qwen-max")
        with self.assertRaisesRegex(ValueError, "不可用"):
            resolve_answer_model(config, "compatible::unconfigured-model")

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value=None)
    def test_multiple_compatible_providers_keep_credentials_server_side(self, _which) -> None:
        config = self.config(
            qa_provider="extractive",
            qa_compatible_providers=(
                CompatibleQAProviderConfig(
                    "kimi", "Kimi", "https://api.moonshot.cn/v1", "kimi-secret", ("kimi-k2.5",)
                ),
                CompatibleQAProviderConfig(
                    "deepseek", "DeepSeek", "https://api.deepseek.com", "deepseek-secret", ("deepseek-v4-flash",)
                ),
            ),
        )
        models = available_answer_models(config)
        serialized = str([model.to_dict() for model in models])
        self.assertIn("compatible::kimi::kimi-k2.5", serialized)
        self.assertIn("compatible::deepseek::deepseek-v4-flash", serialized)
        self.assertNotIn("secret", serialized)
        provider = build_answer_provider(config, model_id="compatible::kimi::kimi-k2.5")
        self.assertIsInstance(provider, OpenAICompatibleAnswerProvider)
        self.assertEqual(provider.model, "kimi-k2.5")

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value="/fake/codex")
    def test_multi_provider_default_can_select_deepseek_flash(self, _which) -> None:
        config = self.config(
            qa_provider="deepseek",
            qa_model="deepseek-v4-flash",
            qa_compatible_providers=(
                CompatibleQAProviderConfig(
                    "deepseek",
                    "DeepSeek",
                    "https://api.deepseek.com",
                    "deepseek-secret",
                    ("deepseek-v4-pro", "deepseek-v4-flash"),
                ),
            ),
        )
        default = next(model for model in available_answer_models(config) if model.default)
        self.assertEqual(default.id, "compatible::deepseek::deepseek-v4-flash")

    @mock.patch("media_knowledge.answer_models.shutil.which", return_value=None)
    def test_deepseek_vision_is_selectable_image_capable_and_default(self, _which) -> None:
        config = self.config(
            qa_provider="deepseek",
            qa_model="deepseek-v4-flash-vision-exp",
            qa_compatible_providers=(
                CompatibleQAProviderConfig(
                    "deepseek",
                    "DeepSeek",
                    "https://api.deepseek.com",
                    "deepseek-secret",
                    ("deepseek-v4-flash-vision-exp", "deepseek-v4-flash"),
                ),
            ),
        )
        models = available_answer_models(config)
        vision = next(item for item in models if item.model == "deepseek-v4-flash-vision-exp")
        self.assertTrue(vision.supports_images)
        self.assertTrue(vision.default)
        self.assertIn("视觉", vision.label)

    def test_openai_compatible_provider_sends_real_multimodal_content_array(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "question.png"
            image.write_bytes(b"normalized-image")
            provider = OpenAICompatibleAnswerProvider(
                "https://provider.example/v1", "test-key", "deepseek-v4-flash-vision-exp"
            )
            response = {
                "choices": [{"message": {"content": "图片中是一个流程图。"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            }
            with mock.patch.object(provider, "request_json", return_value=response) as request:
                result = provider.generate(
                    AnswerRequest(
                        "图里是什么？",
                        "system",
                        "user prompt",
                        [],
                        "zh-CN",
                        [ImageAttachment(str(image), image.name)],
                    )
                )
            payload = request.call_args.args[0]
            content = payload["messages"][1]["content"]
            self.assertIsInstance(content, list)
            self.assertEqual(content[0], {"type": "text", "text": "user prompt"})
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertEqual(result.markdown, "图片中是一个流程图。")


if __name__ == "__main__":
    unittest.main()
