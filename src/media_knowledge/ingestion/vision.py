from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ..config import AppConfig
from ..providers import OpenAICompatibleAnswerProvider


class MultimodalInterpreter:
    """Use the selected DeepSeek or Kimi multimodal model for visual understanding."""

    supported_models = {
        "deepseek-v4-flash-vision-exp",
        "kimi-k3",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
    }

    def __init__(self, config: AppConfig, *, enabled: bool = True, max_images: int = 12) -> None:
        self.enabled = enabled
        self.remaining = max(0, max_images)
        provider = next(
            (
                item for item in config.qa_compatible_providers
                if item.id == config.qa_provider and config.qa_model in self.supported_models
                and config.qa_model in item.models
            ),
            None,
        )
        model = config.qa_model if provider else None
        if provider is None:
            provider = next(
                (
                    item for item in config.qa_compatible_providers
                    if any(name in self.supported_models for name in item.models)
                ),
                None,
            )
            model = next(
                (name for name in provider.models if name in self.supported_models),
                None,
            ) if provider else None
        self.provider = (
            OpenAICompatibleAnswerProvider(
                provider.base_url,
                provider.api_key,
                model,
                temperature=provider.temperature,
                timeout=180,
            )
            if enabled and provider and model
            else None
        )

    @property
    def available(self) -> bool:
        return self.provider is not None and self.remaining > 0

    def describe(self, image_path: str | Path, *, context: str = "") -> str:
        if not self.available or self.provider is None:
            return ""
        path = Path(image_path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "请用简体中文分析这张知识资料图片。不要只做 OCR；请说明图表、流程、空间关系、"
            "关键数字、结论及其在上下文中的作用。对无法确认的内容明确标注不确定。"
        )
        if context.strip():
            prompt += "\n\n上下文：" + context.strip()[:4000]
        payload: dict[str, object] = {
            "model": self.provider.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        body = self.provider.request_json(payload)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("视觉模型没有返回分析结果")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("视觉模型返回了空内容")
        self.remaining -= 1
        return content.strip()
