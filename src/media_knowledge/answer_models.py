from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, replace

from .config import AppConfig


@dataclass(frozen=True, slots=True)
class AnswerModelSpec:
    id: str
    label: str
    provider: str
    model: str
    description: str
    reasoning_effort: str | None = None
    deep_reasoning_effort: str | None = None
    default: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


CODEX_MODEL_SPECS = (
    AnswerModelSpec(
        "auto",
        "自动",
        "codex",
        "gpt-5.6-luna",
        "普通问答使用 Luna；深度分析自动切换到 Terra。",
        "low",
        "medium",
    ),
    AnswerModelSpec(
        "gpt-5.6-luna",
        "GPT-5.6 Luna · 极速",
        "codex",
        "gpt-5.6-luna",
        "适合高频、低延迟的日常知识问答。",
        "low",
        "medium",
    ),
    AnswerModelSpec(
        "gpt-5.6-terra",
        "GPT-5.6 Terra · 均衡",
        "codex",
        "gpt-5.6-terra",
        "在回答质量、速度和成本之间取得平衡。",
        "medium",
        "high",
    ),
    AnswerModelSpec(
        "gpt-5.6-sol",
        "GPT-5.6 Sol · 最强",
        "codex",
        "gpt-5.6-sol",
        "适合复杂推理、跨资料综合和重要结论。",
        "medium",
        "high",
    ),
    AnswerModelSpec(
        "gpt-5.5",
        "GPT-5.5 · 兼容",
        "codex",
        "gpt-5.5",
        "保留 GPT-5.5 工作流兼容性。",
        "low",
        "high",
    ),
    AnswerModelSpec(
        "gpt-5.4",
        "GPT-5.4 · 兼容",
        "codex",
        "gpt-5.4",
        "保留 GPT-5.4 工作流兼容性。",
        "low",
        "high",
    ),
)

LOCAL_MODEL = AnswerModelSpec(
    "local-extractive",
    "本地证据模型 · 离线",
    "local",
    "grounded-extractive-v1",
    "完全离线，仅压缩和引用检索到的证据。",
)


def _compatible_models(config: AppConfig) -> list[AnswerModelSpec]:
    choices: list[AnswerModelSpec] = []
    if config.qa_base_url and config.qa_api_key:
        names = config.qa_models or ((config.qa_model,) if config.qa_model else ())
        choices.extend(
            AnswerModelSpec(
                f"compatible::default::{model}",
                f"{model} · 兼容接口",
                "openai-compatible",
                model,
                "使用环境变量配置的 OpenAI-compatible 回答接口。",
            )
            for model in names
            if model and model != "grounded-extractive-v1"
        )
    for provider in config.qa_compatible_providers:
        choices.extend(
            AnswerModelSpec(
                f"compatible::{provider.id}::{model}",
                f"{model} · {provider.label}",
                f"openai-compatible:{provider.id}",
                model,
                f"使用本机配置的 {provider.label} API。",
            )
            for model in provider.models
        )
    return choices


def available_answer_models(
    config: AppConfig,
    *,
    codex_available: bool | None = None,
) -> list[AnswerModelSpec]:
    has_codex = bool(shutil.which("codex")) if codex_available is None else codex_available
    choices = [*(CODEX_MODEL_SPECS if has_codex else ()), LOCAL_MODEL, *_compatible_models(config)]
    provider = config.qa_provider.casefold()
    if provider in {"codex", "codex-local"} and has_codex:
        default_id = "auto"
    elif provider in {"openai", "openai-compatible"} and _compatible_models(config):
        default_id = _compatible_models(config)[0].id
    elif any(item.id == provider for item in config.qa_compatible_providers):
        provider_choices = [
            item
            for item in _compatible_models(config)
            if item.provider == f"openai-compatible:{provider}"
        ]
        selected = next(
            (item for item in provider_choices if item.model == config.qa_model),
            provider_choices[0],
        )
        default_id = selected.id
    else:
        default_id = LOCAL_MODEL.id
    return [replace(choice, default=choice.id == default_id) for choice in choices]


def resolve_answer_model(
    config: AppConfig,
    requested: str | None,
    *,
    deep_analysis: bool = False,
) -> AnswerModelSpec:
    choices = available_answer_models(config)
    requested_id = str(requested or "").strip()
    if not requested_id:
        choice = next(item for item in choices if item.default)
    elif requested_id.casefold() in {"auto", "automatic"}:
        choice = next((item for item in choices if item.id == "auto"), None)
        if choice is None:
            choice = next(item for item in choices if item.default)
    else:
        choice = next(
            (
                item
                for item in choices
                if item.id.casefold() == requested_id.casefold()
                or item.model.casefold() == requested_id.casefold()
            ),
            None,
        )
    if choice is None:
        raise ValueError("所选回答模型不可用，请刷新页面后重新选择")
    if choice.id == "auto" and deep_analysis:
        return replace(
            choice,
            model="gpt-5.6-terra",
            reasoning_effort=choice.deep_reasoning_effort,
        )
    if deep_analysis and choice.deep_reasoning_effort:
        return replace(choice, reasoning_effort=choice.deep_reasoning_effort)
    return choice
