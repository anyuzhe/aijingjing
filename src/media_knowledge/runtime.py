from __future__ import annotations

from .answer_models import resolve_answer_model
from .config import AppConfig
from .embedding import FastEmbedProvider, HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from .embedding.base import EmbeddingProvider
from .rerank import DisabledRerankProvider, HTTPRerankProvider, LocalLexicalRerankProvider, RerankProvider
from .providers import (
    AnswerProvider,
    CodexAnswerProvider,
    ExtractiveGroundedProvider,
    OpenAICompatibleAnswerProvider,
)


def build_embedding_provider(config: AppConfig) -> EmbeddingProvider:
    provider = config.embedding_provider.casefold()
    if provider in {"fastembed", "semantic", "onnx"}:
        return FastEmbedProvider(config.embedding_model, cache_dir=config.embedding_cache_dir)
    if provider in {"hash", "local", "local-hash"}:
        return HashEmbeddingProvider(config.embedding_dimensions, config.embedding_model)
    if provider in {"openai", "openai-compatible"}:
        if not config.embedding_base_url or not config.embedding_api_key:
            raise ValueError(
                "OpenAI-compatible embeddings require KNOWLEDGE_EMBEDDING_BASE_URL and "
                "KNOWLEDGE_EMBEDDING_API_KEY"
            )
        return OpenAICompatibleEmbeddingProvider(
            config.embedding_base_url, config.embedding_api_key, config.embedding_model
        )
    raise ValueError(f"unsupported embedding provider: {config.embedding_provider}")


def build_rerank_provider(config: AppConfig) -> RerankProvider:
    provider = config.rerank_provider.casefold()
    if provider in {"local", "local-lexical"}:
        return LocalLexicalRerankProvider()
    if provider in {"disabled", "none", "off"}:
        return DisabledRerankProvider()
    if provider in {"api", "http", "openai-compatible"}:
        if not config.rerank_base_url or not config.rerank_api_key or not config.rerank_model:
            raise ValueError(
                "API reranking requires KNOWLEDGE_RERANK_BASE_URL, KNOWLEDGE_RERANK_API_KEY, "
                "and KNOWLEDGE_RERANK_MODEL"
            )
        return HTTPRerankProvider(
            config.rerank_base_url, config.rerank_api_key, config.rerank_model
        )
    raise ValueError(f"unsupported rerank provider: {config.rerank_provider}")


def build_answer_provider(
    config: AppConfig,
    *,
    model_id: str | None = None,
    deep_analysis: bool = False,
) -> AnswerProvider:
    choice = resolve_answer_model(config, model_id, deep_analysis=deep_analysis)
    if choice.provider == "local":
        return ExtractiveGroundedProvider()
    if choice.provider == "codex":
        return CodexAnswerProvider(
            model=choice.model,
            reasoning_effort=choice.reasoning_effort or "low",
        )
    if choice.provider == "openai-compatible":
        if not config.qa_base_url or not config.qa_api_key:
            raise ValueError(
                "OpenAI-compatible QA requires KNOWLEDGE_QA_BASE_URL, KNOWLEDGE_QA_API_KEY, "
                "and KNOWLEDGE_QA_MODEL or KNOWLEDGE_QA_MODELS"
            )
        return OpenAICompatibleAnswerProvider(config.qa_base_url, config.qa_api_key, choice.model)
    if choice.provider.startswith("openai-compatible:"):
        provider_id = choice.provider.partition(":")[2]
        provider_config = next(
            (provider for provider in config.qa_compatible_providers if provider.id == provider_id),
            None,
        )
        if provider_config is None:
            raise ValueError(f"configured QA provider is unavailable: {provider_id}")
        return OpenAICompatibleAnswerProvider(
            provider_config.base_url,
            provider_config.api_key,
            choice.model,
            temperature=provider_config.temperature,
        )
    raise ValueError(f"unsupported QA model provider: {choice.provider}")
