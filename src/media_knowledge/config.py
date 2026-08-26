from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


KEYRING_SERVICE = "com.aijingjing.knowledge"


def _keyring_secret(reference: str) -> str:
    if not reference:
        return ""
    try:
        import keyring
        return str(keyring.get_password(KEYRING_SERVICE, reference) or "").strip()
    except Exception:
        return ""


@dataclass(frozen=True, slots=True)
class CompatibleQAProviderConfig:
    id: str
    label: str
    base_url: str
    api_key: str
    models: tuple[str, ...]
    temperature: float | None = 0.1


def _load_qa_providers(path: Path) -> tuple[CompatibleQAProviderConfig, ...]:
    if not path.is_file():
        return ()
    if path.stat().st_size > 64 * 1024:
        raise ValueError("QA provider config is too large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid QA provider config: {path}") from exc
    values = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError("QA provider config must contain a providers list")
    providers: list[CompatibleQAProviderConfig] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each QA provider must be an object")
        provider_id = str(value.get("id") or "").strip().casefold()
        label = str(value.get("label") or provider_id).strip()
        base_url = str(value.get("base_url") or "").strip().rstrip("/")
        api_key = str(value.get("api_key") or "").strip()
        if not api_key:
            api_key = _keyring_secret(str(value.get("credential_ref") or ""))
        raw_models = value.get("models")
        raw_temperature = value.get("temperature", 0.1)
        if raw_temperature is None:
            temperature = None
        elif isinstance(raw_temperature, (int, float)) and not isinstance(raw_temperature, bool):
            temperature = float(raw_temperature)
            if not 0 <= temperature <= 2:
                raise ValueError(f"invalid QA provider temperature: {provider_id}")
        else:
            raise ValueError(f"invalid QA provider temperature: {provider_id}")
        models = tuple(
            dict.fromkeys(
                str(model).strip()
                for model in (raw_models if isinstance(raw_models, list) else [])
                if str(model).strip()
            )
        )
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", provider_id):
            raise ValueError("QA provider id must contain only lowercase letters, numbers, _ or -")
        if provider_id in seen:
            raise ValueError(f"duplicate QA provider id: {provider_id}")
        if not label or not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError(f"invalid QA provider endpoint: {provider_id}")
        if not api_key or not models:
            raise ValueError(f"QA provider requires api_key and models: {provider_id}")
        seen.add(provider_id)
        providers.append(
            CompatibleQAProviderConfig(provider_id, label, base_url, api_key, models, temperature)
        )
    return tuple(providers)


@dataclass(slots=True)
class AppConfig:
    database_path: Path
    embedding_provider: str = "hash"
    embedding_model: str = "hash-384-v1"
    embedding_dimensions: int = 384
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_cache_dir: Path | None = None
    rerank_provider: str = "local"
    rerank_model: str | None = None
    rerank_base_url: str | None = None
    rerank_api_key: str | None = None
    qa_provider: str = "extractive"
    qa_model: str = "grounded-extractive-v1"
    qa_models: tuple[str, ...] = ()
    qa_base_url: str | None = None
    qa_api_key: str | None = None
    qa_compatible_providers: tuple[CompatibleQAProviderConfig, ...] = ()
    qa_providers_file: Path | None = None
    obsidian_vault_root: Path | None = None

    @classmethod
    def from_env(cls, database_path: str | Path | None = None) -> "AppConfig":
        path = Path(database_path or os.environ.get("KNOWLEDGE_DB", ".knowledge/knowledge.db")).expanduser().resolve()
        vault = os.environ.get("KNOWLEDGE_OBSIDIAN_VAULT")
        providers_path = Path(
            os.environ.get("KNOWLEDGE_QA_PROVIDERS_FILE") or path.with_name("providers.json")
        ).expanduser().resolve()
        qa_model = os.environ.get("KNOWLEDGE_QA_MODEL", "grounded-extractive-v1")
        configured_models = os.environ.get("KNOWLEDGE_QA_MODELS", "")
        qa_models = tuple(
            dict.fromkeys(
                model.strip()
                for model in configured_models.split(",")
                if model.strip()
            )
        )
        return cls(
            database_path=path,
            embedding_provider=os.environ.get("KNOWLEDGE_EMBEDDING_PROVIDER", "hash"),
            embedding_model=os.environ.get("KNOWLEDGE_EMBEDDING_MODEL", "hash-384-v1"),
            embedding_dimensions=int(os.environ.get("KNOWLEDGE_EMBEDDING_DIMENSIONS", "384")),
            embedding_base_url=os.environ.get("KNOWLEDGE_EMBEDDING_BASE_URL"),
            embedding_api_key=os.environ.get("KNOWLEDGE_EMBEDDING_API_KEY"),
            embedding_cache_dir=Path(
                os.environ.get("KNOWLEDGE_EMBEDDING_CACHE_DIR") or path.parent / "cache" / "models"
            ).expanduser().resolve(),
            rerank_provider=os.environ.get("KNOWLEDGE_RERANK_PROVIDER", "local"),
            rerank_model=os.environ.get("KNOWLEDGE_RERANK_MODEL"),
            rerank_base_url=os.environ.get("KNOWLEDGE_RERANK_BASE_URL"),
            rerank_api_key=os.environ.get("KNOWLEDGE_RERANK_API_KEY"),
            qa_provider=os.environ.get("KNOWLEDGE_QA_PROVIDER", "extractive"),
            qa_model=qa_model,
            qa_models=qa_models,
            qa_base_url=os.environ.get("KNOWLEDGE_QA_BASE_URL"),
            qa_api_key=os.environ.get("KNOWLEDGE_QA_API_KEY"),
            qa_compatible_providers=_load_qa_providers(providers_path),
            qa_providers_file=providers_path,
            obsidian_vault_root=Path(vault).expanduser().resolve() if vault else None,
        )
