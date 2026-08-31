from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .. import __version__
from ..answer_models import available_answer_models
from ..config import AppConfig, KEYRING_SERVICE
from ..ingestion import CancellationToken, IngestionService, IngestionSummary, ProgressEvent
from ..chunking import MediaAwareChunker
from ..models import ContentSegment, KnowledgeDocument, SearchResult, SourceReference, utcnow_iso
from ..product import (
    DEFAULT_ANSWER_MODEL,
    LEGACY_DEFAULT_ANSWER_MODELS,
    DesktopSettings,
    ProductPaths,
    PRODUCT_NAME,
)
from ..qa.engine import KnowledgeQAEngine
from ..qa.models import ImageAttachment, KnowledgeAnswer
from ..retrieval import KnowledgeRetriever
from ..runtime import build_answer_provider, build_embedding_provider, build_rerank_provider
from ..storage import (
    ConversationRepository,
    IngestionJobRepository,
    KnowledgeDatabase,
    KnowledgeGovernanceRepository,
    KnowledgeOperationsRepository,
    SQLiteVectorStore,
)
from ..transcripts import DeepCorrectionRepository, TranscriptRepository
from ..transcripts.workflow import DeepCorrectionWorkflow
from ..sync import ObsidianMarkdownSync, scan_folder
from ..indexing import IndexingService
from ..evaluation import GoldenEvaluator, load_golden_dataset
from ..wiki import PortableWikiCompiler
from .backup import create_backup as create_product_backup
from .backup import restore_backup as restore_product_backup
from .model_manager import LocalModelManager
from .privacy import (
    PrivacyViolationError,
    ShareCopyOptions,
    create_share_copy,
    scan_privacy,
)
from .update import check_for_update, download_verified_update


DEFAULT_PROVIDERS = {
    "deepseek": {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-v4-flash-vision-exp", "deepseek-v4-flash", "deepseek-v4-pro"],
        "temperature": 0.1,
    },
    "kimi": {
        "id": "kimi",
        "label": "Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["kimi-k3", "kimi-k2.6", "kimi-k2.7-code-highspeed", "kimi-k2.7-code"],
        "temperature": None,
    },
}


def _atomic_json(path: Path, payload: object, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        if private:
            try:
                os.fchmod(descriptor, 0o600)
            except OSError:
                pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if private:
            try:
                path.chmod(0o600)
            except OSError:
                pass
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, content: str) -> None:
    """Write a UTF-8 note without leaving a partially written knowledge file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            if content and not content.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file durably without exposing a partial destination."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    published = False
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
        published = True
        shutil.copystat(source, destination, follow_symlinks=False)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        if published:
            destination.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProviderConfigStore:
    """Credential store whose public status methods never expose API keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._migrate_plaintext_secrets()
        self._merge_provider_defaults()

    def _reference(self, provider_id: str) -> str:
        namespace = hashlib.sha256(str(self.path).encode("utf-8")).hexdigest()[:12]
        return f"provider:{provider_id}:{namespace}"

    @staticmethod
    def _get_secret(reference: str) -> str:
        try:
            import keyring
            return str(keyring.get_password(KEYRING_SERVICE, reference) or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _set_secret(reference: str, value: str) -> bool:
        try:
            import keyring
            keyring.set_password(KEYRING_SERVICE, reference, value)
            return True
        except Exception:
            return False

    def _migrate_plaintext_secrets(self) -> None:
        payload = self._load()
        changed = False
        for value in payload.get("providers", []):
            if not isinstance(value, dict):
                continue
            provider_id = str(value.get("id") or "").strip()
            secret = str(value.get("api_key") or "").strip()
            if not provider_id or not secret:
                continue
            reference = self._reference(provider_id)
            if self._set_secret(reference, secret):
                value.pop("api_key", None)
                value["credential_ref"] = reference
                changed = True
        if changed:
            _atomic_json(self.path, payload, private=True)

    def _load(self) -> dict[str, object]:
        if not self.path.is_file():
            return {"providers": []}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {"providers": []}
        return value if isinstance(value, dict) and isinstance(value.get("providers"), list) else {"providers": []}

    def _merge_provider_defaults(self) -> None:
        """Add newly supported built-in models without discarding custom model entries."""

        payload = self._load()
        providers = payload.get("providers", [])
        if not isinstance(providers, list):
            return
        changed = False
        for value in providers:
            if not isinstance(value, dict):
                continue
            defaults = DEFAULT_PROVIDERS.get(str(value.get("id") or ""))
            if defaults is None:
                continue
            raw_models = value.get("models", [])
            current = (
                [str(model).strip() for model in raw_models if str(model).strip()]
                if isinstance(raw_models, list)
                else []
            )
            merged = list(dict.fromkeys([*defaults["models"], *current]))
            if merged != current:
                value["models"] = merged
                changed = True
        if changed:
            _atomic_json(self.path, payload, private=True)

    def status(self) -> list[dict[str, object]]:
        current = {
            str(item.get("id")): item
            for item in self._load().get("providers", [])
            if isinstance(item, dict)
        }
        return [
            {
                "id": provider_id,
                "label": defaults["label"],
                "configured": bool(
                    str(current.get(provider_id, {}).get("api_key") or "").strip()
                    or self._get_secret(str(current.get(provider_id, {}).get("credential_ref") or ""))
                ),
                "base_url": str(current.get(provider_id, {}).get("base_url") or defaults["base_url"]),
                "models": list(current.get(provider_id, {}).get("models") or defaults["models"]),
            }
            for provider_id, defaults in DEFAULT_PROVIDERS.items()
        ]

    def update(
        self,
        provider_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        models: list[str] | None = None,
    ) -> None:
        if provider_id not in DEFAULT_PROVIDERS:
            raise ValueError("暂不支持该模型服务商")
        payload = self._load()
        existing = {
            str(item.get("id")): dict(item)
            for item in payload.get("providers", [])
            if isinstance(item, dict) and item.get("id")
        }
        defaults = dict(DEFAULT_PROVIDERS[provider_id])
        value = {**defaults, **existing.get(provider_id, {})}
        if api_key is not None and api_key.strip():
            reference = self._reference(provider_id)
            if self._set_secret(reference, api_key.strip()):
                value.pop("api_key", None)
                value["credential_ref"] = reference
            else:
                value["api_key"] = api_key.strip()
        if base_url is not None and base_url.strip():
            value["base_url"] = base_url.strip().rstrip("/")
        if models:
            value["models"] = list(dict.fromkeys(model.strip() for model in models if model.strip()))
        existing[provider_id] = value
        ordered = [existing[key] for key in DEFAULT_PROVIDERS if key in existing]
        ordered.extend(value for key, value in existing.items() if key not in DEFAULT_PROVIDERS)
        _atomic_json(self.path, {"providers": ordered}, private=True)


class DesktopController:
    """Thread-safe facade: each operation owns its SQLite connection."""

    def __init__(self, data_root: str | Path | None = None, *, migrate_legacy: bool = True) -> None:
        self.paths = ProductPaths.resolve(data_root)
        self.migrated_product_root: Path | None = None
        if migrate_legacy and data_root is None:
            self.migrated_product_root = self.paths.migrate_renamed_product()
        self.paths.ensure()
        self.migrated_database: Path | None = None
        if migrate_legacy:
            self.paths.migrate_legacy_providers()
            self.migrated_database = self.paths.migrate_legacy_database()
        self.settings = DesktopSettings.load(self.paths.settings)
        self.providers = ProviderConfigStore(self.paths.providers)
        self.local_models = LocalModelManager(self.paths)
        settings_changed = False
        for path_field, checksum_field in (
            ("asr_model_path", "asr_model_sha256"),
            (
                "asr_whisper_fallback_model_path",
                "asr_whisper_fallback_model_sha256",
            ),
            ("diarization_model_path", "diarization_model_sha256"),
        ):
            checksum = self.local_models.verified_content_sha256_for_path(
                getattr(self.settings, path_field)
            )
            if checksum and checksum != getattr(self.settings, checksum_field):
                setattr(self.settings, checksum_field, checksum)
                settings_changed = True
        if self.settings.default_model in LEGACY_DEFAULT_ANSWER_MODELS:
            self.settings.default_model = DEFAULT_ANSWER_MODEL
            settings_changed = True
        if settings_changed:
            self.settings.save(self.paths.settings)
        with KnowledgeDatabase(self.paths.database) as database:
            self.recovered_ingestion_jobs = IngestionJobRepository(
                database
            ).recover_interrupted_jobs()

    def reload(self) -> None:
        self.settings = DesktopSettings.load(self.paths.settings)

    def config(self) -> AppConfig:
        config = AppConfig.from_env(self.paths.database)
        config.embedding_provider = self.settings.embedding_provider
        config.embedding_model = self.settings.embedding_model
        config.embedding_dimensions = 384
        config.embedding_cache_dir = self.paths.cache / "models"
        selected = self.settings.default_model.split("::", 2)
        if len(selected) == 3 and selected[0] == "compatible":
            config.qa_provider = selected[1]
            config.qa_model = selected[2]
        elif self.settings.default_model == "local-extractive":
            config.qa_provider = "extractive"
            config.qa_model = "grounded-extractive-v1"
        config.obsidian_vault_root = (
            Path(self.settings.obsidian_vault).expanduser().resolve()
            if self.settings.obsidian_vault
            else None
        )
        return config

    def save_settings(self, settings: DesktopSettings) -> None:
        settings.save(self.paths.settings)
        self.settings = settings

    def model_choices(self) -> list[dict[str, object]]:
        return [item.to_dict() for item in available_answer_models(self.config(), codex_available=False)]

    def transcription_model_statuses(self) -> list[dict[str, object]]:
        """Return local-only model state without downloading any weights."""

        return [item.to_dict() for item in self.local_models.statuses()]

    def resolve_transcription_model(self, model_id: str) -> str | None:
        path = self.local_models.resolve(model_id)
        return str(path) if path else None

    def status(self) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            result = database.status()
            result["embedding_profiles"] = database.embedding_profile()
        result.update(
            {
                "product": PRODUCT_NAME,
                "data_root": str(self.paths.root),
                "providers": self.providers.status(),
                "obsidian_configured": bool(self.settings.obsidian_vault),
            }
        )
        return result

    def documents(self, *, limit: int = 500) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            rows = database.connection.execute(
                """SELECT d.*, COUNT(c.id) AS chunk_count
                   FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
                   GROUP BY d.id ORDER BY d.updated_at DESC LIMIT ?""",
                (max(1, min(2000, limit)),),
            ).fetchall()
            values = []
            for row in rows:
                metadata = json.loads(row["metadata_json"] or "{}")
                facets = database.document_facets(str(row["id"]))
                values.append({
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "media_type": row["media_type"],
                    "local_path": row["local_path"],
                    "original_uri": row["original_uri"],
                    "updated_at": row["updated_at"],
                    "enabled": bool(row["enabled"]),
                    "chunks": row["chunk_count"],
                    "metadata": metadata,
                    "collections": facets["collections"],
                    "tags": facets["tags"],
                    "quality": metadata.get("quality_report") or {},
                })
            return values

    def knowledge_items(
        self,
        query: str = "",
        *,
        item_type: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, object]]:
        """Return governed knowledge, separate from raw source documents."""

        item_types = (item_type,) if item_type else ()
        statuses = (status,) if status else ()
        with KnowledgeDatabase(self.paths.database) as database:
            repository = KnowledgeGovernanceRepository(database)
            if query.strip():
                values = repository.search(
                    query,
                    item_types=item_types,
                    statuses=statuses,
                    limit=limit,
                )
            else:
                values = repository.list_items(
                    item_types=item_types,
                    statuses=statuses,
                    limit=limit,
                )
        return [item.to_dict() for item in values]

    def knowledge_item(self, item_id: str) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            repository = KnowledgeGovernanceRepository(database)
            item = repository.get_item(item_id)
            if item is None:
                raise ValueError("知识条目不存在或已删除")
            result = item.to_dict()
            relations: list[dict[str, object]] = []
            for related in repository.related_items(item_id):
                relation = related.relation.to_dict()
                relations.append(
                    {
                        **relation,
                        "direction": related.direction,
                        "related_id": related.item.item_id,
                        "related_title": related.item.title,
                        "related_type": related.item.item_type,
                    }
                )
            result["relations"] = relations
            return result

    def compile_portable_wiki(self) -> dict[str, object]:
        """Rebuild the portable Markdown mirror from governed SQLite facts."""

        with KnowledgeDatabase(self.paths.database) as database:
            result = PortableWikiCompiler(
                database, self.paths.notes / "LLM-Wiki"
            ).compile()
        return result.to_dict()

    def knowledge_proposals(
        self, *, include_reviewed: bool = False, limit: int = 500
    ) -> list[dict[str, object]]:
        statuses = () if include_reviewed else ("proposed",)
        with KnowledgeDatabase(self.paths.database) as database:
            values = KnowledgeOperationsRepository(database).list_proposals(
                statuses=statuses, limit=limit
            )
        return [value.to_dict() for value in values]

    def review_knowledge_proposal(
        self,
        proposal_id: str,
        *,
        decision: str,
        merge_duplicate: bool = False,
        reason: str = "人工复核",
    ) -> dict[str, object]:
        clean = str(decision or "").strip().casefold()
        with KnowledgeDatabase(self.paths.database) as database:
            operations = KnowledgeOperationsRepository(database)
            proposal = operations.get_proposal(proposal_id)
            if clean == "reject":
                reviewed = operations.reject_proposal(proposal_id, reason=reason)
                result: dict[str, object] = {
                    "proposal": reviewed.to_dict(), "item": None
                }
            elif clean == "accept":
                item = operations.accept_proposal(
                    proposal_id, merge_duplicate=merge_duplicate
                )
                repository = KnowledgeGovernanceRepository(database)
                metadata = dict(item.metadata)
                if not metadata.get("note_relative_path"):
                    relative = (
                        Path("正式知识")
                        / item.item_type
                        / self._safe_note_name(item.title, item.item_id)
                    )
                    metadata["note_relative_path"] = relative.as_posix()
                metadata["managed_from"] = metadata.get(
                    "managed_from", "knowledge-proposal"
                )
                evidence_sources: list[dict[str, object]] = []
                if proposal.source_document_id:
                    document = database.get_document(proposal.source_document_id)
                    if document is not None:
                        evidence_sources.append(
                            {
                                "document_id": proposal.source_document_id,
                                "title": str(document["title"]),
                                "media_type": str(document["media_type"]),
                            }
                        )
                if evidence_sources:
                    existing = metadata.get("evidence_sources")
                    metadata["evidence_sources"] = list(existing) if isinstance(existing, list) else []
                    metadata["evidence_sources"].extend(evidence_sources)
                item = repository.update_item(item.item_id, metadata=metadata)
                value = item.to_dict()
                note = self._governed_note_path(metadata)
                if note is not None:
                    _atomic_text(note, self._governed_note_markdown(value))
                result = {
                    "proposal": operations.get_proposal(proposal_id).to_dict(),
                    "item": value,
                }
            else:
                raise ValueError("decision 只能是 accept 或 reject")
            result["wiki"] = PortableWikiCompiler(
                database, self.paths.notes / "LLM-Wiki"
            ).compile().to_dict()
            return result

    def knowledge_space_policy(self, policy_id: str | None = None) -> dict[str, object]:
        clean_id = (policy_id or self.settings.asr_knowledge_space_id).strip()
        with KnowledgeDatabase(self.paths.database) as database:
            return KnowledgeOperationsRepository(database).get_policy(clean_id).to_dict()

    def save_knowledge_space_policy(
        self,
        *,
        policy_id: str,
        name: str,
        policy: dict[str, object],
    ) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            value = KnowledgeOperationsRepository(database).upsert_policy(
                policy_id, name, policy
            )
        return value.to_dict()

    def source_assessment(self, document_id: str) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            return KnowledgeOperationsRepository(database).get_source_assessment(
                document_id
            ).to_dict()

    def save_source_assessment(
        self, document_id: str, **changes: object
    ) -> dict[str, object]:
        allowed = {
            "source_class", "reliability", "extraction_completeness",
            "published_at", "valid_until", "notes", "checked", "metadata",
        }
        invalid = set(changes) - allowed
        if invalid:
            raise ValueError(f"不支持的来源评估字段：{', '.join(sorted(invalid))}")
        with KnowledgeDatabase(self.paths.database) as database:
            value = KnowledgeOperationsRepository(database).upsert_source_assessment(
                document_id, **changes
            )
        return value.to_dict()

    def source_quality_center(self) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            operations = KnowledgeOperationsRepository(database)
            issues = operations.source_quality_issues()
            governance = KnowledgeGovernanceRepository(database)
            contradictions = []
            for relation in governance.list_relations(
                relation_types=("contradicts",), limit=1000
            ):
                source = governance.get_item(relation.source_item_id)
                target = governance.get_item(relation.target_item_id)
                contradictions.append({
                    **relation.to_dict(),
                    "source_title": source.title if source is not None else relation.source_item_id,
                    "target_title": target.title if target is not None else relation.target_item_id,
                })
            duplicates = database.duplicate_groups()
        return {
            "issues": issues,
            "contradictions": contradictions,
            "duplicates": duplicates,
            "counts": {
                "issues": len(issues),
                "contradictions": len(contradictions),
                "duplicate_groups": len(duplicates),
            },
        }

    def workflows(self, *, include_archived: bool = False) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            values = KnowledgeOperationsRepository(database).list_workflows(
                include_archived=include_archived
            )
        return [value.to_dict() for value in values]

    def save_workflow(self, **value: object) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            workflow = KnowledgeOperationsRepository(database).upsert_workflow(**value)
        return workflow.to_dict()

    def run_golden_evaluation(
        self,
        dataset_path: str | Path,
        *,
        top_k: int = 10,
        evaluate_citations: bool = False,
        model_id: str | None = None,
    ) -> dict[str, object]:
        dataset = load_golden_dataset(dataset_path)
        config = self.config()
        embedding = build_embedding_provider(config)
        with KnowledgeDatabase(self.paths.database) as database:
            IndexingService(database, embedding).ensure_embedding_profile()
            retriever = KnowledgeRetriever(
                database, embedding,
                rerank_provider=build_rerank_provider(config),
            )
            engine = None
            if evaluate_citations:
                engine = KnowledgeQAEngine(
                    database,
                    retriever,
                    answer_provider=build_answer_provider(
                        config,
                        model_id=model_id or self.settings.default_model,
                    ),
                )
            return GoldenEvaluator(retriever, qa_engine=engine).evaluate(
                dataset, top_k=top_k, evaluate_citations=evaluate_citations
            )

    def update_knowledge_item(self, item_id: str, **changes: object) -> dict[str, object]:
        allowed = {
            "item_type",
            "title",
            "status",
            "maturity",
            "summary",
            "body",
            "high_value",
            "aliases",
            "tags",
            "metadata",
        }
        invalid = sorted(set(changes) - allowed)
        if invalid:
            raise ValueError(f"不支持更新字段：{', '.join(invalid)}")
        with KnowledgeDatabase(self.paths.database) as database:
            repository = KnowledgeGovernanceRepository(database)
            item = repository.update_item(item_id, **changes)
            value = item.to_dict()
        self._refresh_governed_note(value)
        return value

    def delete_knowledge_item(self, item_id: str) -> bool:
        """Move a governed item to the recoverable product trash.

        Filesystem evidence is copied and the tombstone is durably written before
        SQLite is touched.  Any later failure restores the database item and leaves
        the original note intact.
        """

        with KnowledgeDatabase(self.paths.database) as database:
            repository = KnowledgeGovernanceRepository(database)
            item = repository.get_item(item_id)
            if item is None:
                return False
            snapshot = repository.snapshot_item(item.item_id)
            note = self._governed_note_path(item.metadata)
        deleted_at = utcnow_iso()
        tombstone_id = (
            f"{datetime.now().astimezone():%Y%m%d-%H%M%S-%f}--{uuid.uuid4().hex[:10]}"
        )
        entry = self._knowledge_trash_root() / tombstone_id
        stored_note = entry / "note.md"
        note_payload: dict[str, object] = {
            "present": False,
            "original_relative_path": None,
            "stored_name": None,
            "sha256": None,
        }
        try:
            entry.mkdir(parents=True, exist_ok=False)
            if note is not None and note.is_file():
                original_relative = note.relative_to(self.paths.notes.resolve()).as_posix()
                _atomic_copy(note, stored_note)
                note_payload = {
                    "present": True,
                    "original_relative_path": original_relative,
                    "stored_name": stored_note.name,
                    "sha256": _sha256_file(stored_note),
                }
            tombstone = {
                "format": "ai-jingjing-knowledge-tombstone",
                "version": 1,
                "tombstone_id": tombstone_id,
                "deleted_at": deleted_at,
                **snapshot,
                "note": note_payload,
            }
            _atomic_json(entry / "tombstone.json", tombstone, private=True)
        except Exception:
            self._remove_knowledge_trash_entry(entry)
            raise

        deleted = False
        try:
            with KnowledgeDatabase(self.paths.database) as database:
                deleted = KnowledgeGovernanceRepository(database).delete_item(item.item_id)
            if not deleted:
                self._remove_knowledge_trash_entry(entry)
                return False
            if note is not None and note.is_file():
                note.unlink()
        except Exception:
            if deleted:
                try:
                    with KnowledgeDatabase(self.paths.database) as database:
                        repository = KnowledgeGovernanceRepository(database)
                        if repository.get_item(item.item_id) is None:
                            repository.restore_item_snapshot(snapshot)
                except Exception as rollback_exc:
                    raise RuntimeError(
                        "删除知识失败，自动回滚也未完成；回收站 tombstone 已保留"
                    ) from rollback_exc
            self._remove_knowledge_trash_entry(entry)
            raise
        return True

    def knowledge_trash_items(self) -> list[dict[str, object]]:
        """List valid recoverable tombstones without exposing internal file paths."""

        values: list[dict[str, object]] = []
        root = self._knowledge_trash_root()
        with KnowledgeDatabase(self.paths.database) as database:
            existing_ids = {
                str(row["id"])
                for row in database.connection.execute(
                    "SELECT id FROM knowledge_items"
                ).fetchall()
            }
        for tombstone_path in root.glob("*/tombstone.json"):
            try:
                payload = self._read_knowledge_tombstone(tombstone_path.parent.name)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            item = payload.get("item")
            note = payload.get("note")
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or item.get("item_id") or "")
            if item_id in existing_ids:
                # A previous restore committed successfully but filesystem cleanup may
                # have been interrupted.  Hide the zombie entry and retry exact cleanup.
                try:
                    self._remove_knowledge_trash_entry(tombstone_path.parent)
                except OSError:
                    pass
                continue
            values.append(
                {
                    "tombstone_id": str(payload["tombstone_id"]),
                    "item_id": item_id,
                    "title": str(item.get("title") or "未命名知识"),
                    "item_type": str(item.get("item_type") or "analysis"),
                    "status": str(item.get("status") or "draft"),
                    "deleted_at": str(payload.get("deleted_at") or ""),
                    "has_note": bool(isinstance(note, dict) and note.get("present")),
                    "relation_count": len(payload.get("relations") or []),
                }
            )
        values.sort(
            key=lambda value: (str(value.get("deleted_at") or ""), str(value["tombstone_id"])),
            reverse=True,
        )
        return values

    def restore_knowledge_item(self, tombstone_id: str) -> dict[str, object]:
        """Restore the same item ID, its note and every still-valid graph edge."""

        payload = self._read_knowledge_tombstone(tombstone_id)
        raw_item = payload.get("item")
        if not isinstance(raw_item, dict):
            raise ValueError("回收站记录缺少知识条目")
        item_id = str(raw_item.get("id") or raw_item.get("item_id") or "").strip()
        if not item_id:
            raise ValueError("回收站记录缺少知识 ID")
        with KnowledgeDatabase(self.paths.database) as database:
            if KnowledgeGovernanceRepository(database).get_item(item_id) is not None:
                raise ValueError("该知识已经恢复，不能重复恢复")

        entry = self._knowledge_trash_entry(tombstone_id)
        note_payload = payload.get("note")
        restored_note: Path | None = None
        if isinstance(note_payload, dict) and bool(note_payload.get("present")):
            stored_name = str(note_payload.get("stored_name") or "").strip()
            if not stored_name or Path(stored_name).name != stored_name:
                raise ValueError("回收站笔记路径无效")
            stored_note = (entry / stored_name).resolve()
            try:
                stored_note.relative_to(entry.resolve())
            except ValueError as exc:
                raise ValueError("回收站笔记越过安全目录") from exc
            if not stored_note.is_file():
                raise ValueError("回收站中的 Markdown 笔记已丢失")
            expected_checksum = str(note_payload.get("sha256") or "").strip()
            if expected_checksum and _sha256_file(stored_note) != expected_checksum:
                raise ValueError("回收站中的 Markdown 笔记校验失败")
            original_relative = str(
                note_payload.get("original_relative_path") or ""
            ).strip()
            original_target = self._safe_note_relative_target(original_relative)
            restored_note = self._available_restored_note_path(original_target, item_id)
            _atomic_copy(stored_note, restored_note)
            metadata = raw_item.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            else:
                metadata = dict(metadata)
            metadata["note_relative_path"] = restored_note.relative_to(
                self.paths.notes.resolve()
            ).as_posix()
            raw_item = dict(raw_item)
            raw_item["metadata"] = metadata
            payload = dict(payload)
            payload["item"] = raw_item

        try:
            with KnowledgeDatabase(self.paths.database) as database:
                result = KnowledgeGovernanceRepository(database).restore_item_snapshot(payload)
        except Exception:
            if restored_note is not None:
                restored_note.unlink(missing_ok=True)
            raise
        cleanup_pending = False
        try:
            self._remove_knowledge_trash_entry(entry)
        except OSError:
            cleanup_pending = True
        value = result.item.to_dict()
        value.update(
            {
                "restored_relation_count": len(result.restored_relations),
                "skipped_relation_ids": list(result.skipped_relation_ids),
                "trash_cleanup_pending": cleanup_pending,
            }
        )
        return value

    def _knowledge_trash_root(self) -> Path:
        root = (self.paths.trash / "knowledge-items").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _knowledge_trash_entry(self, tombstone_id: str) -> Path:
        clean_id = str(tombstone_id or "").strip()
        if not clean_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in clean_id):
            raise ValueError("回收站记录 ID 无效")
        candidate = (self._knowledge_trash_root() / clean_id).resolve()
        try:
            candidate.relative_to(self._knowledge_trash_root())
        except ValueError as exc:
            raise ValueError("回收站记录越过安全目录") from exc
        return candidate

    def _read_knowledge_tombstone(self, tombstone_id: str) -> dict[str, object]:
        entry = self._knowledge_trash_entry(tombstone_id)
        path = entry / "tombstone.json"
        if not path.is_file():
            raise ValueError("回收站记录不存在")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("回收站记录格式无效")
        if value.get("format") != "ai-jingjing-knowledge-tombstone" or value.get("version") != 1:
            raise ValueError("不支持的回收站记录版本")
        if str(value.get("tombstone_id") or "") != entry.name:
            raise ValueError("回收站记录 ID 不匹配")
        return value

    def _safe_note_relative_target(self, relative_path: str) -> Path:
        if not relative_path:
            raise ValueError("回收站记录缺少原始笔记路径")
        candidate = (self.paths.notes / relative_path).resolve()
        try:
            candidate.relative_to(self.paths.notes.resolve())
        except ValueError as exc:
            raise ValueError("原始笔记路径越过安全目录") from exc
        if candidate.suffix.casefold() != ".md":
            raise ValueError("只允许恢复 Markdown 知识笔记")
        return candidate

    @staticmethod
    def _available_restored_note_path(original: Path, item_id: str) -> Path:
        if not original.exists():
            return original
        suffix = original.suffix or ".md"
        stem = original.stem
        marker = item_id[-8:] if item_id else uuid.uuid4().hex[:8]
        for index in range(1, 10_000):
            counter = "" if index == 1 else f"-{index}"
            candidate = original.with_name(f"{stem}--restored-{marker}{counter}{suffix}")
            if not candidate.exists():
                return candidate
        raise OSError("无法为恢复的知识笔记分配安全文件名")

    def _remove_knowledge_trash_entry(self, entry: Path) -> None:
        candidate = entry.resolve()
        root = self._knowledge_trash_root()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("拒绝清理回收站目录之外的路径") from exc
        if candidate == root:
            raise ValueError("拒绝清理整个知识回收站")
        if candidate.exists():
            shutil.rmtree(candidate)

    def create_knowledge_relation(
        self,
        source_item_id: str,
        target_item_id: str,
        relation_type: str,
        *,
        summary: str = "",
    ) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            relation = KnowledgeGovernanceRepository(database).create_relation(
                source_item_id,
                target_item_id,
                relation_type,
                summary=summary,
            )
        return relation.to_dict()

    @staticmethod
    def _knowledge_issue_suggestion(code: str) -> str:
        return {
            "missing_metadata": "补充来源、适用范围或生成方式等元数据",
            "missing_summary": "为该知识补充一至两句可检索摘要",
            "missing_body": "补充结论、边界和依据，或标记为低价值",
            "source_without_evidence": "重新导入或恢复该来源的归档文件",
            "orphan_item": "建立 supports、extends 或 opens 关系",
            "isolated_source": "将高价值内容沉淀为主题、分析或决策",
            "stale_current": "复核内容；仍有效则确认更新，否则标记过期",
            "marked_stale": "用最新证据更新，或归档保留历史版本",
            "high_value_uncompiled": "对来源进行知识编译并保留直接证据关系",
            "compiled_without_source": "关联至少一条 source 类证据",
            "noncanonical_tag": "将标签统一为小写 kebab-case，减少重复",
            "ambiguous_alias": "为冲突条目使用更具体的别名",
        }.get(code, "打开该知识条目并根据提醒复核")

    def knowledge_health(self, *, stale_after_days: int = 120) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            repository = KnowledgeGovernanceRepository(database)
            operations = KnowledgeOperationsRepository(database)
            raw = repository.health_report(stale_after_days=stale_after_days).to_dict()
            item_map = {
                item.item_id: item
                for item in repository.list_items(limit=1000)
            }
            source_issues = operations.source_quality_issues()
            proposal_count = len(operations.list_proposals(statuses=("proposed",), limit=2000))
        nested_counts = raw.get("counts") if isinstance(raw.get("counts"), dict) else {}
        by_status = (
            nested_counts.get("by_status")
            if isinstance(nested_counts.get("by_status"), dict)
            else {}
        )
        by_type = (
            nested_counts.get("by_type")
            if isinstance(nested_counts.get("by_type"), dict)
            else {}
        )
        normalized_issues: list[dict[str, object]] = []
        for raw_issue in raw.get("issues") or []:
            if not isinstance(raw_issue, dict):
                continue
            code = str(raw_issue.get("code") or "governance")
            item_id = str(raw_issue.get("item_id") or "")
            item = item_map.get(item_id)
            normalized_issues.append(
                {
                    **raw_issue,
                    "category": code,
                    "item_title": item.title if item is not None else raw_issue.get("title"),
                    "item_type": item.item_type if item is not None else "",
                    "suggestion": self._knowledge_issue_suggestion(code),
                }
            )
        for issue in source_issues:
            normalized_issues.append(
                {
                    **issue,
                    "category": issue.get("code", "source_quality"),
                    "item_id": None,
                    "item_title": issue.get("title"),
                    "item_type": "source",
                    "suggestion": "打开来源质量中心，补充来源类型、可靠性和有效期",
                }
            )
        return {
            **raw,
            "counts": {
                "items": int(raw.get("total_items") or 0),
                "needs_review": int(by_status.get("needs-review", 0)),
                "stale": int(by_status.get("stale", 0)),
                "current": int(by_status.get("current", 0)),
                "proposals": proposal_count,
                "source_quality_issues": len(source_issues),
                "by_status": by_status,
                "by_type": by_type,
            },
            "issue_count": len(normalized_issues),
            "issues": normalized_issues,
        }

    def _governed_note_path(self, metadata: dict[str, object]) -> Path | None:
        raw = str(metadata.get("note_relative_path") or "").strip()
        if not raw:
            return None
        candidate = (self.paths.notes / raw).resolve()
        try:
            candidate.relative_to(self.paths.notes.resolve())
        except ValueError:
            return None
        return candidate

    @staticmethod
    def _safe_note_name(title: str, item_id: str) -> str:
        safe = "".join(
            "-" if char in '/:*?\"<>|\\' else char for char in title
        ).strip(" .-")[:90]
        return f"{safe or '未命名知识'}--{item_id[-8:]}.md"

    @staticmethod
    def _governed_note_markdown(item: dict[str, object]) -> str:
        aliases = [str(value) for value in item.get("aliases") or []]
        tags = [str(value) for value in item.get("tags") or []]
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        raw_sources = metadata.get("evidence_sources") if isinstance(metadata, dict) else []
        sources = [value for value in raw_sources or [] if isinstance(value, dict)]
        lines = [
            "---",
            f"knowledge_id: {json.dumps(str(item.get('id') or ''), ensure_ascii=False)}",
            f"knowledge_type: {json.dumps(str(item.get('item_type') or ''), ensure_ascii=False)}",
            f"status: {json.dumps(str(item.get('status') or ''), ensure_ascii=False)}",
            f"maturity: {json.dumps(str(item.get('maturity') or ''), ensure_ascii=False)}",
            f"aliases: {json.dumps(aliases, ensure_ascii=False)}",
            f"tags: {json.dumps(tags, ensure_ascii=False)}",
            f"updated_at: {json.dumps(str(item.get('updated_at') or utcnow_iso()))}",
            "---",
            "",
            f"# {str(item.get('title') or '未命名知识')}",
            "",
            "## 摘要",
            "",
            str(item.get("summary") or "尚未填写摘要。"),
            "",
            "## 正文",
            "",
            str(item.get("body") or ""),
            "",
            "## 来源证据",
            "",
        ]
        if sources:
            for source in sources:
                source_title = str(source.get("title") or "未命名来源").replace("\n", " ").strip()
                media_type = str(source.get("media_type") or "source")
                document_id = str(source.get("document_id") or "")
                lines.append(f"- **{source_title}** · `{media_type}` · `{document_id}`")
        else:
            lines.append("- 尚未关联原始资料；请在确认为“当前有效”前补充来源。")
        lines.append("")
        return "\n".join(lines)

    def _refresh_governed_note(self, item: dict[str, object]) -> None:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        note = self._governed_note_path(metadata)
        if note is not None:
            _atomic_text(note, self._governed_note_markdown(item))

    def capture_answer_as_knowledge(
        self,
        *,
        markdown: str,
        question: str,
        title: str,
        item_type: str = "analysis",
        status: str = "needs-review",
        summary: str = "",
        aliases: Iterable[str] = (),
        tags: Iterable[str] = (),
        conversation_id: str | None = None,
        answer_id: str | None = None,
        evidence_document_ids: Iterable[str] = (),
    ) -> dict[str, object]:
        body = markdown.strip()
        if not body:
            raise ValueError("没有可沉淀的回答内容")
        if item_type not in {"topic", "entity", "analysis", "decision", "output"}:
            raise ValueError("回答只能沉淀为主题、实体、分析、决策或成果")
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("知识标题不能为空")
        item_id = "kg:" + uuid.uuid4().hex
        relative_note = (
            Path("正式知识")
            / item_type
            / self._safe_note_name(clean_title, item_id)
        )
        note_path = self.paths.notes / relative_note
        document_ids = list(
            dict.fromkeys(str(value).strip() for value in evidence_document_ids if str(value).strip())
        )
        metadata: dict[str, object] = {
            "managed_from": "answer-capture",
            "note_relative_path": relative_note.as_posix(),
            "question": question.strip(),
            "conversation_id": conversation_id,
            "answer_id": answer_id,
            "evidence_document_ids": document_ids,
        }
        with KnowledgeDatabase(self.paths.database) as database:
            repository = KnowledgeGovernanceRepository(database)
            existing_documents: dict[str, object] = {}
            evidence_sources: list[dict[str, object]] = []
            for document_id in document_ids:
                document = database.get_document(document_id)
                if document is None:
                    continue
                existing_documents[document_id] = document
                evidence_sources.append(
                    {
                        "document_id": document_id,
                        "title": str(document["title"]),
                        "media_type": str(document["media_type"]),
                    }
                )
            metadata["evidence_sources"] = evidence_sources
            item = repository.create_item(
                item_id=item_id,
                item_type=item_type,
                title=clean_title,
                status=status,
                maturity="compiled",
                summary=summary.strip(),
                body=body,
                aliases=tuple(aliases),
                tags=tuple(tags),
                metadata=metadata,
            )
            try:
                for document_id in document_ids:
                    if document_id not in existing_documents:
                        continue
                    source = repository.get_item_for_document(document_id)
                    if source is not None:
                        repository.create_relation(
                            source.item_id,
                            item.item_id,
                            "supports",
                            summary="问答证据支持该知识",
                            metadata={"answer_id": answer_id},
                        )
                value = repository.get_item(item.item_id)
                if value is None:
                    raise RuntimeError("知识条目保存后未能读取")
                result = value.to_dict()
                _atomic_text(note_path, self._governed_note_markdown(result))
            except Exception:
                repository.delete_item(item.item_id)
                note_path.unlink(missing_ok=True)
                raise
        return result

    def ingest(
        self,
        items: Iterable[str | Path],
        *,
        progress: Callable[[ProgressEvent], None] | None = None,
        cancellation: CancellationToken | None = None,
        job_id: str | None = None,
    ) -> IngestionSummary:
        values = [str(item).strip() for item in items if str(item).strip()]
        if not values:
            raise ValueError("请至少选择一个文件或网页地址")
        with KnowledgeDatabase(self.paths.database) as database:
            jobs = IngestionJobRepository(database)
            if job_id is None:
                active_job_id = str(jobs.create_job(values)["id"])
            else:
                active_job_id = job_id
                jobs.job_record(active_job_id)
            jobs.begin_job(active_job_id)

        def tracked_progress(event: ProgressEvent) -> None:
            with KnowledgeDatabase(self.paths.database) as database:
                IngestionJobRepository(database).record_progress(
                    active_job_id, event.item, event.stage, event.percent, event.message
                )
            if progress:
                progress(event)

        try:
            service = IngestionService(self.paths, self.config(), self.settings)
            result = service.ingest(
                values, progress=tracked_progress, cancellation=cancellation
            )
            if self.settings.deep_correction_enabled:
                for item_result in result.results:
                    if (
                        not item_result.transcript_run_id
                        or item_result.status in {"failed", "cancelled"}
                    ):
                        continue
                    if cancellation is not None and cancellation.cancelled:
                        item_result.warnings.append(
                            "用户已取消后续深度精校；首轮转写和入库结果已安全保留"
                        )
                        continue

                    def correction_progress(
                        *values: object,
                        source_item: str = item_result.item,
                    ) -> None:
                        if len(values) == 1 and isinstance(values[0], dict):
                            payload = values[0]
                            stage = str(payload.get("stage") or "validation")
                            completed = int(payload.get("completed") or 0)
                            total = int(payload.get("total") or 11)
                            message = str(payload.get("message") or "")
                        else:
                            stage = str(values[0] if len(values) > 0 else "semantic")
                            completed = int(values[1] if len(values) > 1 else 0)
                            total = int(values[2] if len(values) > 2 else 11)
                            message = str(values[3] if len(values) > 3 else "")
                        tracked_progress(ProgressEvent(
                            source_item,
                            f"deep_correction:{stage}",
                            100,
                            f"深度精校 {completed}/{total}：{message}",
                        ))

                    try:
                        corrected = self.deep_correct_transcript(
                            item_result.transcript_run_id,
                            progress=correction_progress,
                            cancellation=cancellation,
                        )
                    except Exception as correction_error:
                        item_result.warnings.append(
                            "资料已正常入库，但自动深度精校未完成："
                            + str(correction_error)[:240]
                        )
                    else:
                        item_result.deep_correction_run_id = str(
                            corrected.get("correction_run_id") or ""
                        ) or None
                        item_result.deep_correction_path = str(
                            corrected.get("output_path") or ""
                        ) or None
        except Exception as exc:
            with KnowledgeDatabase(self.paths.database) as database:
                IngestionJobRepository(database).fail_job(active_job_id, str(exc))
            raise
        result.job_id = active_job_id
        with KnowledgeDatabase(self.paths.database) as database:
            jobs = IngestionJobRepository(database)
            operations = KnowledgeOperationsRepository(database)
            policy = operations.get_policy(self.settings.asr_knowledge_space_id)
            default_reliability = str(
                policy.policy.get("default_source_reliability") or "unassessed"
            )
            for item in result.results:
                jobs.record_result(active_job_id, item.item, item.to_dict())
                if item.document_id and database.get_document(item.document_id) is not None:
                    assessment = operations.get_source_assessment(item.document_id)
                    if (
                        assessment.reliability == "unassessed"
                        and default_reliability != "unassessed"
                    ):
                        operations.upsert_source_assessment(
                            item.document_id,
                            source_class=assessment.source_class,
                            reliability=default_reliability,
                            extraction_completeness=assessment.extraction_completeness,
                            notes="由知识空间策略提供的默认值；尚未人工确认",
                            checked=False,
                            metadata={**assessment.metadata, "policy_default": True},
                        )
            jobs.finalize_job(active_job_id)
        recent = [str(item) for item in values]
        self.settings.recent_imports = list(dict.fromkeys([*recent, *self.settings.recent_imports]))[:20]
        self.settings.save(self.paths.settings)
        return result

    def search(
        self,
        query: str,
        *,
        top_k: int = 12,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        config = self.config()
        embedding = build_embedding_provider(config)
        with KnowledgeDatabase(self.paths.database) as database:
            IndexingService(database, embedding).ensure_embedding_profile()
            retriever = KnowledgeRetriever(
                database,
                embedding,
                rerank_provider=build_rerank_provider(config),
            )
            return retriever.search_knowledge(
                query,
                top_k=top_k,
                collections=collections,
                tags=tags,
                document_ids=document_ids,
            )

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        model_id: str | None = None,
        deep_analysis: bool = False,
        top_k: int = 10,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        document_ids: list[str] | None = None,
        progress: Callable[[str, str], None] | None = None,
        delta_callback: Callable[[str], None] | None = None,
        image_attachments: list[ImageAttachment] | None = None,
    ) -> KnowledgeAnswer:
        config = self.config()
        requested_model = model_id or self.settings.default_model
        model_ids = {str(item["id"]) for item in self.model_choices()}
        if requested_model not in model_ids:
            requested_model = "local-extractive"
        selected = next(
            (item for item in self.model_choices() if str(item["id"]) == requested_model),
            None,
        )
        if image_attachments and not bool(selected and selected.get("supports_images")):
            raise ValueError("当前模型不支持图片理解，请选择带“视觉”标识的 DeepSeek Vision 或 Kimi 模型。")
        embedding = build_embedding_provider(config)
        with KnowledgeDatabase(self.paths.database) as database:
            if progress:
                progress("embedding", "正在检查本地中文语义索引")
            IndexingService(database, embedding).ensure_embedding_profile()
            retriever = KnowledgeRetriever(
                database,
                embedding,
                rerank_provider=build_rerank_provider(config),
            )
            engine = KnowledgeQAEngine(
                database,
                retriever,
                answer_provider=build_answer_provider(
                    config,
                    model_id=requested_model,
                    deep_analysis=deep_analysis,
                ),
            )
            return engine.ask(
                question,
                conversation_id=conversation_id,
                top_k=top_k,
                collections=collections,
                tags=tags,
                document_ids=document_ids,
                response_language=self.settings.answer_language,
                progress_callback=progress,
                delta_callback=delta_callback,
                image_attachments=image_attachments,
            )

    def conversations(
        self, query: str = "", limit: int = 200, offset: int = 0
    ) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            return ConversationRepository(database).list_conversations(
                query, limit=limit, offset=offset
            )

    def conversation_record(self, conversation_id: str) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            return ConversationRepository(database).conversation_record(conversation_id)

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        with KnowledgeDatabase(self.paths.database) as database:
            return ConversationRepository(database).rename_conversation(conversation_id, title)

    def delete_conversation(self, conversation_id: str) -> bool:
        with KnowledgeDatabase(self.paths.database) as database:
            return ConversationRepository(database).delete_conversation(conversation_id)

    def export_conversation(
        self,
        conversation_id: str,
        destination: str | Path | None = None,
    ) -> Path:
        with KnowledgeDatabase(self.paths.database) as database:
            repository = ConversationRepository(database)
            record = repository.conversation_record(conversation_id)
            markdown = repository.conversation_markdown(conversation_id)
        title = str(record.get("title") or "新对话")
        safe_title = "".join(
            "-" if char in '/:*?\"<>|\\' else char for char in title
        ).strip(" .-")[:80] or "新对话"
        date = str(record.get("created_at") or "")[:10] or datetime.now().astimezone().strftime("%Y-%m-%d")
        filename = f"{date}--{safe_title}--{conversation_id[-8:]}.md"
        if destination is None:
            target = self.paths.notes / "对话记录" / filename
        else:
            requested = Path(destination).expanduser()
            target = requested / filename if requested.suffix.casefold() != ".md" else requested
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(markdown)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return target

    def save_answer_feedback(
        self, answer_id: str, rating: str, comment: str = ""
    ) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            return ConversationRepository(database).save_answer_feedback(
                answer_id, rating, comment
            )

    def save_partial_answer(self, conversation_id: str, markdown: str) -> str:
        content = str(markdown or "").strip()
        if not content:
            raise ValueError("没有可保存的部分回答")
        with KnowledgeDatabase(self.paths.database) as database:
            message = ConversationRepository(database).add_message(
                conversation_id,
                "assistant",
                content,
                {"partial": True, "stopped": True},
            )
        return message.message_id

    def create_ingestion_job(
        self, items: Iterable[str | Path], *, metadata: dict[str, object] | None = None
    ) -> dict[str, object]:
        values = [str(item).strip() for item in items if str(item).strip()]
        with KnowledgeDatabase(self.paths.database) as database:
            return IngestionJobRepository(database).create_job(values, metadata=metadata)

    def ingestion_jobs(
        self,
        limit: int = 200,
        *,
        statuses: Iterable[str] | None = None,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            return IngestionJobRepository(database).list_jobs(
                statuses=statuses, limit=limit, offset=offset
            )

    def ingestion_job(self, job_id: str) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            return IngestionJobRepository(database).job_record(job_id)

    def resume_ingestion_job(
        self,
        job_id: str,
        *,
        progress: Callable[[ProgressEvent], None] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> IngestionSummary:
        with KnowledgeDatabase(self.paths.database) as database:
            sources = IngestionJobRepository(database).pending_sources(job_id)
        if not sources:
            raise ValueError("该任务没有可继续处理的资料")
        return self.ingest(
            sources, progress=progress, cancellation=cancellation, job_id=job_id
        )

    def retry_ingestion_job(
        self,
        job_id: str,
        *,
        progress: Callable[[ProgressEvent], None] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> IngestionSummary:
        with KnowledgeDatabase(self.paths.database) as database:
            jobs = IngestionJobRepository(database)
            jobs.reset_failed_items(job_id)
        return self.resume_ingestion_job(
            job_id, progress=progress, cancellation=cancellation
        )

    def cancel_ingestion_job(self, job_id: str) -> bool:
        with KnowledgeDatabase(self.paths.database) as database:
            return IngestionJobRepository(database).cancel_job(job_id)

    def rebuild_search_index(self) -> dict[str, int]:
        config = self.config()
        embedding = build_embedding_provider(config)
        with KnowledgeDatabase(self.paths.database) as database:
            return IndexingService(database, embedding).reindex()

    def facets(self) -> dict[str, list[dict[str, object]]]:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.list_facets()

    def document_chunks(self, document_id: str) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            values = database.list_chunks(document_id)
        for value in values:
            for key in ("heading_path_json", "source_reference_json", "metadata_json"):
                if key in value:
                    try:
                        value[key.removesuffix("_json")] = json.loads(value.pop(key) or "{}")
                    except json.JSONDecodeError:
                        value[key.removesuffix("_json")] = {}
        return values

    def latest_transcript(self, document_id: str) -> dict[str, object] | None:
        """Return the latest persisted Transcript V2 and its playable local source."""

        with KnowledgeDatabase(self.paths.database) as database:
            document = database.get_document(document_id)
            if document is None:
                raise ValueError("资料不存在或已被移除")
            repository = TranscriptRepository(database)
            runs = repository.list_runs(document_id=document_id, limit=1)
            if not runs:
                return None
            run = runs[0]
            transcript = repository.get_transcript(run.id)
            if transcript is None:
                return None
            media_path = str(document["local_path"] or "").strip()
            if not media_path:
                candidate = str(transcript.source.original_uri or "").strip()
                if candidate and not candidate.casefold().startswith(("http://", "https://")):
                    media_path = candidate
            return {
                "run": run.to_dict(),
                "transcript": transcript.to_dict(),
                "media_path": media_path or None,
                "document_id": document_id,
                "document_title": str(document["title"]),
                "document_enabled": bool(document["enabled"]),
            }

    def _deep_correction_workflow(self) -> DeepCorrectionWorkflow:
        return DeepCorrectionWorkflow(self.paths, self.config(), self.settings)

    def deep_correct_transcript(
        self,
        run_id: str,
        *,
        progress: Callable[..., None] | None = None,
        cancellation: CancellationToken | None = None,
        correction_run_id: str | None = None,
    ) -> dict[str, object]:
        """Run the first-party deep-correction service without Codex CLI."""

        def report(stage: str, completed: int, total: int, message: str) -> None:
            if progress:
                progress(stage, completed, total, message)

        def created(value: str) -> None:
            if progress:
                progress({
                    "stage": "validation",
                    "completed": 0,
                    "total": 11,
                    "message": "深度精校任务已建立，原始转写已锁定为不可变证据",
                    "correction_run_id": value,
                })

        result = self._deep_correction_workflow().run(
            run_id,
            progress=report,
            cancellation=cancellation,
            correction_run_id=correction_run_id,
            run_created=created,
        )
        resolved = str(result.get("correction_run_id") or "").strip()
        if not resolved:
            raise RuntimeError("深度精校完成但没有返回运行 ID")
        auto_change_ids = [
            str(item)
            for item in (result.get("auto_accepted_change_ids") or [])
            if str(item).strip()
        ]
        index_report: dict[str, object] | None = None
        if auto_change_ids:
            with KnowledgeDatabase(self.paths.database) as database:
                correction_repository = DeepCorrectionRepository(database)
                changes = [
                    correction_repository.get_change(change_id)
                    for change_id in auto_change_ids
                ]
                correction = correction_repository.get_run(resolved)
                transcript_run = (
                    TranscriptRepository(database).get_run(correction.transcript_run_id)
                    if correction else None
                )
                document = (
                    database.get_document(transcript_run.document_id)
                    if transcript_run and transcript_run.document_id else None
                )
                affected_segment_ids = list(dict.fromkeys(
                    segment_id
                    for change in changes if change is not None
                    for segment_id in change.source_segment_ids
                ))
                should_refresh = bool(document is not None and document["enabled"])
            if should_refresh:
                try:
                    index_report = self.refresh_transcript_index(
                        run_id,
                        affected_segment_ids=affected_segment_ids,
                    )
                except Exception as exc:
                    index_report = {
                        "status": "failed",
                        "error": str(exc)[:500],
                        "retry": "rebuild_search_index",
                    }
        return {
            "correction_run_id": resolved,
            "snapshot": self.deep_correction_snapshot(resolved),
            "output_path": result.get("output_path"),
            "output_checksum": result.get("output_checksum"),
            "warnings": result.get("warnings") or [],
            "auto_accepted_change_ids": auto_change_ids,
            "index": index_report,
        }

    def deep_correction_snapshot(self, correction_run_id: str) -> dict[str, object]:
        """Return the durable, mapping-only audit snapshot consumed by the UI."""

        with KnowledgeDatabase(self.paths.database) as database:
            repository = DeepCorrectionRepository(database)
            if repository.get_run(correction_run_id) is None:
                raise ValueError("深度精校任务不存在")
            return repository.snapshot(correction_run_id)

    def review_deep_correction_change(
        self,
        change_id: str,
        decision: str,
    ) -> dict[str, object]:
        """Persist one human decision; accepted text is atomically applied."""

        reviewed = self._deep_correction_workflow().review_change(
            change_id, decision=decision
        )
        change = reviewed.get("change")
        change_value = dict(change) if isinstance(change, dict) else {}
        correction_run_id = str(change_value.get("correction_run_id") or "")
        index_report: dict[str, object] | None = None
        if str(decision).strip().casefold() == "accepted" and correction_run_id:
            with KnowledgeDatabase(self.paths.database) as database:
                correction = DeepCorrectionRepository(database).get_run(correction_run_id)
                transcript_run = (
                    TranscriptRepository(database).get_run(correction.transcript_run_id)
                    if correction else None
                )
                document = (
                    database.get_document(transcript_run.document_id)
                    if transcript_run and transcript_run.document_id else None
                )
                should_refresh = bool(document is not None and document["enabled"])
                transcript_run_id = transcript_run.id if transcript_run else None
            if should_refresh and transcript_run_id:
                try:
                    index_report = self.refresh_transcript_index(
                        transcript_run_id,
                        affected_segment_ids=change_value.get("source_segment_ids") or (),
                    )
                except Exception as exc:
                    # The human decision and corrected V2 are already durable.
                    # Report an independently retryable index failure instead of
                    # pretending the atomic review itself was rolled back.
                    index_report = {
                        "status": "failed",
                        "error": str(exc)[:500],
                        "retry": "rebuild_search_index",
                    }
        return {"change": change_value, "index": index_report}

    def export_deep_correction(self, correction_run_id: str) -> dict[str, object]:
        return self._deep_correction_workflow().export(correction_run_id)

    def refresh_transcript_index(
        self,
        run_id: str,
        *,
        affected_segment_ids: Iterable[str] = (),
    ) -> dict[str, object]:
        """Atomically rebuild one run's speaker-aware chunks after human edits.

        The whole run is regrouped because changing a speaker can move a segment
        across chunk boundaries.  Embeddings are computed before SQLite changes,
        so a provider failure leaves the previous searchable version untouched.
        """

        config = self.config()
        embedding = build_embedding_provider(config)
        with KnowledgeDatabase(self.paths.database) as database:
            repository = TranscriptRepository(database)
            record = repository.get_run(run_id)
            transcript = repository.get_transcript(run_id)
            if record is None or transcript is None:
                raise ValueError("转写任务不存在或事实层不完整")
            if not record.document_id:
                raise ValueError("该转写任务尚未关联知识库资料")
            if record.status != "completed" or str(record.quality.get("status") or "") != "pass":
                raise ValueError("转写尚未通过人工复核，不能刷新检索索引")
            document_row = database.get_document(record.document_id)
            if document_row is None:
                raise ValueError("转写任务关联的资料已被移除")

            speaker_names = {
                speaker.id: speaker.display_name or speaker.id
                for speaker in transcript.speakers
            }
            segments = [
                ContentSegment(
                    id=segment.id,
                    sequence=segment.start_ms / 1000.0,
                    modality="speech",
                    text=segment.effective_text,
                    location={
                        "timestamp_start": segment.start_ms / 1000.0,
                        "timestamp_end": segment.end_ms / 1000.0,
                        "speaker_id": segment.speaker_id,
                    },
                    metadata={
                        "language": transcript.run.language,
                        "provider": transcript.run.provider,
                        "model": transcript.run.model,
                        "confidence": segment.confidence,
                        "speaker_id": segment.speaker_id,
                        "speaker_name": speaker_names.get(segment.speaker_id or ""),
                        "overlap": "overlap" in segment.flags,
                        "asr_run_id": transcript.run.id,
                        "quality_status": transcript.quality.status,
                        "quality_flags": list(segment.flags),
                        "raw_text": segment.raw_text,
                    },
                )
                for segment in transcript.segments
                if segment.effective_text.strip()
            ]
            if not segments:
                raise ValueError("转写中没有可建立索引的文字")

            source = SourceReference(
                source_id=str(document_row["source_id"]),
                media_type=str(document_row["media_type"]),
                title=str(document_row["title"]),
                document_id=record.document_id,
                original_uri=document_row["original_uri"],
                local_path=document_row["local_path"],
                obsidian_path=document_row["obsidian_path"],
                checksum=document_row["checksum"],
            )
            document = KnowledgeDocument(
                source_id=str(document_row["source_id"]),
                title=str(document_row["title"]),
                media_type=str(document_row["media_type"]),
                segments=segments,
                source=source,
                document_id=record.document_id,
                metadata=json.loads(document_row["metadata_json"] or "{}"),
            )
            rebuilt = MediaAwareChunker().chunk(document)
            existing_rows = list(database.get_chunks(record.document_id).values())

            def belongs_to_run(row: object) -> bool:
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")  # type: ignore[index]
                except (TypeError, json.JSONDecodeError):
                    return False
                run_ids = metadata.get("asr_run_ids")
                return metadata.get("asr_run_id") == run_id or (
                    isinstance(run_ids, list) and run_id in run_ids
                )

            old_run_rows = [row for row in existing_rows if belongs_to_run(row)]
            old_by_id = {str(row["id"]): row for row in old_run_rows}
            base_ordinal = min(
                (int(row["ordinal"]) for row in old_run_rows),
                default=max((int(row["ordinal"]) for row in existing_rows), default=-1) + 1,
            )
            to_embed = []
            for offset, chunk in enumerate(rebuilt):
                chunk.ordinal = base_ordinal + offset
                previous = old_by_id.get(chunk.id)
                if previous is not None and previous["content_hash"] == chunk.content_hash:
                    chunk.embedding_status = str(previous["embedding_status"])
                    chunk.created_at = str(previous["created_at"])
                else:
                    chunk.embedding_status = "pending"
                    to_embed.append(chunk)

            vectors = embedding.embed([chunk.content for chunk in to_embed]) if to_embed else []
            if len(vectors) != len(to_embed):
                raise RuntimeError("embedding provider returned a mismatched vector count")
            stale_ids = sorted(set(old_by_id) - {chunk.id for chunk in rebuilt})
            vector_store = SQLiteVectorStore(
                database,
                provider=embedding.name,
                model=embedding.model,
            )
            with database.connection:
                database.delete_chunks(stale_ids)
                for chunk in rebuilt:
                    database.upsert_chunk(chunk, document.title)
                for chunk, vector in zip(to_embed, vectors):
                    vector_store.upsert(
                        chunk.id,
                        vector,
                        provider=embedding.name,
                        model=embedding.model,
                        content_hash=chunk.content_hash,
                    )
            requested = tuple(dict.fromkeys(str(value) for value in affected_segment_ids if str(value)))
            return {
                "run_id": run_id,
                "document_id": record.document_id,
                "rebuilt_chunks": len(rebuilt),
                "embedded_chunks": len(to_embed),
                "deleted_chunks": len(stale_ids),
                "affected_segment_ids": list(requested),
            }

    def approve_transcript_for_retrieval(self, run_id: str) -> dict[str, object]:
        """Record human review, build the deferred index, and enable Q&A."""

        with KnowledgeDatabase(self.paths.database) as database:
            repository = TranscriptRepository(database)
            record = repository.get_run(run_id)
            if record is None or not record.document_id:
                raise ValueError("转写任务不存在或尚未关联资料")
            quality = dict(record.quality)
            metrics = quality.get("metrics")
            metrics = dict(metrics) if isinstance(metrics, dict) else {}
            metrics.update({"human_reviewed": True, "human_reviewed_at": utcnow_iso()})
            quality["status"] = "pass"
            quality["metrics"] = metrics
            repository.update_run_status(run_id, "completed", quality=quality)
        try:
            index_report = self.refresh_transcript_index(run_id)
        except BaseException:
            # A failed embedding/index refresh must not turn a deferred source
            # into an apparently approved-but-unsearchable one.
            with KnowledgeDatabase(self.paths.database) as database:
                TranscriptRepository(database).update_run_status(
                    run_id,
                    record.status,
                    quality=record.quality,
                )
            raise
        with KnowledgeDatabase(self.paths.database) as database:
            if not database.set_document_enabled(record.document_id, True):
                raise ValueError("无法重新启用关联资料")
            return {
                "run_id": run_id,
                "document_id": record.document_id,
                "quality_status": "pass",
                "enabled": True,
                "rebuilt_chunks": index_report["rebuilt_chunks"],
                "embedded_chunks": index_report["embedded_chunks"],
            }

    def rename_document(self, document_id: str, title: str) -> bool:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.rename_document(document_id, title)

    def set_document_enabled(self, document_id: str, enabled: bool) -> bool:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.set_document_enabled(document_id, enabled)

    def update_document_facets(
        self, document_id: str, *, collections: list[str], tags: list[str]
    ) -> None:
        with KnowledgeDatabase(self.paths.database) as database:
            if database.get_document(document_id) is None:
                raise ValueError("资料不存在或已被移除")
            database.update_document_facets(document_id, collections, tags)

    def delete_document(self, document_id: str) -> bool:
        """Remove the searchable record but leave archived originals recoverable."""
        with KnowledgeDatabase(self.paths.database) as database:
            return database.delete_document(document_id)

    def reingest_document(self, document_id: str) -> IngestionSummary:
        with KnowledgeDatabase(self.paths.database) as database:
            row = database.get_document(document_id)
            if row is None:
                raise ValueError("资料不存在")
            metadata = json.loads(row["metadata_json"] or "{}")
            source = metadata.get("source_identity") or row["original_uri"] or row["local_path"]
        if not source:
            raise ValueError("没有记录可重新解析的源位置")
        return self.ingest([str(source)])

    def duplicate_groups(self) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.duplicate_groups()

    def quality_overview(self) -> list[dict[str, object]]:
        values = []
        for document in self.documents(limit=2000):
            report = document.get("quality") or {}
            values.append({
                "document_id": document["id"],
                "title": document["title"],
                "media_type": document["media_type"],
                "enabled": document["enabled"],
                "score": int(report.get("score", 0)) if isinstance(report, dict) else 0,
                "grade": str(report.get("grade", "历史资料")) if isinstance(report, dict) else "历史资料",
                "accepted": bool(report.get("accepted", True)) if isinstance(report, dict) else True,
                "checks": list(report.get("checks", [])) if isinstance(report, dict) else [],
                "warnings": list((document.get("metadata") or {}).get("warnings", [])),
            })
        return values

    def add_annotation(
        self,
        document_id: str,
        content: str,
        *,
        chunk_id: str | None = None,
        locator: dict[str, object] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        if not content.strip():
            raise ValueError("批注内容不能为空")
        annotation_id = "annotation-" + uuid.uuid4().hex
        with KnowledgeDatabase(self.paths.database) as database:
            database.save_annotation(
                annotation_id, document_id, content, chunk_id=chunk_id,
                locator=locator, tags=tags or [],
            )
        return annotation_id

    def annotations(self, document_id: str | None = None) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            values = database.list_annotations(document_id)
        for value in values:
            value["locator"] = json.loads(value.pop("locator_json") or "{}")
            value["tags"] = json.loads(value.pop("tags_json") or "[]")
        return values

    def create_artifact(
        self,
        artifact_type: str,
        title: str,
        *,
        document_ids: list[str] | None = None,
        model_id: str | None = None,
    ) -> dict[str, object]:
        prompts = {
            "report": "请把选定资料整理成一份结构清晰的综合研究报告，区分事实、推断和待验证项。",
            "compare": "请比较选定资料的核心观点、共识、分歧、证据强弱和适用边界，用表格总结。",
            "timeline": "请抽取选定资料中的时间、事件和因果关系，生成按时间排序的时间线。",
            "quiz": "请根据选定资料生成10道由浅入深的测验题，包含答案、解析和证据引用。",
            "flashcards": "请生成可复习的问答闪卡，每张只覆盖一个知识点，并保留证据引用。",
            "mindmap": "请生成层级清晰的 Markdown 思维导图大纲，覆盖主题、概念、关系和证据引用。",
        }
        if artifact_type not in prompts:
            raise ValueError("不支持的知识工坊类型")
        answer = self.ask(
            prompts[artifact_type], model_id=model_id, deep_analysis=True,
            document_ids=document_ids, top_k=12,
        )
        artifact_id = "artifact-" + uuid.uuid4().hex
        clean_title = title.strip() or f"知识工坊-{artifact_type}"
        with KnowledgeDatabase(self.paths.database) as database:
            database.save_artifact(
                artifact_id, artifact_type, clean_title, answer.markdown,
                document_ids or [], {"answer_id": answer.answer_id, "model": answer.model},
            )
        folder = self.paths.notes / "知识工坊"
        safe = "".join("-" if char in '/:*?\"<>|\\' else char for char in clean_title).strip(" .-")[:80]
        path = folder / f"{datetime.now().astimezone():%Y-%m-%d}--{safe}--{artifact_id[-8:]}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {clean_title}\n\n{answer.markdown}\n", encoding="utf-8")
        return {"id": artifact_id, "type": artifact_type, "title": clean_title, "markdown": answer.markdown, "path": str(path)}

    def artifacts(self) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            values = database.list_artifacts()
        for value in values:
            value["source_document_ids"] = json.loads(value.pop("source_document_ids_json") or "[]")
            value["metadata"] = json.loads(value.pop("metadata_json") or "{}")
        return values

    def add_watched_folder(
        self, path: str | Path, *, collection: str = "自动同步", recursive: bool = True
    ) -> str:
        folder = Path(path).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError("请选择存在的文件夹")
        watcher_id = "watch-" + uuid.uuid4().hex
        with KnowledgeDatabase(self.paths.database) as database:
            database.add_watched_folder(
                watcher_id, str(folder), collection=collection, recursive=recursive
            )
            rows = database.list_watched_folders()
        return str(next(item["id"] for item in rows if item["path"] == str(folder)))

    def watched_folders(self) -> list[dict[str, object]]:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.list_watched_folders()

    def remove_watched_folder(self, watcher_id: str) -> bool:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.delete_watched_folder(watcher_id)

    def scan_watched_folders(self) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            watchers = [item for item in database.list_watched_folders() if item["enabled"]]
        reports = []
        for watcher in watchers:
            previous = dict((watcher.get("metadata") or {}).get("files", {}))
            try:
                scan = scan_folder(
                    str(watcher["path"]), previous,
                    recursive=bool(watcher["recursive"]),
                )
            except ValueError as exc:
                with KnowledgeDatabase(self.paths.database) as database:
                    database.update_watched_folder(str(watcher["id"]), enabled=False)
                reports.append({"watcher_id": watcher["id"], "status": "disabled", "error": str(exc)})
                continue
            imported = self.ingest(scan.changed) if scan.changed else None
            file_state: dict[str, dict[str, object]] = {
                key: dict(value) for key, value in scan.current.items()
            }
            if imported:
                root = Path(str(watcher["path"]))
                for result in imported.results:
                    if not result.document_id:
                        continue
                    try:
                        relative = Path(result.item).resolve().relative_to(root).as_posix()
                    except (OSError, ValueError):
                        continue
                    file_state.setdefault(relative, {})["document_id"] = result.document_id
                    with KnowledgeDatabase(self.paths.database) as database:
                        facets = database.document_facets(result.document_id)
                        database.update_document_facets(
                            result.document_id,
                            [*facets["collections"], str(watcher["collection"])],
                            facets["tags"],
                        )
            for relative, state in previous.items():
                if relative in file_state and state.get("document_id"):
                    file_state[relative]["document_id"] = state["document_id"]
                elif relative in scan.removed and state.get("document_id"):
                    with KnowledgeDatabase(self.paths.database) as database:
                        database.set_document_enabled(str(state["document_id"]), False)
            now = utcnow_iso()
            with KnowledgeDatabase(self.paths.database) as database:
                database.update_watched_folder(
                    str(watcher["id"]),
                    metadata={"files": file_state},
                    last_scan_at=now,
                )
            reports.append({
                "watcher_id": watcher["id"], "status": "complete",
                "imported": imported.succeeded if imported else 0,
                "failed": imported.failed if imported else 0,
                "removed_disabled": len(scan.removed), "unchanged": scan.unchanged,
            })
        return {"status": "complete", "folders": len(watchers), "reports": reports}

    def source_packages(self) -> list[dict[str, object]]:
        root = self.paths.archive / "source-packages"
        values = []
        if not root.is_dir():
            return values
        for manifest in root.glob("*/manifest.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload["manifest_path"] = str(manifest)
                values.append(payload)
        return sorted(values, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def _privacy_ocr_reader(self) -> Callable[[Path], object]:
        """Build a local-only OCR callback for privacy inspection."""

        from ..ingestion.ocr import extract_ocr

        def read(path: Path) -> object:
            return extract_ocr(
                path,
                requested_engine=self.settings.ocr_engine,
                complex_layout=self.settings.ocr_complex_layout_enabled,
                allow_paddle=(
                    self.settings.ocr_complex_layout_enabled
                    or self.settings.ocr_engine == "paddleocr"
                ),
                low_confidence_threshold=self.settings.ocr_low_confidence_threshold,
            ).to_dict()

        return read

    def privacy_scan(self, *, enable_image_ocr: bool = False) -> dict[str, object]:
        """Scan shareable knowledge areas while keeping credentials and paths private."""

        ocr_reader = self._privacy_ocr_reader() if enable_image_ocr else None
        roots = (
            ("原始资料", self.paths.archive),
            ("知识笔记", self.paths.notes),
            ("图片资产", self.paths.assets),
            ("音视频转写", self.paths.transcripts),
        )
        reports: list[tuple[str, dict[str, object]]] = []
        for label, root in roots:
            if not root.exists():
                continue
            reports.append(
                (
                    label,
                    scan_privacy(
                        root,
                        enable_image_ocr=enable_image_ocr,
                        ocr_reader=ocr_reader,
                    ).to_dict(),
                )
            )
        findings: list[dict[str, object]] = []
        limitations: list[str] = []
        totals = {
            "scanned_files": 0,
            "text_files_scanned": 0,
            "image_files_checked": 0,
            "ocr_images_scanned": 0,
            "skipped_files": 0,
        }
        statuses: list[str] = []
        for label, report in reports:
            statuses.append(str(report.get("status") or "review"))
            for key in totals:
                totals[key] += int(report.get(key) or 0)
            for value in report.get("limitations") or []:
                text = str(value)
                if text not in limitations:
                    limitations.append(text)
            for raw_finding in report.get("findings") or []:
                if not isinstance(raw_finding, dict):
                    continue
                finding = dict(raw_finding)
                finding["redacted_path"] = (
                    f"{label}/{finding.get('redacted_path') or '<已脱敏路径>'}"
                )
                findings.append(finding)
        status = (
            "blocked"
            if "blocked" in statuses
            else "review"
            if "review" in statuses
            else "clean"
        )
        return {
            "status": status,
            "root_name": "AI静静可分享知识",
            **totals,
            "has_blockers": status == "blocked",
            "findings": findings,
            "limitations": limitations,
        }

    def create_safe_share_copy(
        self,
        destination: str | Path,
        *,
        include_notes: bool = False,
        document_ids: Iterable[str] = (),
        scan_images_with_ocr: bool = False,
        allow_review_findings: bool = False,
    ) -> dict[str, object]:
        ids = list(dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip()))
        if not include_notes and not ids:
            raise ValueError("请至少选择知识笔记或一份原始资料")
        selected: list[str] = []
        root = self.paths.root.resolve()
        if include_notes:
            # Source Notes and saved answers may intentionally retain private local
            # locators.  Share only user-promoted/workshop knowledge by default;
            # the privacy scanner still checks every selected byte afterwards.
            for relative in (
                Path("notes") / "正式知识",
                Path("notes") / "知识工坊",
            ):
                candidate = root / relative
                if candidate.is_dir() and any(path.is_file() for path in candidate.rglob("*")):
                    selected.append(relative.as_posix())
        with KnowledgeDatabase(self.paths.database) as database:
            for document_id in ids:
                row = database.get_document(document_id)
                if row is None:
                    raise ValueError("选中的资料不存在或已被移除")
                raw_path = str(row["local_path"] or "").strip()
                if not raw_path:
                    raise ValueError(
                        f"资料“{row['title']}”没有可分享的本地归档，请先重新导入"
                    )
                source = Path(raw_path).expanduser().resolve()
                try:
                    relative = source.relative_to(root)
                except ValueError:
                    raise ValueError(
                        f"资料“{row['title']}”仍在应用目录外，请先重新导入并归档"
                    ) from None
                if not source.exists():
                    raise ValueError(f"资料“{row['title']}”的归档文件已丢失")
                selected.append(relative.as_posix())
        if not selected:
            raise ValueError("尚无可分享的正式知识、知识工坊成果或已选资料")
        options = ShareCopyOptions(
            include_notes=False,
            public_sources=tuple(dict.fromkeys(selected)),
            scan_images_with_ocr=scan_images_with_ocr,
            require_clean_scan=not allow_review_findings,
        )
        try:
            report = create_share_copy(
                self.paths.root,
                destination,
                options=options,
                ocr_reader=self._privacy_ocr_reader() if scan_images_with_ocr else None,
            )
        except PrivacyViolationError as exc:
            count = len(exc.report.findings)
            raise ValueError(
                f"安全分享已停止：发现 {count} 项需处理或无法完整检查的风险。"
                "请先运行本地隐私扫描，调整选中内容后再试。"
            ) from None
        return report.to_dict(include_destination=True)

    def create_backup(self) -> Path:
        return create_product_backup(self.paths)

    def restore_backup(self, backup: str | Path) -> dict[str, object]:
        report = restore_product_backup(self.paths, backup)
        self.reload()
        return report

    def repair_database(self) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            before = database.integrity_report()
            rebuilt = database.rebuild_fts()
            database.connection.execute("PRAGMA optimize")
            database.connection.execute("VACUUM")
            after = database.integrity_report()
        return {"before": before, "after": after, "fts_rows": rebuilt}

    def database_health(self) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            return database.integrity_report()

    def check_for_updates(self) -> dict[str, object]:
        return check_for_update(
            __version__, self.settings.update_manifest_url
        ).to_dict()

    def download_update(
        self,
        download_url: str,
        sha256: str,
        destination_dir: str | Path | None = None,
    ) -> Path:
        destination = (
            Path(destination_dir).expanduser().resolve()
            if destination_dir is not None
            else self.paths.cache / "updates"
        )
        return download_verified_update(download_url, sha256, destination)

    def save_answer_note(self, answer: KnowledgeAnswer, question: str) -> Path:
        folder = self.paths.notes / "AI回答"
        date = datetime.now().astimezone().strftime("%Y-%m-%d")
        safe = "".join("-" if char in '/:*?\"<>|\\' else char for char in question).strip(" .-")[:80]
        path = folder / f"{date}--{safe or '知识问答'}--{answer.answer_id[-8:]}.md"
        lines = [
            "---",
            f"answer_id: {json.dumps(answer.answer_id)}",
            f"conversation_id: {json.dumps(answer.conversation_id)}",
            f"model: {json.dumps(answer.model)}",
            f"created_at: {json.dumps(answer.created_at)}",
            'tags: ["AI静静/问答"]',
            "---",
            "",
            f"# {question.strip() or '知识问答'}",
            "",
            answer.markdown,
            "",
            "## 证据来源",
            "",
        ]
        for citation in answer.citations:
            location = []
            if citation.page_number is not None:
                location.append(f"P{citation.page_number}")
            if citation.slide_number is not None:
                location.append(f"S{citation.slide_number}")
            if citation.timestamp_start is not None:
                location.append(f"{citation.timestamp_start:g}s")
            target = citation.original_uri or citation.local_path or "本地知识库"
            lines.append(f"- [{citation.citation_id}] {citation.title} {' / '.join(location)} — {target}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def sync_from_obsidian(self) -> dict[str, object]:
        if not self.settings.obsidian_vault:
            raise ValueError("请先在设置中选择 Obsidian Vault")
        config = self.config()
        embedding = build_embedding_provider(config)
        with KnowledgeDatabase(self.paths.database) as database:
            indexing = IndexingService(database, embedding)
            report = ObsidianMarkdownSync(database, indexing, self.settings.obsidian_vault).sync()
            return report.to_dict()

    def export_notes_to_obsidian(self) -> dict[str, object]:
        if not self.settings.obsidian_vault:
            raise ValueError("请先在设置中选择 Obsidian Vault")
        vault = Path(self.settings.obsidian_vault).expanduser().resolve()
        if not vault.is_dir():
            raise ValueError("Obsidian Vault 路径不存在")
        destination = vault / "AI静静知识库"
        copied = 0
        for source in self.paths.notes.rglob("*.md"):
            relative = source.relative_to(self.paths.notes)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or source.read_bytes() != target.read_bytes():
                shutil.copy2(source, target)
                copied += 1
        return {"status": "complete", "copied": copied, "destination": str(destination), "completed_at": utcnow_iso()}
