from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .. import __version__
from ..answer_models import available_answer_models
from ..config import AppConfig, KEYRING_SERVICE
from ..ingestion import CancellationToken, IngestionService, IngestionSummary, ProgressEvent
from ..models import SearchResult, utcnow_iso
from ..product import DesktopSettings, ProductPaths, PRODUCT_NAME
from ..qa.engine import KnowledgeQAEngine
from ..qa.models import KnowledgeAnswer
from ..retrieval import KnowledgeRetriever
from ..runtime import build_answer_provider, build_embedding_provider, build_rerank_provider
from ..storage import KnowledgeDatabase
from ..sync import ObsidianMarkdownSync, scan_folder
from ..indexing import IndexingService
from .update import check_for_update


DEFAULT_PROVIDERS = {
    "deepseek": {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
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


class ProviderConfigStore:
    """Credential store whose public status methods never expose API keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._migrate_plaintext_secrets()

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

    def ingest(
        self,
        items: Iterable[str | Path],
        *,
        progress: Callable[[ProgressEvent], None] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> IngestionSummary:
        values = list(items)
        service = IngestionService(self.paths, self.config(), self.settings)
        result = service.ingest(values, progress=progress, cancellation=cancellation)
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
    ) -> KnowledgeAnswer:
        config = self.config()
        requested_model = model_id or self.settings.default_model
        model_ids = {str(item["id"]) for item in self.model_choices()}
        if requested_model not in model_ids:
            requested_model = "local-extractive"
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
            )

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

    def create_backup(self) -> Path:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        target = self.paths.backups / f"AI静静备份-{stamp}.aijjbackup"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".backup-", suffix=".tmp", dir=self.paths.backups
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        db_copy = self.paths.backups / f".knowledge-{stamp}.db"
        try:
            source = sqlite3.connect(self.paths.database)
            destination = sqlite3.connect(db_copy)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(db_copy, "knowledge.db")
                if self.paths.settings.is_file():
                    archive.write(self.paths.settings, "settings.json")
                manifest = {
                    "format": "ai-jingjing-backup-v1",
                    "created_at": utcnow_iso(),
                    "product": PRODUCT_NAME,
                    "includes_credentials": False,
                }
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
                for base, prefix in ((self.paths.notes, "notes"), (self.paths.archive, "archive")):
                    for file in base.rglob("*"):
                        if file.is_file():
                            archive.write(file, f"{prefix}/{file.relative_to(base).as_posix()}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
            db_copy.unlink(missing_ok=True)
        return target

    def restore_backup(self, backup: str | Path) -> dict[str, object]:
        source = Path(backup).expanduser().resolve()
        if not source.is_file():
            raise ValueError("备份文件不存在")
        safety = self.create_backup()
        with zipfile.ZipFile(source) as archive:
            names = set(archive.namelist())
            if "manifest.json" not in names or "knowledge.db" not in names:
                raise ValueError("不是有效的 AI静静备份")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("format") != "ai-jingjing-backup-v1":
                raise ValueError("不支持的备份版本")
            for name in names:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("备份包含不安全路径")
            descriptor, temp_db_name = tempfile.mkstemp(prefix=".restore-", suffix=".db", dir=self.paths.root)
            os.close(descriptor)
            temp_db = Path(temp_db_name)
            try:
                temp_db.write_bytes(archive.read("knowledge.db"))
                check = sqlite3.connect(temp_db)
                try:
                    if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("备份数据库完整性检查失败")
                finally:
                    check.close()
                restore_source = sqlite3.connect(temp_db)
                restore_destination = sqlite3.connect(self.paths.database)
                try:
                    restore_source.backup(restore_destination)
                finally:
                    restore_destination.close()
                    restore_source.close()
                if "settings.json" in names:
                    self.paths.settings.write_bytes(archive.read("settings.json"))
                for prefix, base in (("notes/", self.paths.notes), ("archive/", self.paths.archive)):
                    for name in names:
                        if not name.startswith(prefix) or name.endswith("/"):
                            continue
                        target = base / Path(name).relative_to(prefix)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.read(name))
            finally:
                temp_db.unlink(missing_ok=True)
        self.reload()
        return {"status": "complete", "safety_backup": str(safety), "restored": str(source)}

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
