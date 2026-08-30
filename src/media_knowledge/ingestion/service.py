from __future__ import annotations

import errno
import json
import os
import re
import shutil
import tempfile
import urllib.parse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

from ..config import AppConfig
from ..indexing import IndexingService
from ..models import KnowledgeDocument, SourceReference, sha256_text, utcnow_iso
from ..product import DesktopSettings, ProductPaths, PRODUCT_NAME
from ..qa.models import AnswerRequest
from ..runtime import build_answer_provider, build_embedding_provider
from ..storage import KnowledgeDatabase
from ..transcripts import TranscriptRepository, transcript_from_dict
from .cleanup import TemporaryCleanupRegistry
from .extractors import (
    ExtractionContext,
    MissingExtractorDependency,
    extractor_for,
    safe_stem,
    url_extractor_for,
)
from .types import CancellationToken, CancelledError, ExtractionResult, ProgressEvent
from .quality import QualityGateError, evaluate_extraction
from .vision import MultimodalInterpreter


ProgressCallback = Callable[[ProgressEvent], None]
_PUBLIC_CLEANUP_WARNING = (
    "临时清理失败：缓存目录已安全登记，将在下次启动或导入时自动重试"
)
_SYNTHESIS_HEADINGS = (
    "## 已确认事实",
    "## 推测与待验证",
    "## 争议与不同观点",
    "## 结论与决策",
    "## 行动项",
)


@dataclass(slots=True)
class IngestionResult:
    item: str
    title: str = ""
    media_type: str = ""
    status: str = "failed"
    document_id: str | None = None
    source_id: str | None = None
    package_id: str | None = None
    archive_path: str | None = None
    note_path: str | None = None
    chunks: int = 0
    extracted_characters: int = 0
    warnings: list[str] = field(default_factory=list)
    quality_report: dict[str, object] = field(default_factory=dict)
    transcript_run_id: str | None = None
    error: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IngestionSummary:
    results: list[IngestionResult] = field(default_factory=list)
    job_id: str | None = None
    started_at: str = field(default_factory=utcnow_iso)
    completed_at: str = ""
    duration_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def succeeded(self) -> int:
        return sum(item.status not in {"failed", "cancelled"} for item in self.results)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.results)

    @property
    def cancelled(self) -> int:
        return sum(item.status == "cancelled" for item in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "results": [item.to_dict() for item in self.results],
        }


@dataclass(frozen=True, slots=True)
class _PersistedEvidence:
    """Evidence prepared for indexing, with an exact rollback target when newly created."""

    package: Path | None
    source_path: Path | None
    rollback_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _PersistedDerivedArtifacts:
    """New immutable transcript/frame files that can be rolled back before indexing."""

    rollback_paths: tuple[Path, ...] = ()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_copy_file(
    source: Path,
    destination: Path,
    *,
    cancellation: CancellationToken | None = None,
) -> None:
    """Copy to a sibling temporary file, then replace the final path atomically."""

    if source.is_symlink() or not source.is_file():
        raise OSError("待归档源文件无效")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as origin, os.fdopen(descriptor, "wb") as target:
            while block := origin.read(1024 * 1024):
                if cancellation:
                    cancellation.check()
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_identity(item: str) -> str:
    if item.casefold().startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(item)
        return urllib.parse.urlunsplit(
            (parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path or "/", parsed.query, "")
        )
    return str(Path(item).expanduser().resolve())


def _source_id(item: str) -> str:
    return "desktop-" + sha256_text(_source_identity(item))[:20]


def _cleanup_warnings_from_exception(error: BaseException) -> list[str]:
    notes = getattr(error, "__notes__", ())
    if any("临时清理失败" in str(note) for note in notes):
        return [_PUBLIC_CLEANUP_WARNING]
    return []


def _public_failure(error: BaseException) -> tuple[str, list[str]]:
    primary = str(error) or type(error).__name__
    warnings = _cleanup_warnings_from_exception(error)
    if warnings:
        return f"{primary}；{warnings[0]}", warnings
    return primary, []


class IngestionService:
    """First-party multimodal ingestion pipeline used by the desktop application.

    It intentionally has no Codex or Obsidian dependency. Every accepted source becomes a
    reproducible archive package, searchable SQLite chunks, and (optionally) a Markdown
    Source Note under the application's own data directory.
    """

    def __init__(
        self,
        paths: ProductPaths,
        config: AppConfig | None = None,
        settings: DesktopSettings | None = None,
    ) -> None:
        self.paths = paths.ensure()
        self.settings = settings or DesktopSettings.load(paths.settings)
        self.config = config or AppConfig.from_env(paths.database)
        self._cleanup_registry = TemporaryCleanupRegistry(self.paths.cache)
        self._pending_cleanup_warning = self._retry_temporary_cleanup()

    def ingest(
        self,
        items: Iterable[str | Path],
        *,
        progress: ProgressCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> IngestionSummary:
        values = [str(item).strip() for item in items if str(item).strip()]
        if not values:
            raise ValueError("请至少选择一个文件或网页地址")
        token = cancellation or CancellationToken()
        started = perf_counter()
        summary = IngestionSummary()
        retry_warning = self._retry_temporary_cleanup()
        cleanup_warnings = [retry_warning] if retry_warning else []
        packages = self._detect_packages(values)
        embedding = build_embedding_provider(self.config)
        with KnowledgeDatabase(self.paths.database) as database:
            indexing = IndexingService(database, embedding)
            vision = MultimodalInterpreter(
                self.config,
                enabled=self.settings.enable_cloud_vision,
                max_images=self.settings.vision_max_images,
            )
            for item in values:
                item_started = perf_counter()
                if token.cancelled:
                    summary.results.append(IngestionResult(item=item, status="cancelled", error="任务已取消"))
                    break
                try:
                    package_id, members = packages[item]
                    glossary_terms = TranscriptRepository(database).context_terms(
                        knowledge_space_id=self.settings.asr_knowledge_space_id,
                        source_id=_source_id(item),
                    )
                    effective_settings = replace(
                        self.settings,
                        asr_context_terms=list(dict.fromkeys([
                            *self.settings.asr_context_terms,
                            *glossary_terms,
                        ]))[:200],
                    )
                    result = self._ingest_one(
                        item, indexing, vision, token, progress,
                        package_id=package_id,
                        package_members=members,
                        settings=effective_settings,
                    )
                except CancelledError as exc:
                    error, warnings = _public_failure(exc)
                    result = IngestionResult(
                        item=item,
                        status="cancelled",
                        error=error,
                        warnings=warnings,
                    )
                except QualityGateError as exc:
                    error, warnings = _public_failure(exc)
                    result = IngestionResult(
                        item=item,
                        status="failed",
                        error=error,
                        warnings=warnings,
                        quality_report=exc.report.to_dict(),
                    )
                    self._emit(progress, item, "failed", 100, f"质检未通过：{result.error}")
                except (OSError, UnicodeError, ValueError, RuntimeError, MissingExtractorDependency) as exc:
                    error, warnings = _public_failure(exc)
                    result = IngestionResult(
                        item=item,
                        status="failed",
                        error=error,
                        warnings=warnings,
                    )
                    self._emit(progress, item, "failed", 100, f"导入失败：{result.error}")
                for warning in cleanup_warnings:
                    if warning not in result.warnings:
                        result.warnings.append(warning)
                result.duration_ms = round((perf_counter() - item_started) * 1000, 3)
                summary.results.append(result)
                if result.status == "cancelled":
                    break
        if len(summary.results) < len(values):
            processed = {item.item for item in summary.results}
            for item in values:
                if item not in processed:
                    package_id, _ = packages[item]
                    summary.results.append(
                        IngestionResult(
                            item=item,
                            status="cancelled",
                            package_id=package_id,
                            error="任务已取消",
                        )
                    )
        self._write_package_manifests(summary, packages)
        summary.completed_at = utcnow_iso()
        summary.duration_ms = round((perf_counter() - started) * 1000, 3)
        return summary

    def _ingest_one(
        self,
        item: str,
        indexing: IndexingService,
        vision: MultimodalInterpreter,
        cancellation: CancellationToken,
        progress: ProgressCallback | None,
        *,
        package_id: str,
        package_members: list[str],
        settings: DesktopSettings | None = None,
    ) -> IngestionResult:
        cancellation.check()
        self._emit(progress, item, "preparing", 3, "正在识别资料类型")

        def extraction_progress(message: str) -> None:
            self._emit(progress, item, "extracting", 24, message)

        context = ExtractionContext(
            paths=self.paths,
            settings=settings or self.settings,
            cancellation=cancellation,
            vision=vision,
            progress=extraction_progress,
            cleanup_registry=self._cleanup_registry,
        )
        try:
            if item.casefold().startswith(("http://", "https://")):
                extracted = url_extractor_for(item).extract(item, context)
            else:
                path = Path(item).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"文件不存在：{path}")
                if path.stat().st_size > 8 * 1024 * 1024 * 1024:
                    raise ValueError("单个文件不能超过 8GB")
                extractor = extractor_for(path)
                if extractor is None:
                    raise ValueError(f"暂不支持该格式：{path.suffix or '无扩展名'}")
                extracted = extractor.extract(path, context)
            result = self._finish_ingestion(
                item=item,
                extracted=extracted,
                context=context,
                indexing=indexing,
                cancellation=cancellation,
                progress=progress,
                package_id=package_id,
                package_members=package_members,
            )
        except BaseException as error:
            try:
                context.cleanup_owned_temporary_paths()
            except OSError as cleanup_error:
                error.add_note(f"附加诊断：{cleanup_error}")
            raise
        else:
            try:
                context.cleanup_owned_temporary_paths()
            except OSError as cleanup_error:
                warning = (
                    "临时清理失败"
                    if str(cleanup_error) == "临时清理失败"
                    else _PUBLIC_CLEANUP_WARNING
                )
                result.warnings.append(f"导入已完成，但{warning}")
            return result

    def _retry_temporary_cleanup(self) -> str | None:
        try:
            report = self._cleanup_registry.retry_pending()
        except OSError:
            self._pending_cleanup_warning = _PUBLIC_CLEANUP_WARNING
            return _PUBLIC_CLEANUP_WARNING
        self._pending_cleanup_warning = report.warning
        return report.warning

    def _finish_ingestion(
        self,
        *,
        item: str,
        extracted: ExtractionResult,
        context: ExtractionContext,
        indexing: IndexingService,
        cancellation: CancellationToken,
        progress: ProgressCallback | None,
        package_id: str,
        package_members: list[str],
    ) -> IngestionResult:
        cancellation.check()
        if not extracted.segments or extracted.extracted_characters == 0:
            raise RuntimeError("资料中没有提取到可检索内容")

        self._emit(progress, item, "validating", 44, "正在执行入库完整性与真实性检查")
        quality = evaluate_extraction(extracted)
        if not quality.accepted:
            failures = "；".join(check.detail for check in quality.checks if check.status == "fail")
            raise QualityGateError(failures or "资料质量未达到入库标准", quality)
        extracted.metadata["quality_report"] = quality.to_dict()

        transcript = None
        transcript_gate_blocked = False
        if extracted.transcript_data is not None:
            try:
                transcript = transcript_from_dict(extracted.transcript_data)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Transcript V2 事实层无效，已停止入库：{str(exc)[:160]}") from exc
            transcript_gate_blocked = bool(
                context.settings.transcript_quality_gate
                and transcript.quality.status != "pass"
            )

        self._emit(progress, item, "archiving", 52, "正在归档原始资料与解析结果")
        source_id = _source_id(item)
        temporary_source = context.owns_temporary_path(extracted.source_path)
        derived = self._publish_derived_artifacts(
            extracted,
            context=context,
            cancellation=cancellation,
        )
        try:
            if self.settings.archive_originals:
                persisted = self._archive(extracted, source_id, cancellation=cancellation)
            else:
                if temporary_source:
                    persisted = self._retain_unarchived_source(
                        extracted,
                        source_id=source_id,
                        cancellation=cancellation,
                    )
                else:
                    persisted = _PersistedEvidence(None, extracted.source_path)
        except BaseException as error:
            try:
                self._rollback_new_derived_artifacts(derived)
            except OSError as rollback_error:
                error.add_note(f"附加诊断：新派生产物回滚失败：{rollback_error}")
            raise
        package = persisted.package
        owned_source = persisted.source_path
        if temporary_source and owned_source:
            extracted.source_path = owned_source
        source = SourceReference(
            source_id=source_id,
            media_type=extracted.media_type,
            title=extracted.title,
            original_uri=extracted.original_uri,
            local_path=str(owned_source) if owned_source else None,
            checksum=extracted.checksum,
        )
        metadata = {
            key: value for key, value in extracted.metadata.items() if key != "snapshot_html"
        }
        metadata.update(
            {
                "ingestion_origin": "desktop",
                "product": PRODUCT_NAME,
                "source_identity": _source_identity(item),
                "source_package_id": package_id,
                "source_package_members": package_members,
                "archive_path": str(package) if package else None,
                "extracted_characters": extracted.extracted_characters,
                "warnings": extracted.warnings,
            }
        )
        document = KnowledgeDocument(
            source_id=source_id,
            title=extracted.title,
            media_type=extracted.media_type,
            segments=extracted.segments,
            source=source,
            collections=["本地知识库"],
            tags=[f"来源/{extracted.media_type}"],
            metadata=metadata,
        )

        try:
            # The permanent evidence is provisional until index_document returns.
            # Keeping this cancellation check inside the same rollback boundary
            # prevents an index-before-cancel race from leaking a new raw source.
            cancellation.check()
            if transcript_gate_blocked:
                self._emit(
                    progress,
                    item,
                    "indexing",
                    68,
                    "转写质量需要复核，正在保存事实层并跳过全文与向量索引",
                )
                indexed = indexing.persist_document_without_search_index(document)
            else:
                self._emit(progress, item, "indexing", 68, "正在分块、建立全文与向量索引")
                indexed = indexing.index_document(document)
        except BaseException as error:
            try:
                self._rollback_new_evidence(persisted)
            except OSError as rollback_error:
                error.add_note(f"附加诊断：新证据回滚失败：{rollback_error}")
            try:
                self._rollback_new_derived_artifacts(derived)
            except OSError as rollback_error:
                error.add_note(f"附加诊断：新派生产物回滚失败：{rollback_error}")
            raise
        if indexed.status == "duplicate":
            # A duplicate points at another source's database row, so paths
            # created for this attempted source have no owner.  By contrast an
            # unchanged result now proves all persisted locators/facets match;
            # a newly recreated same-path file may be repairing missing evidence
            # and must be kept.
            cleanup_errors: list[OSError] = []
            try:
                self._rollback_new_evidence(persisted)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
            try:
                self._rollback_new_derived_artifacts(derived)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise OSError(
                    f"重复资料已识别，但 {len(cleanup_errors)} 组本次冗余文件清理失败"
                ) from cleanup_errors[0]
            if persisted.rollback_path is not None:
                package = None
                owned_source = None
            derived = _PersistedDerivedArtifacts()
        warnings = list(extracted.warnings)
        transcript_run_id: str | None = None
        if transcript is not None and (
            indexed.status != "duplicate" or transcript_gate_blocked
        ):
            try:
                transcription_metadata = extracted.metadata.get("transcription")
                transcript_artifact: str | None = None
                if indexed.status != "duplicate" and isinstance(transcription_metadata, dict):
                    artifacts = transcription_metadata.get("artifacts")
                    if isinstance(artifacts, dict) and artifacts.get("v2"):
                        transcript_artifact = str(artifacts["v2"])
                TranscriptRepository(indexing.database).save_transcript(
                    transcript,
                    document_id=indexed.document_id,
                    transcript_path=transcript_artifact,
                )
                transcript_run_id = transcript.run.id
                if transcript_gate_blocked:
                    if indexed.status == "duplicate":
                        warnings.append(
                            "重复资料已有独立的检索记录；本次待复核转写事实已保存，"
                            "且未新建全文或向量索引。"
                        )
                    else:
                        warnings.append(
                            "转写质量需要人工复核，资料与转写事实已保存，但未生成全文或向量索引；"
                            "校订并确认质量后可加入问答。"
                        )
            except (OSError, ValueError, RuntimeError) as exc:
                if indexed.status != "duplicate":
                    indexing.remove_document_search_index(indexed.document_id)
                    indexing.database.set_document_enabled(indexed.document_id, False)
                warnings.append(
                    f"转写事实层保存失败，已移除检索索引并保留资料：{str(exc)[:160]}"
                )

        note_path: Path | None = None
        if self.settings.create_source_notes and indexed.status in {"created", "updated"}:
            self._emit(progress, item, "noting", 84, "正在生成本地知识笔记")
            synthesis = ""
            if self.settings.auto_synthesize_notes:
                try:
                    synthesis = self._synthesize(extracted)
                except (ValueError, RuntimeError, OSError) as exc:
                    warnings.append(f"AI 知识提炼未完成：{str(exc)[:160]}")
            note_path = self._write_source_note(extracted, source, indexed.document_id, package, synthesis)

        self._emit(progress, item, "complete", 100, f"已{self._status_label(indexed.status)}：{extracted.title}")
        return IngestionResult(
            item=item,
            title=extracted.title,
            media_type=extracted.media_type,
            status=indexed.status,
            document_id=indexed.document_id,
            source_id=source_id,
            package_id=package_id,
            archive_path=str(package) if package else None,
            note_path=str(note_path) if note_path else None,
            chunks=indexed.created_chunks + indexed.updated_chunks + indexed.unchanged_chunks,
            extracted_characters=extracted.extracted_characters,
            warnings=warnings,
            quality_report=quality.to_dict(),
            transcript_run_id=transcript_run_id,
        )

    @staticmethod
    def _package_key(item: str) -> str:
        if item.casefold().startswith(("http://", "https://")):
            parsed = urllib.parse.urlsplit(item)
            stem = Path(parsed.path).stem or parsed.netloc
        else:
            stem = Path(item).stem
        normalized = stem.casefold()
        normalized = re.sub(
            r"(?:[\s._-]*(?:pptx?|pdf|slides?|deck|recording|audio|video|\u8bfe\u4ef6|\u5f55\u97f3|\u5f55\u50cf|\u89c6\u9891|\u97f3\u9891|\u6587\u7a3f|\u8d44\u6599|\u5bfc\u51fa\u7248|\u6700\u7ec8\u7248))+$",
            "",
            normalized,
        )
        normalized = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", normalized)
        return normalized or sha256_text(_source_identity(item))[:16]

    @classmethod
    def _detect_packages(cls, items: list[str]) -> dict[str, tuple[str, list[str]]]:
        groups: dict[str, list[str]] = {}
        for item in items:
            groups.setdefault(cls._package_key(item), []).append(item)
        result: dict[str, tuple[str, list[str]]] = {}
        for key, members in groups.items():
            package_id = "package-" + sha256_text(key)[:20]
            for item in members:
                result[item] = (package_id, list(members))
        return result

    def _write_package_manifests(
        self,
        summary: IngestionSummary,
        packages: dict[str, tuple[str, list[str]]],
    ) -> None:
        by_package: dict[str, list[IngestionResult]] = {}
        for result in summary.results:
            package_id = result.package_id or packages.get(result.item, (None, []))[0]
            if package_id:
                by_package.setdefault(package_id, []).append(result)
        for package_id, results in by_package.items():
            payload = {
                "format": "ai-jingjing-source-package-manifest-v1",
                "package_id": package_id,
                "updated_at": utcnow_iso(),
                "multimodal": len(results) > 1,
                "members": [result.to_dict() for result in results],
            }
            _atomic_json(self.paths.archive / "source-packages" / package_id / "manifest.json", payload)

    def _publish_derived_artifacts(
        self,
        extracted: ExtractionResult,
        *,
        context: ExtractionContext,
        cancellation: CancellationToken,
    ) -> _PersistedDerivedArtifacts:
        """Publish context-owned transcripts and frames after the quality gate.

        Every destination contains the file's real SHA-256 digest and is created
        without replacement.  The returned list contains only files created by this
        attempt, so an indexing failure can never delete a prior successful artifact.
        """

        candidates: dict[Path, str] = {}

        def add_candidate(raw: object, kind: str) -> None:
            if not raw:
                return
            path = Path(str(raw)).resolve()
            if not context.owns_temporary_path(path):
                return
            if extracted.source_path and path == extracted.source_path.resolve():
                return
            if path.is_file() and not path.is_symlink():
                candidates.setdefault(path, kind)

        add_candidate(extracted.transcript_path, "transcript")
        transcription = extracted.metadata.get("transcription")
        if isinstance(transcription, dict):
            artifacts = transcription.get("artifacts")
            if isinstance(artifacts, dict):
                for raw in artifacts.values():
                    add_candidate(raw, "transcript")
        for asset in extracted.retained_assets:
            add_candidate(asset, "frame")
        for segment in extracted.segments:
            add_candidate(segment.asset, "frame")

        if not candidates:
            return _PersistedDerivedArtifacts()

        mapping: dict[Path, Path] = {}
        rollback_paths: list[Path] = []
        safe_title = safe_stem(extracted.title)
        try:
            for source, kind in sorted(candidates.items(), key=lambda item: str(item[0])):
                cancellation.check()
                digest = self._checksum_if_file(source)
                if not digest:
                    raise OSError("待发布派生产物不存在或无法校验")
                suffix = source.suffix.casefold()
                if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                    suffix = ".bin"
                if kind == "transcript":
                    destination = self.paths.transcripts / f"{safe_title}-{digest[:24]}{suffix}"
                else:
                    destination = self.paths.assets / "frames" / f"{safe_title}-{digest[:24]}{suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                created = self._publish_content_addressed_file(
                    source,
                    destination,
                    digest,
                    cancellation=cancellation,
                )
                mapping[source] = destination
                if created:
                    rollback_paths.append(destination)
        except BaseException as error:
            partial = _PersistedDerivedArtifacts(tuple(rollback_paths))
            try:
                self._rollback_new_derived_artifacts(partial)
            except OSError as rollback_error:
                error.add_note(f"附加诊断：未完成派生产物回滚失败：{rollback_error}")
            raise

        def published(raw: object) -> Path | None:
            if not raw:
                return None
            try:
                candidate = Path(str(raw)).resolve()
            except (OSError, RuntimeError, ValueError):
                return None
            return mapping.get(candidate)

        replacement = published(extracted.transcript_path)
        if replacement is not None:
            extracted.transcript_path = replacement
        extracted.retained_assets = [
            published(asset) or asset for asset in extracted.retained_assets
        ]
        for segment in extracted.segments:
            replacement = published(segment.asset)
            if replacement is not None:
                segment.asset = str(replacement)
        if isinstance(transcription, dict):
            artifacts = transcription.get("artifacts")
            if isinstance(artifacts, dict):
                transcription["artifacts"] = {
                    str(name): str(published(raw) or raw)
                    for name, raw in artifacts.items()
                }
        extracted.metadata["derived_artifacts"] = [
            str(path) for path in sorted(set(mapping.values()), key=str)
        ]
        return _PersistedDerivedArtifacts(tuple(rollback_paths))

    def _rollback_new_derived_artifacts(
        self, persisted: _PersistedDerivedArtifacts
    ) -> None:
        roots = (self.paths.transcripts.resolve(), (self.paths.assets / "frames").resolve())
        errors: list[OSError] = []
        for target in reversed(persisted.rollback_paths):
            resolved = target.resolve(strict=False)
            if not any(resolved != root and resolved.is_relative_to(root) for root in roots):
                raise OSError("拒绝回滚派生产物目录之外的路径")
            try:
                if target.is_symlink() or target.is_file():
                    target.unlink(missing_ok=True)
            except OSError as error:
                errors.append(error)
        frames_root = self.paths.assets / "frames"
        try:
            if frames_root.is_dir() and not any(frames_root.iterdir()):
                frames_root.rmdir()
        except OSError:
            pass
        if errors:
            raise OSError(f"{len(errors)} 个新派生产物回滚失败") from errors[0]

    def _archive(
        self,
        extracted: ExtractionResult,
        source_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> _PersistedEvidence:
        evidence_digest = self._evidence_digest(extracted)
        parse_digest = self._parse_digest(extracted)
        day = datetime.now().astimezone().strftime("%Y%m%d")
        package = (
            self.paths.archive
            / day[:4]
            / day[4:6]
            / (
                f"{day}_{safe_stem(extracted.title)}_{source_id[-8:]}_"
                f"{evidence_digest[:16]}_{parse_digest[:16]}"
            )
        )
        package.parent.mkdir(parents=True, exist_ok=True)
        try:
            package.mkdir()
        except FileExistsError:
            owned_source = self._existing_archived_source(
                extracted,
                package,
                expected_digest=evidence_digest,
                expected_parse_digest=parse_digest,
            )
            self._mark_persisted_source(extracted, owned_source)
            return _PersistedEvidence(package, owned_source)

        try:
            source_dir = package / "source"
            source_dir.mkdir()
            if extracted.source_path:
                destination = source_dir / extracted.source_path.name
                _atomic_copy_file(
                    extracted.source_path,
                    destination,
                    cancellation=cancellation,
                )
                owned_source: Path | None = destination
                self._mark_persisted_source(extracted, destination)
            else:
                owned_source = None
            snapshot = extracted.metadata.get("snapshot_html")
            if isinstance(snapshot, str):
                archived_page = source_dir / "page.html"
                _atomic_text(archived_page, snapshot)
                owned_source = owned_source or archived_page
            segments = [
                {
                    "id": segment.id,
                    "sequence": segment.sequence,
                    "modality": segment.modality,
                    "text": segment.text,
                    "description": segment.description,
                    "location": segment.location,
                    "heading_path": segment.heading_path,
                    "asset": segment.asset,
                    "metadata": segment.metadata,
                }
                for segment in extracted.segments
            ]
            bundle = {
                "format": "ai-jingjing-source-package-v1",
                "created_at": utcnow_iso(),
                "source_id": source_id,
                "title": extracted.title,
                "media_type": extracted.media_type,
                "checksum": extracted.checksum,
                "parse_digest": parse_digest,
                "original_uri": extracted.original_uri,
                "metadata": {
                    key: value for key, value in extracted.metadata.items() if key != "snapshot_html"
                },
                "warnings": extracted.warnings,
                "assets": [str(path) for path in extracted.retained_assets],
                "transcript": str(extracted.transcript_path) if extracted.transcript_path else None,
                "segments": segments,
            }
            bundle_path = package / "bundle.json"
            _atomic_json(bundle_path, bundle)
            owned_source = owned_source or bundle_path
            return _PersistedEvidence(package, owned_source, package)
        except BaseException as error:
            try:
                shutil.rmtree(package)
            except OSError as cleanup_error:
                error.add_note(f"附加诊断：未完成归档包清理失败：{cleanup_error}")
            raise

    def _parse_digest(self, extracted: ExtractionResult) -> str:
        """Fingerprint the logical parse independently from raw source bytes.

        Storage paths are represented by the bytes they identify, so a random
        cache directory cannot create a new package while a changed transcript,
        OCR result, keyframe, or parser metadata always can.
        """

        def portable(value: object) -> object:
            if isinstance(value, dict):
                return {
                    str(key): portable(child)
                    for key, child in sorted(value.items(), key=lambda item: str(item[0]))
                    if str(key) not in {"created_at", "updated_at"}
                }
            if isinstance(value, set):
                return [portable(child) for child in sorted(value, key=repr)]
            if isinstance(value, (list, tuple)):
                return [portable(child) for child in value]
            if isinstance(value, Path):
                candidate = value
            elif isinstance(value, str):
                try:
                    candidate = Path(value)
                except (OSError, ValueError):
                    return value
            else:
                return value
            try:
                if candidate.is_file() and not candidate.is_symlink():
                    digest = self._checksum_if_file(candidate)
                    if digest:
                        return {
                            "file_sha256": digest,
                            "suffix": candidate.suffix.casefold(),
                        }
                if candidate.is_absolute():
                    return {"local_name": candidate.name}
            except (OSError, RuntimeError, ValueError):
                if candidate.is_absolute():
                    return {"local_name": candidate.name}
            return str(value)

        payload = {
            "title": extracted.title,
            "media_type": extracted.media_type,
            "original_uri": extracted.original_uri,
            "segments": [
                {
                    "id": segment.id,
                    "sequence": segment.sequence,
                    "modality": segment.modality,
                    "text": segment.text,
                    "description": segment.description,
                    "location": portable(segment.location),
                    "heading_path": segment.heading_path,
                    "asset": portable(segment.asset),
                    "metadata": portable(segment.metadata),
                }
                for segment in sorted(
                    extracted.segments,
                    key=lambda item: (item.sequence, item.id),
                )
            ],
            "metadata": portable(extracted.metadata),
            "warnings": extracted.warnings,
            "transcript": portable(extracted.transcript_path),
            "assets": portable(extracted.retained_assets),
        }
        return sha256_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )

    def _retain_unarchived_source(
        self,
        extracted: ExtractionResult,
        *,
        source_id: str,
        cancellation: CancellationToken,
    ) -> _PersistedEvidence:
        source = extracted.source_path
        if source is None or source.is_symlink() or not source.is_file():
            raise OSError("临时公开媒体源文件无效")
        evidence_digest = self._evidence_digest(extracted)
        suffix = source.suffix.casefold()
        identity = str(
            extracted.metadata.get("temporary_source_identity")
            or source_id.removeprefix("desktop-")
        )
        identity = re.sub(r"[^a-z0-9_-]", "", identity.casefold())[:64] or source_id[-20:]
        destination = (
            self.paths.assets
            / "public-platform"
            / "sources"
            / f"{identity}-{evidence_digest[:16]}{suffix}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        cancellation.check()
        try:
            created = self._publish_content_addressed_file(
                source,
                destination,
                evidence_digest,
                cancellation=cancellation,
            )
            self._mark_persisted_source(extracted, destination)
            return _PersistedEvidence(
                None,
                destination,
                destination if created else None,
            )
        except BaseException:
            if "created" in locals() and created:
                destination.unlink(missing_ok=True)
            raise

    def _evidence_digest(self, extracted: ExtractionResult) -> str:
        """Return a real content digest suitable for immutable evidence paths."""

        if extracted.source_path and extracted.source_path.is_file():
            digest = self._checksum_if_file(extracted.source_path)
            if digest:
                extracted.checksum = digest
                return digest
        payload = {
            "title": extracted.title,
            "media_type": extracted.media_type,
            "original_uri": extracted.original_uri,
            "checksum": extracted.checksum,
            "snapshot_html": extracted.metadata.get("snapshot_html"),
            "segments": [
                {
                    "sequence": segment.sequence,
                    "modality": segment.modality,
                    "text": segment.text,
                    "description": segment.description,
                    "location": segment.location,
                }
                for segment in extracted.segments
            ],
        }
        digest = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        extracted.checksum = digest
        return digest

    def _publish_content_addressed_file(
        self,
        source: Path,
        destination: Path,
        expected_digest: str,
        *,
        cancellation: CancellationToken,
    ) -> bool:
        """Publish without ever replacing an existing content-addressed target."""

        if destination.exists():
            if self._checksum_if_file(destination) != expected_digest:
                raise OSError("内容寻址目标已存在但校验不一致，已拒绝覆盖")
            return False
        try:
            # On the normal single-volume layout a hard link gives us an atomic,
            # zero-copy publish. The disposable cache link is removed later.
            os.link(source, destination)
            return True
        except FileExistsError:
            if self._checksum_if_file(destination) != expected_digest:
                raise OSError("内容寻址目标发生并发冲突，已拒绝覆盖")
            return False
        except OSError as error:
            fallback_errors = {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                getattr(errno, "ENOTSUP", errno.EPERM),
            }
            if error.errno not in fallback_errors:
                raise

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as origin, os.fdopen(descriptor, "wb") as target:
                while block := origin.read(1024 * 1024):
                    cancellation.check()
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            try:
                os.link(temporary, destination)
                return True
            except FileExistsError:
                if self._checksum_if_file(destination) != expected_digest:
                    raise OSError("内容寻址目标发生并发冲突，已拒绝覆盖")
                return False
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _mark_persisted_source(extracted: ExtractionResult, source: Path) -> None:
        extracted.metadata.pop("temporary_source_owned_by", None)
        extracted.metadata.pop("temporary_source_identity", None)
        extracted.metadata["source_media"] = str(source)
        extracted.metadata["source_media_bytes"] = source.stat().st_size
        if extracted.metadata.get("source_subtitle"):
            extracted.metadata["source_subtitle"] = str(source)

    def _existing_archived_source(
        self,
        extracted: ExtractionResult,
        package: Path,
        *,
        expected_digest: str,
        expected_parse_digest: str,
    ) -> Path:
        bundle = package / "bundle.json"
        if not bundle.is_file():
            raise OSError("内容寻址归档包不完整，已拒绝覆盖")
        try:
            archived = json.loads(bundle.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise OSError("内容寻址归档包清单损坏，已拒绝复用") from error
        if (
            not isinstance(archived, dict)
            or archived.get("checksum") != expected_digest
            or archived.get("parse_digest") != expected_parse_digest
        ):
            raise OSError("内容寻址归档包校验不一致，已拒绝复用")
        source_dir = package / "source"
        if extracted.source_path:
            exact = source_dir / extracted.source_path.name
            if exact.is_file():
                candidate = exact
            else:
                try:
                    candidates = [path for path in source_dir.iterdir() if path.is_file()]
                except OSError as error:
                    raise OSError("内容寻址归档包缺少原始证据，已拒绝复用") from error
                if len(candidates) != 1:
                    raise OSError("内容寻址归档包缺少原始证据，已拒绝复用")
                candidate = candidates[0]
            if self._checksum_if_file(candidate) != expected_digest:
                raise OSError("内容寻址归档源文件校验失败，已拒绝复用")
            return candidate
        archived_page = source_dir / "page.html"
        snapshot = extracted.metadata.get("snapshot_html")
        if isinstance(snapshot, str):
            try:
                archived_snapshot = archived_page.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise OSError("内容寻址网页快照损坏，已拒绝复用") from error
            if archived_snapshot != snapshot:
                raise OSError("内容寻址网页快照校验失败，已拒绝复用")
        return archived_page if archived_page.is_file() else bundle

    def _rollback_new_evidence(self, persisted: _PersistedEvidence) -> None:
        target = persisted.rollback_path
        if target is None:
            return
        root = self.paths.archive if persisted.package else (
            self.paths.assets / "public-platform" / "sources"
        )
        resolved_root = root.resolve()
        resolved_target = target.resolve(strict=False)
        if resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
            raise OSError("拒绝回滚产品证据目录之外的路径")
        if target.is_symlink() or target.is_file():
            target.unlink(missing_ok=True)
        elif target.is_dir():
            shutil.rmtree(target)

    @staticmethod
    def _checksum_if_file(path: Path) -> str | None:
        if not path.is_file():
            return None
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _owned_source_path(extracted: ExtractionResult, package: Path | None) -> Path | None:
        if package:
            if extracted.source_path:
                archived_source = package / "source" / extracted.source_path.name
                if archived_source.is_file():
                    return archived_source
            archived_page = package / "source" / "page.html"
            if archived_page.is_file():
                return archived_page
            # Every package has a self-contained evidence manifest.  Returning
            # it is safer than persisting a guessed path for URL extractors that
            # intentionally have no HTML snapshot.
            archived_bundle = package / "bundle.json"
            if archived_bundle.is_file():
                return archived_bundle
        return extracted.source_path

    def _synthesize(self, extracted: ExtractionResult) -> str:
        provider = build_answer_provider(self.config, model_id=self._deepseek_synthesis_model_id())
        located_segments = [
            self._located_segment(segment)
            for segment in extracted.segments
            if segment.retrieval_text
        ]
        content = "\n\n".join(located_segments)[:80_000]
        system = (
            "你是 AI知识库-AI静静 的知识整理引擎。只能根据用户给定的原始资料，"
            "用简体中文生成派生整理层 Markdown。这一层永远不是原始转写事实："
            "不得改写、纠正、覆盖或补齐原始识别文字，不得把意译写成原话。"
            "保留关键数字、条件、公式、流程和不确定性，不得臆测。"
            "已被资料明确支持的内容才能放入‘已确认事实’；未被证实的因果、"
            "意图、概括和建议必须放入‘推测与待验证’并明示不确定性；"
            "不同说话人或段落的冲突说法必须并列，不得自行调和。"
            "每条关键事实、争议说法、结论、决策和来自资料的行动项，"
            "都必须在该条末尾原样复用原文段前的页码、幻灯片号或时间戳定位；"
            "不得编造页码或时间戳。原文没有页码/时间戳时，只能复用其【片段】定位，"
            "并且不得将无法定位的附加推断写成已确认事实。"
            "不要输出一级标题或 YAML。"
        )
        user = (
            f"资料标题：{extracted.title}\n类型：{extracted.media_type}\n\n"
            "必须严格按下列二级标题和顺序输出，一个也不能省略：\n"
            + "\n".join(_SYNTHESIS_HEADINGS)
            + "\n若某层没有材料，写‘- 原始资料未提供。’；不得为填满栏目而推断。"
            "行动项只记录资料明确提出的任务；责任人、时限或验收标准未说明时标为‘未说明’。\n\n"
            "原始解析内容（每段开头的方括号是唯一可用定位）：\n" + content
        )
        markdown = provider.generate(
            AnswerRequest("整理这份资料", system, user, [])
        ).markdown.strip()
        positions = [markdown.find(heading) for heading in _SYNTHESIS_HEADINGS]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            raise RuntimeError("模型返回的知识整理缺少必需分层，已拒绝写入 Source Note")
        allowed_citations = {
            item.splitlines()[0]
            for item in located_segments
            if item.startswith("[") and item.splitlines()
        }
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("- ", "* ")) or "原始资料未提供" in stripped:
                continue
            citations = set(re.findall(r"\[[^\]\n]+\]", stripped))
            if not citations:
                raise RuntimeError("模型返回的关键条目缺少原始定位，已拒绝写入 Source Note")
            fabricated = citations - allowed_citations
            if fabricated:
                raise RuntimeError("模型返回了原始资料中不存在的定位，已拒绝写入 Source Note")
        return markdown

    def _deepseek_synthesis_model_id(self) -> str:
        """Resolve ingestion synthesis to DeepSeek without any Codex fallback."""

        provider = next(
            (item for item in self.config.qa_compatible_providers if item.id == "deepseek"),
            None,
        )
        if provider is None or not provider.models:
            raise ValueError("AI 知识提炼需要先在设置中配置 DeepSeek API Key")
        model = next(
            (name for name in provider.models if name == "deepseek-v4-flash"),
            provider.models[0],
        )
        return f"compatible::deepseek::{model}"

    @staticmethod
    def _located_segment(segment) -> str:
        location = []
        if segment.location.get("page") is not None:
            location.append(f"P{segment.location['page']}")
        if segment.location.get("slide") is not None:
            location.append(f"S{segment.location['slide']}")
        if segment.location.get("timestamp_start") is not None:
            try:
                start = IngestionService._format_timestamp(
                    float(segment.location["timestamp_start"])
                )
                end_value = segment.location.get("timestamp_end")
                if end_value is not None:
                    end = IngestionService._format_timestamp(float(end_value))
                    location.append(f"{start}–{end}")
                else:
                    location.append(start)
            except (TypeError, ValueError, OverflowError):
                pass
        if not location:
            location.append(f"片段:{segment.id}")
        return f"[{' / '.join(location)}]\n{segment.retrieval_text}"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        total_milliseconds = max(0, round(seconds * 1000))
        total_seconds, milliseconds = divmod(total_milliseconds, 1000)
        minutes, second = divmod(total_seconds, 60)
        hours, minute = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minute:02d}:{second:02d}.{milliseconds:03d}"
        return f"{minute:02d}:{second:02d}.{milliseconds:03d}"

    @staticmethod
    def _transcript_note_metadata(extracted: ExtractionResult) -> dict[str, str]:
        """Resolve Source Note provenance from the canonical Transcript V2 payload."""

        transcript = extracted.transcript_data if isinstance(extracted.transcript_data, dict) else {}
        transcription = extracted.metadata.get("transcription")
        transcription = transcription if isinstance(transcription, dict) else {}
        run = transcript.get("run")
        run = run if isinstance(run, dict) else {}
        quality = transcript.get("quality")
        quality = quality if isinstance(quality, dict) else {}
        artifacts = transcription.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}

        values: dict[str, str] = {}
        if artifacts.get("v2"):
            values["v2_path"] = str(artifacts["v2"])
        run_id = run.get("id") or transcription.get("v2_run_id")
        if run_id:
            values["run_id"] = str(run_id)
        quality_status = quality.get("status")
        transcription_quality = transcription.get("quality")
        if not quality_status and isinstance(transcription_quality, dict):
            quality_status = transcription_quality.get("status")
        if quality_status:
            values["quality"] = str(quality_status)

        profile = run.get("profile") or transcription.get("profile")
        provider = (
            run.get("provider")
            or transcription.get("provider")
            or transcription.get("engine")
        )
        model = run.get("model") or transcription.get("model")
        diarization = run.get("diarization_provider")
        route = [str(item) for item in (profile, provider, model) if item]
        if diarization:
            route.append(f"说话人:{diarization}")
        if route:
            values["model_route"] = " -> ".join(route)
        return values

    def _write_source_note(
        self,
        extracted: ExtractionResult,
        source: SourceReference,
        document_id: str,
        package: Path | None,
        synthesis: str,
    ) -> Path:
        note = (
            self.paths.notes
            / "Sources"
            / f"{safe_stem(extracted.title)}--{source.source_id[-8:]}.md"
        )
        transcript_metadata = self._transcript_note_metadata(extracted)
        frontmatter = [
            "---",
            f"title: {json.dumps(extracted.title, ensure_ascii=False)}",
            'type: "source-note"',
            f"document_id: {json.dumps(document_id)}",
            f"source_id: {json.dumps(source.source_id)}",
            f"media_type: {json.dumps(extracted.media_type)}",
            f"checksum: {json.dumps(extracted.checksum)}",
            f"updated_at: {json.dumps(utcnow_iso())}",
            *(
                [
                    "transcript_v2_path: "
                    + json.dumps(transcript_metadata["v2_path"], ensure_ascii=False)
                ]
                if transcript_metadata.get("v2_path") else []
            ),
            *(
                [
                    "transcript_run_id: "
                    + json.dumps(transcript_metadata["run_id"], ensure_ascii=False)
                ]
                if transcript_metadata.get("run_id") else []
            ),
            *(
                [
                    "transcript_quality: "
                    + json.dumps(transcript_metadata["quality"], ensure_ascii=False)
                ]
                if transcript_metadata.get("quality") else []
            ),
            *(
                [
                    "transcript_model_route: "
                    + json.dumps(transcript_metadata["model_route"], ensure_ascii=False)
                ]
                if transcript_metadata.get("model_route") else []
            ),
            "tags:",
            '  - "AI静静/原始资料"',
            f'  - "来源/{extracted.media_type}"',
            "---",
            "",
        ]
        source_value = extracted.original_uri or (str(source.local_path) if source.local_path else "未记录")
        body = [
            f"# {extracted.title}",
            "",
            "## 资料档案",
            "",
            f"- 类型：`{extracted.media_type}`",
            f"- 原始来源：{source_value}",
            f"- 归档包：{package or '未归档'}",
            f"- 解析段落：{len(extracted.segments)}",
            f"- 可检索字符：{extracted.extracted_characters}",
            "",
        ]
        if transcript_metadata:
            body.extend(["## 转写事实与模型路线", ""])
            if transcript_metadata.get("v2_path"):
                transcript_path = transcript_metadata["v2_path"].replace(">", "%3E")
                body.append(
                    f"- Transcript V2（原始识别与人工校订分层保存）："
                    f"[打开事实文件](<{transcript_path}>)"
                )
            if transcript_metadata.get("run_id"):
                body.append(f"- Run ID：`{transcript_metadata['run_id']}`")
            if transcript_metadata.get("quality"):
                body.append(
                    f"- 质量状态：`{transcript_metadata['quality']}`"
                    "（以 Transcript V2 `quality` 字段为准）"
                )
            if transcript_metadata.get("model_route"):
                body.append(
                    f"- 模型路线：`{transcript_metadata['model_route']}`"
                    "（以 Transcript V2 `run` 字段为准）"
                )
            body.append("")
        if synthesis:
            body.extend(["## AI 知识提炼（派生层，不覆盖原始事实）", "", synthesis, ""])
        body.extend(["## 原始内容导航", ""])
        for segment in extracted.segments[:80]:
            excerpt = segment.retrieval_text.replace("\n", " ").strip()[:240]
            if excerpt:
                locator = self._located_segment(segment).splitlines()[0]
                if not locator.startswith("["):
                    locator = f"片段 {segment.id}"
                body.append(f"- **{locator}** {excerpt}")
        if len(extracted.segments) > 80:
            body.append(f"- 其余 {len(extracted.segments) - 80} 个片段请在应用中检索。")
        if extracted.warnings:
            body.extend(["", "## 解析提醒", ""])
            body.extend(f"- {warning}" for warning in extracted.warnings)
        body.extend(["", "---", f"*由 {PRODUCT_NAME} 内置摄取服务生成。*", ""])
        _atomic_text(note, "\n".join([*frontmatter, *body]))
        return note

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "created": "完成入库",
            "updated": "更新入库",
            "unchanged": "确认无变化",
            "duplicate": "识别为重复资料",
        }.get(status, status)

    @staticmethod
    def _emit(
        callback: ProgressCallback | None,
        item: str,
        stage: str,
        percent: int,
        message: str,
    ) -> None:
        if callback:
            callback(ProgressEvent(item, stage, max(0, min(100, percent)), message))
