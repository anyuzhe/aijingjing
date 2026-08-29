from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.parse
from dataclasses import asdict, dataclass, field
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


def _source_identity(item: str) -> str:
    if item.casefold().startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(item)
        return urllib.parse.urlunsplit(
            (parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path or "/", parsed.query, "")
        )
    return str(Path(item).expanduser().resolve())


def _source_id(item: str) -> str:
    return "desktop-" + sha256_text(_source_identity(item))[:20]


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
                    result = self._ingest_one(
                        item, indexing, vision, token, progress,
                        package_id=package_id,
                        package_members=members,
                    )
                except CancelledError as exc:
                    result = IngestionResult(item=item, status="cancelled", error=str(exc))
                except QualityGateError as exc:
                    result = IngestionResult(
                        item=item,
                        status="failed",
                        error=str(exc),
                        quality_report=exc.report.to_dict(),
                    )
                    self._emit(progress, item, "failed", 100, f"质检未通过：{result.error}")
                except (OSError, UnicodeError, ValueError, RuntimeError, MissingExtractorDependency) as exc:
                    result = IngestionResult(
                        item=item,
                        status="failed",
                        error=str(exc) or type(exc).__name__,
                    )
                    self._emit(progress, item, "failed", 100, f"导入失败：{result.error}")
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
    ) -> IngestionResult:
        cancellation.check()
        self._emit(progress, item, "preparing", 3, "正在识别资料类型")

        def extraction_progress(message: str) -> None:
            self._emit(progress, item, "extracting", 24, message)

        context = ExtractionContext(
            paths=self.paths,
            settings=self.settings,
            cancellation=cancellation,
            vision=vision,
            progress=extraction_progress,
        )
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
        cancellation.check()
        if not extracted.segments or extracted.extracted_characters == 0:
            raise RuntimeError("资料中没有提取到可检索内容")

        self._emit(progress, item, "validating", 44, "正在执行入库完整性与真实性检查")
        quality = evaluate_extraction(extracted)
        if not quality.accepted:
            failures = "；".join(check.detail for check in quality.checks if check.status == "fail")
            raise QualityGateError(failures or "资料质量未达到入库标准", quality)
        extracted.metadata["quality_report"] = quality.to_dict()

        self._emit(progress, item, "archiving", 52, "正在归档原始资料与解析结果")
        source_id = _source_id(item)
        package = self._archive(extracted, source_id) if self.settings.archive_originals else None
        owned_source = self._owned_source_path(extracted, package)
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

        cancellation.check()
        self._emit(progress, item, "indexing", 68, "正在分块、建立全文与向量索引")
        indexed = indexing.index_document(document)
        note_path: Path | None = None
        warnings = list(extracted.warnings)
        if self.settings.create_source_notes and indexed.status != "duplicate":
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

    def _archive(self, extracted: ExtractionResult, source_id: str) -> Path:
        day = datetime.now().astimezone().strftime("%Y%m%d")
        package = (
            self.paths.archive
            / day[:4]
            / day[4:6]
            / f"{day}_{safe_stem(extracted.title)}_{source_id[-8:]}"
        )
        source_dir = package / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        if extracted.source_path:
            destination = source_dir / extracted.source_path.name
            if not destination.exists() or extracted.checksum != self._checksum_if_file(destination):
                shutil.copy2(extracted.source_path, destination)
        snapshot = extracted.metadata.get("snapshot_html")
        if isinstance(snapshot, str):
            _atomic_text(source_dir / "page.html", snapshot)
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
            "original_uri": extracted.original_uri,
            "metadata": {key: value for key, value in extracted.metadata.items() if key != "snapshot_html"},
            "warnings": extracted.warnings,
            "assets": [str(path) for path in extracted.retained_assets],
            "transcript": str(extracted.transcript_path) if extracted.transcript_path else None,
            "segments": segments,
        }
        _atomic_json(package / "bundle.json", bundle)
        return package

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
                return package / "source" / extracted.source_path.name
            if extracted.original_uri:
                return package / "source" / "page.html"
        return extracted.source_path

    def _synthesize(self, extracted: ExtractionResult) -> str:
        provider = build_answer_provider(self.config, model_id=self._deepseek_synthesis_model_id())
        content = "\n\n".join(
            self._located_segment(segment) for segment in extracted.segments if segment.retrieval_text
        )[:80_000]
        system = (
            "你是 AI知识库-AI静静 的知识整理引擎。只根据用户给定的原始资料，"
            "用简体中文生成结构化 Markdown。保留关键数字、条件、公式、流程和不确定性；"
            "不得臆测。不要输出一级标题或 YAML。"
        )
        user = (
            f"资料标题：{extracted.title}\n类型：{extracted.media_type}\n\n"
            "请依次输出：## 核心摘要、## 关键知识、## 逻辑与流程、"
            "## 重要证据与定位、## 局限与待验证、## 可拆分的知识卡片。\n\n"
            "原始解析内容：\n" + content
        )
        return provider.generate(AnswerRequest("整理这份资料", system, user, [])).markdown.strip()

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
        if "page" in segment.location:
            location.append(f"P{segment.location['page']}")
        if "slide" in segment.location:
            location.append(f"S{segment.location['slide']}")
        if "timestamp_start" in segment.location:
            location.append(f"{float(segment.location['timestamp_start']):.1f}s")
        prefix = f"[{' / '.join(location)}]\n" if location else ""
        return prefix + segment.retrieval_text

    def _write_source_note(
        self,
        extracted: ExtractionResult,
        source: SourceReference,
        document_id: str,
        package: Path | None,
        synthesis: str,
    ) -> Path:
        note = self.paths.notes / "Sources" / f"{safe_stem(extracted.title)}--{source.source_id[-8:]}.md"
        frontmatter = [
            "---",
            f"title: {json.dumps(extracted.title, ensure_ascii=False)}",
            'type: "source-note"',
            f"document_id: {json.dumps(document_id)}",
            f"source_id: {json.dumps(source.source_id)}",
            f"media_type: {json.dumps(extracted.media_type)}",
            f"checksum: {json.dumps(extracted.checksum)}",
            f"updated_at: {json.dumps(utcnow_iso())}",
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
        if synthesis:
            body.extend(["## AI 知识提炼", "", synthesis, ""])
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
