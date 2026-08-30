from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..config import AppConfig
from ..ingestion.retranscription import LocalASRReRecognizer
from ..ingestion.types import CancelledError, CancellationToken
from ..models import utcnow_iso
from ..product import DesktopSettings, ProductPaths
from ..providers.web import DuckDuckGoWebSearchProvider, WebSearchProvider
from ..runtime import build_answer_provider
from ..storage import KnowledgeDatabase
from .deep_correction import (
    CorrectionAuditItem,
    CorrectionLLM,
    DeepCorrectionCancelled,
    DeepCorrectionConfig,
    DeepCorrectionEngine,
    DeepCorrectionResult,
    detect_correction_issues,
)
from .deep_repository import DeepCorrectionRepository
from .evidence import collect_external_evidence
from .exporter import DeepCorrectionMarkdownExporter
from .repository import TranscriptRepository
from .runtime import AnswerProviderCorrectionLLM, FileCorrectionCheckpointStore
from .schema import TranscriptSegment, TranscriptV2


WorkflowProgress = Callable[[str, int, int, str], None]
RunCreatedCallback = Callable[[str], None]
LLMFactory = Callable[[AppConfig, str], CorrectionLLM]
ReRecognizerFactory = Callable[[Path, DesktopSettings, str, Callable[[], None]], object]


_SAFE_STEM_RE = re.compile(r"[^\w\-.\u3400-\u9fff]+", re.UNICODE)


def _safe_stem(value: str, fallback: str) -> str:
    result = _SAFE_STEM_RE.sub("-", Path(str(value or "")).stem).strip("-._")[:80]
    return result or fallback


def _timestamp(milliseconds: int) -> str:
    seconds = max(0, int(milliseconds)) // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _provider_identity(model_id: str) -> tuple[str, str]:
    parts = str(model_id or "").split("::", 2)
    if len(parts) != 3 or parts[0] != "compatible" or not parts[1] or not parts[2]:
        raise ValueError("深度精校只允许使用已配置的 DeepSeek/Kimi 兼容 API，不调用 Codex CLI")
    return parts[1], parts[2]


def _default_llm_factory(config: AppConfig, model_id: str) -> CorrectionLLM:
    return AnswerProviderCorrectionLLM(build_answer_provider(config, model_id=model_id))


def _default_rerecognizer_factory(
    media_path: Path,
    settings: DesktopSettings,
    provider: str,
    check_cancelled: Callable[[], None],
) -> LocalASRReRecognizer:
    return LocalASRReRecognizer(
        media_path,
        settings,
        original_provider=provider,
        check_cancelled=check_cancelled,
    )


class DeepCorrectionWorkflow:
    """Orchestrate correction, evidence, audit persistence and Markdown export."""

    def __init__(
        self,
        paths: ProductPaths,
        config: AppConfig,
        settings: DesktopSettings,
        *,
        llm_factory: LLMFactory | None = None,
        web_provider: WebSearchProvider | None = None,
        rerecognizer_factory: ReRecognizerFactory | None = None,
    ) -> None:
        self.paths = paths.ensure()
        self.config = config
        self.settings = settings
        self.llm_factory = llm_factory or _default_llm_factory
        self.web_provider = web_provider or DuckDuckGoWebSearchProvider()
        self.rerecognizer_factory = rerecognizer_factory or _default_rerecognizer_factory

    def run(
        self,
        transcript_run_id: str,
        *,
        progress: WorkflowProgress | None = None,
        cancellation: CancellationToken | None = None,
        correction_run_id: str | None = None,
        run_created: RunCreatedCallback | None = None,
    ) -> dict[str, object]:
        token = cancellation or CancellationToken()
        provider_id, model = _provider_identity(self.settings.deep_correction_model)

        def emit(stage: str, completed: int, total: int, message: str) -> None:
            if progress:
                progress(stage, completed, total, message)

        def check_cancelled() -> None:
            try:
                token.check()
            except CancelledError as exc:
                raise DeepCorrectionCancelled("深度精校已由用户取消") from exc

        with KnowledgeDatabase(self.paths.database) as database:
            transcripts = TranscriptRepository(database)
            repository = DeepCorrectionRepository(database)
            transcript = transcripts.get_transcript(transcript_run_id)
            source_run = transcripts.get_run(transcript_run_id)
            if transcript is None or source_run is None:
                raise ValueError("没有找到可精校的 Transcript V2 事实层")
            if correction_run_id:
                correction_run = repository.get_run(correction_run_id)
                if correction_run is None or correction_run.transcript_run_id != transcript_run_id:
                    raise ValueError("指定的深度精校任务与当前转写不匹配")
                if correction_run.status == "completed":
                    return self._snapshot(database, correction_run.id)
                if correction_run.status == "failed":
                    correction_run = repository.retry_run(correction_run.id)
                if correction_run.status != "queued":
                    raise ValueError("当前深度精校任务不能开始或重试")
            else:
                correction_run = repository.create_run(
                    transcript_run_id,
                    provider=provider_id,
                    model=model,
                    config=self._run_config(),
                )
            correction_run = repository.start_run(correction_run.id)
            if run_created:
                run_created(correction_run.id)
            try:
                check_cancelled()
                emit("validation", 0, 11, "正在校验原始转写、时间轴与不可变来源")
                issues = detect_correction_issues(transcript, config=self._engine_config())
                emit("audio_quality", 1, 11, f"已识别 {len(issues)} 个需核对区段")
                emit("chunking", 2, 11, "已规划按时间与说话人合并可读段落")
                speaker_count = len({item.speaker_id for item in transcript.segments if item.speaker_id})
                emit(
                    "speakers",
                    3,
                    11,
                    f"已载入 {speaker_count} 个匿名说话人轨道"
                    if speaker_count else "当前没有可靠说话人轨道，将保留未确认标记",
                )
                known_terms = self._context_term_map(database, source_run.document_id)
                emit("terminology", 4, 11, f"已载入 {len(known_terms)} 个标准术语与变体")

                external = ()
                collection_warnings: list[str] = []
                if self.settings.deep_correction_web_verification:
                    collection = collect_external_evidence(
                        transcript,
                        self.web_provider,
                        known_terms=known_terms,
                        max_queries=self.settings.deep_correction_max_external_queries,
                    )
                    external = collection.evidence
                    collection_warnings.extend(collection.warnings)

                media_path = self._media_path(database, source_run.document_id, transcript)
                rerecognizer = None
                if self.settings.deep_correction_retranscribe_anomalies and media_path is not None:
                    rerecognizer = self.rerecognizer_factory(
                        media_path, self.settings, source_run.provider, check_cancelled
                    )
                elif self.settings.deep_correction_retranscribe_anomalies:
                    collection_warnings.append("找不到可播放的本地原始音视频，异常区间未做局部重识别")

                emit("semantic_correction", 5, 11, "正在进行跨片段语义精校与实体一致性检查")
                engine = DeepCorrectionEngine(
                    self.llm_factory(self.config, self.settings.deep_correction_model),
                    rerecognizer=rerecognizer,  # type: ignore[arg-type]
                    checkpoint_store=FileCorrectionCheckpointStore(
                        self.paths.cache / "deep-correction" / transcript_run_id
                    ),
                    config=self._engine_config(),
                    external_evidence=external,
                )

                def engine_progress(
                    stage: str, completed: int, total: int, message: str
                ) -> None:
                    suffix = f"（{completed}/{total}）" if total else ""
                    emit("semantic_correction", 5, 11, message + suffix)

                result = engine.run(
                    transcript,
                    known_terms=known_terms,
                    progress=engine_progress,
                    check_cancelled=check_cancelled,
                )
                result.warnings.extend(collection_warnings)
                if not self.settings.deep_correction_generate_knowledge_cards:
                    result.knowledge_cards.clear()
                if not self.settings.deep_correction_generate_mermaid:
                    result.mermaid = ""
                emit("evidence", 6, 11, f"已校验 {len(external)} 条外部检索候选证据")
                emit("consistency", 7, 11, f"已统一 {len(result.entities)} 组术语与实体")
                emit(
                    "uncertainty",
                    8,
                    11,
                    f"已保守标注 {sum(item.uncertain for item in result.audit)} 条不确定修改",
                )
                emit(
                    "knowledge_cards",
                    9,
                    11,
                    f"已生成 {len(result.knowledge_cards)} 张知识卡和可追溯关系图",
                )
                paragraphs, paragraph_by_segment = self._paragraphs(correction_run.id, result)
                changes, evidence = self._bundle_records(
                    correction_run.id,
                    result,
                    paragraph_by_segment,
                    transcript,
                    media_path,
                    external,
                )
                payload = self._result_payload(
                    result,
                    include_mermaid=self.settings.deep_correction_generate_mermaid,
                )
                quality = {
                    "issue_count": len(result.issues),
                    "change_count": len(result.audit),
                    "uncertain_count": sum(item.uncertain for item in result.audit),
                    "external_evidence_count": len(external),
                    "raw_facts_preserved": True,
                    "warnings": list(dict.fromkeys(result.warnings)),
                }
                check_cancelled()
                if hasattr(repository, "persist_result_bundle"):
                    repository.persist_result_bundle(
                        correction_run.id,
                        paragraphs=paragraphs,
                        changes=changes,
                        evidence=evidence,
                        result=payload,
                        quality_summary=quality,
                    )
                else:  # pragma: no cover - compatibility with pre-v13 development DB code
                    repository.save_paragraphs(correction_run.id, paragraphs)
                    for change in changes:
                        repository.propose_change(correction_run.id, **change)
                    for item in evidence:
                        repository.add_evidence(correction_run.id, **item)
                    repository.complete_run(
                        correction_run.id, result=payload, quality_summary=quality
                    )
                auto_accepted = self._auto_accept_verified_changes(
                    database,
                    repository,
                    correction_run.id,
                )
                emit("quality_gate", 10, 11, "完整性、引用和原稿差异审计已通过")
                export = self._export(database, correction_run.id, transcript.source.name)
                snapshot = self._snapshot(database, correction_run.id)
                snapshot["output_path"] = str(export.path)
                snapshot["output_checksum"] = export.sha256
                snapshot["auto_accepted_change_ids"] = auto_accepted
                return snapshot
            except DeepCorrectionCancelled as exc:
                current = repository.get_run(correction_run.id)
                if current is not None and current.status in {"queued", "running", "failed"}:
                    repository.mark_cancelled(correction_run.id, reason=str(exc))
                raise CancelledError(str(exc)) from exc
            except Exception as exc:
                current = repository.get_run(correction_run.id)
                if current is not None and current.status == "running":
                    repository.fail_run(correction_run.id, str(exc))
                try:
                    setattr(exc, "correction_run_id", correction_run.id)
                except Exception:
                    pass
                raise

    def snapshot(self, correction_run_id: str) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            return self._snapshot(database, correction_run_id)

    def review_change(
        self,
        change_id: str,
        *,
        decision: str,
        actor: str = "user",
        reason: str = "人工复核深度精校建议",
    ) -> dict[str, object]:
        clean = str(decision or "").strip().casefold()
        with KnowledgeDatabase(self.paths.database) as database:
            repository = DeepCorrectionRepository(database)
            change = repository.get_change(change_id)
            if change is None:
                raise ValueError("精校修改不存在")
            if clean == "accepted":
                if not hasattr(repository, "accept_change_and_apply"):
                    raise RuntimeError("数据库尚不支持原子接受精校修改")
                reviewed = repository.accept_change_and_apply(
                    change_id, actor=actor, reason=reason
                )
                run = repository.get_run(reviewed.correction_run_id)
                assert run is not None
                self._write_latest_v2(database, run.transcript_run_id)
            elif clean == "rejected":
                reviewed = repository.review_change(
                    change_id, decision="rejected", actor=actor, reason=reason
                )
            else:
                raise ValueError("decision 只能是 accepted 或 rejected")
            return {
                "change": reviewed.to_dict(),
                "snapshot": self._snapshot(database, reviewed.correction_run_id),
            }

    def export(
        self,
        correction_run_id: str,
        target: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> dict[str, object]:
        with KnowledgeDatabase(self.paths.database) as database:
            run = DeepCorrectionRepository(database).get_run(correction_run_id)
            if run is None:
                raise ValueError("深度精校任务不存在")
            transcript = TranscriptRepository(database).get_transcript(run.transcript_run_id)
            if transcript is None:
                raise ValueError("原始转写不存在")
            if target is None and run.output_path:
                target = Path(run.output_path)
                overwrite = True
            elif target is None:
                target = self._default_output_path(run.id, transcript.source.name)
            exported = DeepCorrectionMarkdownExporter(
                DeepCorrectionRepository(database)
            ).export(
                run.id,
                target,
                overwrite=overwrite,
                allowed_root=self.paths.transcripts,
            )
            return {
                "path": str(exported.path),
                "sha256": exported.sha256,
                "bytes_written": exported.bytes_written,
                "existing": False,
            }

    def _run_config(self) -> dict[str, object]:
        return {
            "pipeline": "deep-correction-v1",
            "model_id": self.settings.deep_correction_model,
            "retranscribe_anomalies": self.settings.deep_correction_retranscribe_anomalies,
            "web_verification": self.settings.deep_correction_web_verification,
            "generate_knowledge_cards": self.settings.deep_correction_generate_knowledge_cards,
            "generate_mermaid": self.settings.deep_correction_generate_mermaid,
            "confidence_threshold": self.settings.deep_correction_confidence_threshold,
            "chunk_seconds": self.settings.deep_correction_chunk_seconds,
            "overlap_seconds": self.settings.deep_correction_overlap_seconds,
            "max_external_queries": self.settings.deep_correction_max_external_queries,
        }

    def _engine_config(self) -> DeepCorrectionConfig:
        return DeepCorrectionConfig(
            target_chunk_ms=self.settings.deep_correction_chunk_seconds * 1000,
            overlap_ms=self.settings.deep_correction_overlap_seconds * 1000,
            confident_apply_threshold=self.settings.deep_correction_confidence_threshold,
        )

    def _context_term_map(
        self, database: KnowledgeDatabase, document_id: str | None
    ) -> dict[str, tuple[str, ...]]:
        source_id = ""
        if document_id:
            row = database.get_document(document_id)
            source_id = str(row["source_id"] or "") if row is not None else ""
        rows = database.connection.execute(
            """SELECT t.canonical_term, t.variants_json
               FROM asr_glossary_terms t JOIN asr_glossaries g ON g.id=t.glossary_id
               WHERE g.enabled=1 AND (
                   g.scope='global'
                   OR (g.scope='knowledge_space' AND g.scope_id=?)
                   OR (g.scope='source' AND g.scope_id=?))
               ORDER BY t.normalized_term, t.id""",
            (self.settings.asr_knowledge_space_id, source_id),
        ).fetchall()
        result: dict[str, tuple[str, ...]] = {}
        for row in rows:
            canonical = str(row["canonical_term"] or "").strip()
            try:
                raw = json.loads(row["variants_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                raw = []
            variants = tuple(str(item).strip() for item in raw if str(item).strip()) if isinstance(raw, list) else ()
            if canonical:
                result[canonical] = tuple(dict.fromkeys(variants))
        return result

    @staticmethod
    def _media_path(
        database: KnowledgeDatabase,
        document_id: str | None,
        transcript: TranscriptV2,
    ) -> Path | None:
        candidates: list[str] = []
        if document_id:
            row = database.get_document(document_id)
            if row is not None:
                candidates.extend([str(row["local_path"] or ""), str(row["original_uri"] or "")])
        candidates.append(str(transcript.source.original_uri or ""))
        for value in candidates:
            if not value or value.casefold().startswith(("http://", "https://")):
                continue
            try:
                path = Path(value).expanduser().resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if path.is_file():
                return path
        return None

    @staticmethod
    def _paragraphs(
        correction_run_id: str,
        result: DeepCorrectionResult,
    ) -> tuple[list[dict[str, object]], dict[str, str]]:
        segments = result.transcript.segments
        chapter_starts = {item.start_segment_id for item in result.chapters}
        groups: list[list[TranscriptSegment]] = []
        current: list[TranscriptSegment] = []
        for segment in segments:
            current_chars = sum(len(item.effective_text) for item in current)
            should_break = bool(current) and (
                segment.id in chapter_starts
                or segment.speaker_id != current[-1].speaker_id
                or segment.start_ms - current[-1].end_ms > 2_500
                or segment.end_ms - current[0].start_ms > 90_000
                or current_chars + len(segment.effective_text) > 700
            )
            if should_break:
                groups.append(current)
                current = []
            current.append(segment)
        if current:
            groups.append(current)
        uncertain_segments = {item.segment_id for item in result.audit if item.uncertain}
        paragraphs: list[dict[str, object]] = []
        lookup: dict[str, str] = {}
        for ordinal, group in enumerate(groups):
            paragraph_id = f"{correction_run_id}-p{ordinal:05d}"
            segment_ids = [item.id for item in group]
            for segment_id in segment_ids:
                lookup[segment_id] = paragraph_id
            paragraphs.append({
                "id": paragraph_id,
                "ordinal": ordinal,
                "start_ms": group[0].start_ms,
                "end_ms": group[-1].end_ms,
                "speaker_id": group[0].speaker_id,
                "source_segment_ids": segment_ids,
                "original_text": " ".join(item.raw_text.strip() for item in group if item.raw_text.strip()),
                "corrected_text": " ".join(item.effective_text.strip() for item in group if item.effective_text.strip()),
                "quality_status": "review" if uncertain_segments.intersection(segment_ids) else "pass",
                "metadata": {"raw_facts_preserved": True},
            })
        return paragraphs, lookup

    def _bundle_records(
        self,
        correction_run_id: str,
        result: DeepCorrectionResult,
        paragraph_by_segment: Mapping[str, str],
        source_transcript: TranscriptV2,
        media_path: Path | None,
        external: Sequence[object],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        source_lookup = {item.id: item for item in source_transcript.segments}
        changes: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []
        cited_web_ids: set[str] = set()
        for ordinal, audit in enumerate(result.audit):
            digest = hashlib.sha256(
                f"{correction_run_id}\0{audit.segment_id}\0{audit.after}".encode("utf-8")
            ).hexdigest()[:18]
            change_id = f"{correction_run_id}-c{digest}"
            change_type = self._change_type(audit)
            changes.append({
                "id": change_id,
                "change_type": change_type,
                "before_text": audit.before,
                "after_text": audit.after,
                "reason": audit.reason,
                "paragraph_id": paragraph_by_segment.get(audit.segment_id),
                "confidence": audit.confidence,
                "source_segment_ids": [audit.segment_id],
                "metadata": {
                    "uncertain": audit.uncertain,
                    "uncertainty_reason": "证据不足或置信度未达到自动确认阈值" if audit.uncertain else "",
                    "issue_codes": list(audit.issue_codes),
                    "speaker_id": audit.speaker_id,
                    "ordinal": ordinal,
                },
                "change_id": change_id,
                "actor": "deep-correction-model",
            })
            segment = source_lookup[audit.segment_id]
            for evidence_index, item in enumerate(audit.evidence):
                evidence_id = f"{change_id}-e{evidence_index:03d}"
                cited_segment = source_lookup.get(item.segment_id or audit.segment_id, segment)
                if item.kind == "web":
                    cited_web_ids.add(str(item.evidence_id or ""))
                    evidence.append({
                        "evidence_type": "external",
                        "title": item.title or "外部核验证据",
                        "url": item.url,
                        "summary": item.quote,
                        "change_id": change_id,
                        "metadata": {"kind": "web", "evidence_id": item.evidence_id, "verified_quote": True},
                        "evidence_id": evidence_id,
                    })
                else:
                    evidence.append({
                        "evidence_type": "source" if item.kind in {"source", "context"} else "model",
                        "title": f"原始音频 {_timestamp(cited_segment.start_ms)} · {item.kind}",
                        "url": media_path.as_uri() if media_path is not None else None,
                        "summary": item.quote,
                        "change_id": change_id,
                        "source_reference": {
                            "segment_id": item.segment_id,
                            "start_ms": cited_segment.start_ms,
                            "end_ms": cited_segment.end_ms,
                            "kind": item.kind,
                        },
                        "metadata": {"verified_quote": True},
                        "evidence_id": evidence_id,
                    })
        for item in external:
            item_id = str(getattr(item, "id", ""))
            if not item_id or item_id in cited_web_ids:
                continue
            evidence.append({
                "evidence_type": "external",
                "title": str(getattr(item, "title", "外部检索候选")),
                "url": str(getattr(item, "url", "")) or None,
                "summary": "检索候选，尚未被模型逐字引用：" + str(getattr(item, "snippet", "")),
                "metadata": {
                    "kind": "web_candidate",
                    "evidence_id": item_id,
                    "query": str(getattr(item, "query", "")),
                    "verified_quote": False,
                },
                "evidence_id": f"{correction_run_id}-{item_id}",
            })
        return changes, evidence

    @staticmethod
    def _change_type(audit: CorrectionAuditItem) -> str:
        codes = set(audit.issue_codes)
        if "professional_term" in codes:
            return "terminology"
        if "number_or_unit" in codes:
            return "fact_or_number"
        if "repetition_loop" in codes or "truncated" in codes:
            return "asr_recovery"
        return "semantic"

    @staticmethod
    def _result_payload(
        result: DeepCorrectionResult,
        *,
        include_mermaid: bool,
    ) -> dict[str, object]:
        payload = result.to_dict()
        payload.update({
            "include_mermaid": bool(include_mermaid),
            "processing_boundaries": [
                "原始 ASR、时间轴和说话人 ID 保持不可变；精校作为独立版本保存。",
                "只有逐条接受的修改才进入 Transcript V2 corrected_text 与问答索引。",
                "外部网页摘要是不可信候选证据；只有 ID、URL、逐字摘录均匹配才可引用。",
                "证据不足、模型冲突或局部重识别失败时保留原文并明确标注待核实。",
            ],
            "uncertain_items": [
                f"{_timestamp(item.start_ms)} {item.before} → {item.after}"
                for item in result.audit if item.uncertain
            ],
            "relations": [
                {"source": chapter.title, "target": card.title, "label": "提炼"}
                for chapter in result.chapters
                for card in result.knowledge_cards
                if set(chapter.evidence_segment_ids).intersection(card.evidence_segment_ids)
            ],
        })
        return payload

    def _auto_accept_verified_changes(
        self,
        database: KnowledgeDatabase,
        repository: DeepCorrectionRepository,
        correction_run_id: str,
    ) -> list[str]:
        """Apply only certain, high-confidence changes with verified web evidence.

        A search result by itself is never enough.  The evidence row must belong
        to the change, contain a navigable URL, and have ``verified_quote=True``;
        that flag is only emitted after the model cites an injected evidence ID,
        exact URL and a verbatim snippet substring.
        """

        if not self.settings.deep_correction_auto_apply_high_confidence:
            return []
        verified_external = {
            item.change_id
            for item in repository.list_evidence(correction_run_id)
            if item.change_id
            and item.evidence_type == "external"
            and bool(item.url)
            and item.metadata.get("verified_quote") is True
        }
        accepted: list[str] = []
        for change in repository.list_changes(
            correction_run_id, statuses=("proposed",)
        ):
            if (
                change.id not in verified_external
                or change.metadata.get("uncertain") is True
                or change.confidence is None
                or change.confidence < self.settings.deep_correction_confidence_threshold
            ):
                continue
            repository.accept_change_and_apply(
                change.id,
                actor="deep-correction-auto",
                reason="达到用户阈值且具有经引用契约校验的外部证据",
                metadata={
                    "automatic": True,
                    "confidence_threshold": (
                        self.settings.deep_correction_confidence_threshold
                    ),
                },
            )
            accepted.append(change.id)
        if accepted:
            run = repository.get_run(correction_run_id)
            assert run is not None
            self._write_latest_v2(database, run.transcript_run_id)
        return accepted

    def _default_output_path(self, correction_run_id: str, source_name: str) -> Path:
        stem = _safe_stem(source_name, "transcript")
        return self.paths.transcripts / f"{stem}-完整精校转写-说话人版-{correction_run_id[-8:]}.md"

    def _export(self, database: KnowledgeDatabase, correction_run_id: str, source_name: str):
        return DeepCorrectionMarkdownExporter(DeepCorrectionRepository(database)).export(
            correction_run_id,
            self._default_output_path(correction_run_id, source_name),
            allowed_root=self.paths.transcripts,
        )

    def _write_latest_v2(self, database: KnowledgeDatabase, transcript_run_id: str) -> Path:
        repository = TranscriptRepository(database)
        target = self.paths.transcripts / f"{_safe_stem(transcript_run_id, 'transcript')}.latest.v2.json"
        expected = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
        return repository.write_latest_v2(
            transcript_run_id,
            target,
            overwrite=target.exists(),
            expected_existing_checksum=expected,
            allowed_root=self.paths.transcripts,
        )

    def _snapshot(self, database: KnowledgeDatabase, correction_run_id: str) -> dict[str, object]:
        repository = DeepCorrectionRepository(database)
        snapshot = repository.snapshot(correction_run_id)
        run = repository.get_run(correction_run_id)
        if run is None:
            raise ValueError("深度精校任务不存在")
        transcript = TranscriptRepository(database).get_transcript(run.transcript_run_id)
        if transcript is None:
            raise ValueError("深度精校关联的转写不存在")
        paragraphs = repository.list_paragraphs(run.id)
        evidence = repository.list_evidence(run.id)
        by_change: dict[str, list[dict[str, str]]] = {}
        for item in evidence:
            if not item.change_id:
                continue
            by_change.setdefault(item.change_id, []).append({
                "label": item.title or item.summary or "证据",
                "url": item.url or "",
            })
        segment_lookup = {item.id: item for item in transcript.segments}
        speaker_names = {
            item.id: item.display_name or item.id for item in transcript.speakers
        }
        changes: list[dict[str, object]] = []
        for item in repository.list_changes(run.id):
            segment = segment_lookup.get(item.source_segment_ids[0]) if item.source_segment_ids else None
            metadata = item.metadata
            changes.append({
                "id": item.id,
                "start_ms": segment.start_ms if segment else 0,
                "end_ms": segment.end_ms if segment else 0,
                "speaker": speaker_names.get(segment.speaker_id or "", segment.speaker_id or "未确认") if segment else "未确认",
                "raw_text": item.before_text,
                "corrected_text": item.after_text,
                "confidence": item.confidence or 0.0,
                "uncertain": bool(metadata.get("uncertain")),
                "uncertainty_reason": str(metadata.get("uncertainty_reason") or ""),
                "evidence": by_change.get(item.id, []),
                "rationale": item.reason,
                "status": "pending" if item.status == "proposed" else item.status,
            })
        raw_text = "\n".join(
            f"[{_timestamp(item.start_ms)}] {speaker_names.get(item.speaker_id or '', item.speaker_id or '未确认')}：{item.raw_text}"
            for item in transcript.segments
        )
        corrected_text = "\n\n".join(
            f"[{_timestamp(item.start_ms)}–{_timestamp(item.end_ms)}] "
            f"{speaker_names.get(item.speaker_id or '', item.speaker_id or '未确认')}\n{item.corrected_text}"
            for item in paragraphs
        )
        return {
            "correction_run_id": run.id,
            "transcript_run_id": run.transcript_run_id,
            "status": run.status,
            "source_name": transcript.source.name,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "changes": changes,
            "warnings": list(run.quality_summary.get("warnings") or []),
            "output_path": run.output_path,
            "output_checksum": run.output_checksum,
            "last_error": run.last_error,
            "attempt_count": run.attempt_count,
            "max_attempts": run.max_attempts,
        }


__all__ = ["DeepCorrectionWorkflow", "WorkflowProgress"]
