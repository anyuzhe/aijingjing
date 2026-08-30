from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from ..models import utcnow_iso
from ..storage.database import KnowledgeDatabase
from .repository import TranscriptRepository


CORRECTION_RUN_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})
CORRECTION_CHANGE_STATUSES = frozenset({"proposed", "accepted", "rejected"})
CORRECTION_EVIDENCE_TYPES = frozenset({"external", "source", "knowledge", "model"})


def _clean(value: object, *, required: bool = False, field_name: str = "文本") -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{field_name}不能为空")
    return result


def _dump(value: object, *, default: object) -> str:
    try:
        return json.dumps(
            default if value is None else value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("记录必须可以序列化为 JSON") from exc


def _mapping(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sequence(value: object) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _canonical_checksum(value: Mapping[str, Any]) -> str:
    payload = _dump(dict(value), default={}).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unique_strings(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = _clean(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DeepCorrectionRunRecord:
    id: str
    transcript_run_id: str
    provider: str
    model: str
    status: str
    attempt_count: int
    max_attempts: int
    cancel_requested: bool
    config: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    quality_summary: dict[str, Any] = field(default_factory=dict)
    result_checksum: str | None = None
    output_path: str | None = None
    output_checksum: str | None = None
    last_error: str | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorrectionParagraphRecord:
    id: str
    correction_run_id: str
    ordinal: int
    start_ms: int
    end_ms: int
    speaker_id: str | None
    source_segment_ids: tuple[str, ...]
    original_text: str
    corrected_text: str
    quality_status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_segment_ids"] = list(self.source_segment_ids)
        return value


@dataclass(frozen=True, slots=True)
class CorrectionChangeRecord:
    id: str
    correction_run_id: str
    paragraph_id: str | None
    change_type: str
    before_text: str
    after_text: str
    reason: str
    confidence: float | None
    status: str
    source_segment_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_segment_ids"] = list(self.source_segment_ids)
        return value


@dataclass(frozen=True, slots=True)
class CorrectionChangeEventRecord:
    id: str
    change_id: str
    from_status: str | None
    to_status: str
    actor: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CorrectionEvidenceRecord:
    id: str
    correction_run_id: str
    paragraph_id: str | None
    change_id: str | None
    evidence_type: str
    title: str
    url: str | None
    summary: str
    source_reference: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepCorrectionRepository:
    """Mapping-oriented persistence independent from any correction model types."""

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    @property
    def connection(self) -> sqlite3.Connection:
        return self.database.connection

    def create_run(
        self,
        transcript_run_id: str,
        *,
        provider: str,
        model: str,
        config: Mapping[str, Any] | None = None,
        max_attempts: int = 3,
        run_id: str | None = None,
    ) -> DeepCorrectionRunRecord:
        transcript_id = _clean(transcript_run_id, required=True, field_name="原转写 run ID")
        exists = self.connection.execute(
            "SELECT 1 FROM transcription_runs WHERE id=?", (transcript_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"原转写任务不存在：{transcript_id}")
        clean_provider = _clean(provider, required=True, field_name="精校 Provider")
        clean_model = _clean(model, required=True, field_name="精校模型")
        attempts = int(max_attempts)
        if attempts < 1 or attempts > 20:
            raise ValueError("max_attempts 必须在 1 到 20 之间")
        clean_id = _clean(run_id or f"correction-{uuid4().hex}")
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO correction_runs(
                           id, transcript_run_id, provider, model, status,
                           attempt_count, max_attempts, cancel_requested, config_json,
                           result_json, quality_summary_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, 'queued', 0, ?, 0, ?, '{}', '{}', ?, ?)""",
                    (clean_id, transcript_id, clean_provider, clean_model, attempts,
                     _dump(dict(config or {}), default={}), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法创建深度精校任务：{exc}") from exc
        return self._require_run(clean_id)

    def get_run(self, run_id: str) -> DeepCorrectionRunRecord | None:
        row = self.connection.execute(
            "SELECT * FROM correction_runs WHERE id=?", (_clean(run_id),)
        ).fetchone()
        return self._run(row) if row is not None else None

    def list_runs(
        self,
        *,
        transcript_run_id: str | None = None,
        statuses: Sequence[str] = (),
        limit: int = 100,
    ) -> list[DeepCorrectionRunRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if transcript_run_id:
            clauses.append("transcript_run_id=?")
            params.append(transcript_run_id)
        clean_statuses = tuple(dict.fromkeys(_clean(item) for item in statuses if _clean(item)))
        if clean_statuses:
            if any(item not in CORRECTION_RUN_STATUSES for item in clean_statuses):
                raise ValueError("包含不支持的精校任务状态")
            clauses.append("status IN (" + ",".join("?" for _ in clean_statuses) + ")")
            params.extend(clean_statuses)
        query = "SELECT * FROM correction_runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, id LIMIT ?"
        params.append(min(1000, max(1, int(limit))))
        return [self._run(row) for row in self.connection.execute(query, params).fetchall()]

    def start_run(self, run_id: str) -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status != "queued":
            raise ValueError("只有排队中的精校任务可以开始")
        if run.cancel_requested:
            raise ValueError("任务已请求取消")
        if run.attempt_count >= run.max_attempts:
            raise ValueError("精校任务已达到最大尝试次数")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_runs SET status='running',
                       attempt_count=attempt_count+1, started_at=?, completed_at=NULL,
                       last_error=NULL, updated_at=? WHERE id=?""",
                (now, now, run.id),
            )
        return self._require_run(run.id)

    def fail_run(self, run_id: str, error: str) -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status != "running":
            raise ValueError("只有运行中的精校任务可以标记失败")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_runs SET status='failed', last_error=?,
                       completed_at=?, updated_at=? WHERE id=?""",
                (_clean(error)[:8000], now, now, run.id),
            )
        return self._require_run(run.id)

    def retry_run(self, run_id: str) -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status != "failed":
            raise ValueError("只有失败的精校任务可以重试")
        if run.attempt_count >= run.max_attempts:
            raise ValueError("精校任务已达到最大尝试次数")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_runs SET status='queued', cancel_requested=0,
                       completed_at=NULL, updated_at=? WHERE id=?""",
                (now, run.id),
            )
        return self._require_run(run.id)

    def request_cancel(self, run_id: str) -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status not in {"queued", "running"}:
            raise ValueError("当前状态不能请求取消")
        now = utcnow_iso()
        if run.status == "queued":
            with self.connection:
                self.connection.execute(
                    """UPDATE correction_runs SET status='cancelled', cancel_requested=1,
                           completed_at=?, updated_at=? WHERE id=?""",
                    (now, now, run.id),
                )
        else:
            with self.connection:
                self.connection.execute(
                    "UPDATE correction_runs SET cancel_requested=1, updated_at=? WHERE id=?",
                    (now, run.id),
                )
        return self._require_run(run.id)

    def mark_cancelled(self, run_id: str, *, reason: str = "") -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status not in {"queued", "running", "failed"}:
            raise ValueError("当前状态不能标记取消")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_runs SET status='cancelled', cancel_requested=1,
                       last_error=?, completed_at=?, updated_at=? WHERE id=?""",
                (_clean(reason) or run.last_error, now, now, run.id),
            )
        return self._require_run(run.id)

    def complete_run(
        self,
        run_id: str,
        *,
        result: Mapping[str, Any],
        quality_summary: Mapping[str, Any] | None = None,
        output_path: str | None = None,
    ) -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status != "running":
            raise ValueError("只有运行中的精校任务可以完成")
        if run.cancel_requested:
            raise ValueError("任务已经请求取消，不能继续完成")
        result_value = dict(result)
        checksum = _canonical_checksum(result_value)
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_runs SET status='completed', result_json=?,
                       quality_summary_json=?, result_checksum=?, output_path=?,
                       last_error=NULL, completed_at=?, updated_at=? WHERE id=?""",
                (_dump(result_value, default={}), _dump(dict(quality_summary or {}), default={}),
                 checksum, _clean(output_path) or None, now, now, run.id),
            )
        return self._require_run(run.id)

    def persist_result_bundle(
        self,
        run_id: str,
        *,
        paragraphs: Iterable[Mapping[str, Any]],
        changes: Iterable[Mapping[str, Any]],
        evidence: Iterable[Mapping[str, Any]],
        result: Mapping[str, Any],
        quality_summary: Mapping[str, Any] | None = None,
        output_path: str | None = None,
    ) -> DeepCorrectionRunRecord:
        """Atomically persist every derived fact and complete a running run.

        Inputs are plain mappings so callers may pass serialized model results
        without importing the correction engine's concrete dataclasses. Public
        CRUD methods are intentionally not called here because their connection
        contexts could commit a partially persisted retry.
        """

        run = self._require_run(run_id)
        if run.status != "running":
            raise ValueError("只有运行中的精校任务可以保存完整结果")
        if run.cancel_requested:
            raise ValueError("任务已经请求取消，不能保存完整结果")
        for table in ("correction_paragraphs", "correction_changes", "correction_evidence"):
            exists = self.connection.execute(
                f"SELECT 1 FROM {table} WHERE correction_run_id=? LIMIT 1", (run.id,)
            ).fetchone()
            if exists is not None:
                raise ValueError("精校任务已经存在派生事实，拒绝覆盖或拼接半成品")

        prepared_paragraphs = [self._prepare_paragraph(run, item) for item in paragraphs]
        if not prepared_paragraphs:
            raise ValueError("完整精校结果必须包含至少一个正文段落")
        paragraph_ids = [str(item[0]) for item in prepared_paragraphs]
        paragraph_ordinals = [int(item[2]) for item in prepared_paragraphs]
        if len(set(paragraph_ids)) != len(paragraph_ids) or len(set(paragraph_ordinals)) != len(paragraph_ordinals):
            raise ValueError("精校段落 ID 和 ordinal 必须唯一")
        paragraph_id_set = set(paragraph_ids)

        prepared_changes: list[tuple[object, ...]] = []
        change_ids: list[str] = []
        for raw in changes:
            change_id = _clean(raw.get("id") or f"change-{uuid4().hex}")
            paragraph_id = _clean(raw.get("paragraph_id")) or None
            if paragraph_id and paragraph_id not in paragraph_id_set:
                raise ValueError("修改建议关联了不属于本次结果的段落")
            source_values = raw.get("source_segment_ids")
            if source_values is None and raw.get("segment_id") is not None:
                source_values = [raw.get("segment_id")]
            source_ids = _unique_strings(source_values or ())
            self._validate_source_segments(run, source_ids)
            confidence_value = raw.get("confidence")
            confidence = None if confidence_value is None else float(confidence_value)
            if confidence is not None and not 0 <= confidence <= 1:
                raise ValueError("修改建议置信度必须在 0 到 1 之间")
            prepared_changes.append((
                change_id,
                run.id,
                paragraph_id,
                _clean(raw.get("change_type") or raw.get("type") or "correction",
                       required=True, field_name="修改类型"),
                str(raw.get("before_text") if raw.get("before_text") is not None else raw.get("before") or ""),
                str(raw.get("after_text") if raw.get("after_text") is not None else raw.get("after") or ""),
                _clean(raw.get("reason")),
                confidence,
                _dump(list(source_ids), default=[]),
                _dump(dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), Mapping) else {}, default={}),
                _clean(raw.get("actor")) or "model",
            ))
            change_ids.append(change_id)
        if len(set(change_ids)) != len(change_ids):
            raise ValueError("修改建议 ID 必须唯一")
        change_id_set = set(change_ids)

        prepared_evidence: list[tuple[object, ...]] = []
        evidence_ids: list[str] = []
        evidence_aliases = {
            "web": "external",
            "context": "source",
            "rerecognition": "source",
            "glossary": "knowledge",
        }
        for raw in evidence:
            evidence_id = _clean(raw.get("id") or raw.get("evidence_id") or f"evidence-{uuid4().hex}")
            paragraph_id = _clean(raw.get("paragraph_id")) or None
            change_id = _clean(raw.get("change_id")) or None
            if paragraph_id and paragraph_id not in paragraph_id_set:
                raise ValueError("证据关联了不属于本次结果的段落")
            if change_id and change_id not in change_id_set:
                raise ValueError("证据关联了不属于本次结果的修改建议")
            evidence_type = _clean(raw.get("evidence_type") or raw.get("kind") or "model").casefold()
            evidence_type = evidence_aliases.get(evidence_type, evidence_type)
            if evidence_type not in CORRECTION_EVIDENCE_TYPES:
                raise ValueError(f"不支持的证据类型：{evidence_type}")
            prepared_evidence.append((
                evidence_id,
                run.id,
                paragraph_id,
                change_id,
                evidence_type,
                _clean(raw.get("title")),
                _clean(raw.get("url")) or None,
                _clean(raw.get("summary") or raw.get("quote") or raw.get("snippet")),
                _dump(dict(raw.get("source_reference") or {})
                      if isinstance(raw.get("source_reference"), Mapping) else {}, default={}),
                _dump(dict(raw.get("metadata") or {})
                      if isinstance(raw.get("metadata"), Mapping) else {}, default={}),
            ))
            evidence_ids.append(evidence_id)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("精校证据 ID 必须唯一")

        result_value = dict(result)
        result_json = _dump(result_value, default={})
        quality_json = _dump(dict(quality_summary or {}), default={})
        result_checksum = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        now = utcnow_iso()
        try:
            with self.connection:
                for paragraph in prepared_paragraphs:
                    self.connection.execute(
                        """INSERT INTO correction_paragraphs(
                               id, correction_run_id, ordinal, start_ms, end_ms, speaker_id,
                               source_segment_ids_json, original_text, corrected_text,
                               quality_status, metadata_json, created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (*paragraph, now, now),
                    )
                for change in prepared_changes:
                    self.connection.execute(
                        """INSERT INTO correction_changes(
                               id, correction_run_id, paragraph_id, change_type, before_text,
                               after_text, reason, confidence, status, source_segment_ids_json,
                               metadata_json, created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?, ?, ?)""",
                        (*change[:10], now, now),
                    )
                    self._insert_change_event(
                        str(change[0]), None, "proposed", actor=str(change[10]),
                        reason="生成修改建议", created_at=now,
                    )
                for item in prepared_evidence:
                    self.connection.execute(
                        """INSERT INTO correction_evidence(
                               id, correction_run_id, paragraph_id, change_id, evidence_type,
                               title, url, summary, source_reference_json, metadata_json, created_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (*item, now),
                    )
                cursor = self.connection.execute(
                    """UPDATE correction_runs SET status='completed', result_json=?,
                           quality_summary_json=?, result_checksum=?, output_path=?,
                           last_error=NULL, completed_at=?, updated_at=?
                       WHERE id=? AND status='running' AND cancel_requested=0""",
                    (result_json, quality_json, result_checksum, _clean(output_path) or None,
                     now, now, run.id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("精校任务状态已变化，完整结果未保存")
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法原子保存完整精校结果：{exc}") from exc
        return self._require_run(run.id)

    def set_export_info(self, run_id: str, path: str, checksum: str) -> DeepCorrectionRunRecord:
        run = self._require_run(run_id)
        if run.status != "completed":
            raise ValueError("只有已完成任务可以登记导出结果")
        clean_path = _clean(path, required=True, field_name="导出路径")
        clean_checksum = _clean(checksum, required=True, field_name="导出校验值")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_runs SET output_path=?, output_checksum=?, updated_at=?
                   WHERE id=?""",
                (clean_path, clean_checksum, now, run.id),
            )
        return self._require_run(run.id)

    def save_paragraphs(
        self,
        run_id: str,
        paragraphs: Iterable[Mapping[str, Any]],
    ) -> list[CorrectionParagraphRecord]:
        run = self._require_run(run_id)
        if run.status != "running":
            raise ValueError("只有运行中的精校任务可以保存段落")
        existing = self.connection.execute(
            "SELECT 1 FROM correction_paragraphs WHERE correction_run_id=? LIMIT 1", (run.id,)
        ).fetchone()
        if existing is not None:
            raise ValueError("精校段落事实已存在，拒绝覆盖")
        prepared = [self._prepare_paragraph(run, value) for value in paragraphs]
        ids = [item[0] for item in prepared]
        ordinals = [item[2] for item in prepared]
        if len(set(ids)) != len(ids) or len(set(ordinals)) != len(ordinals):
            raise ValueError("精校段落 ID 和 ordinal 必须唯一")
        now = utcnow_iso()
        try:
            with self.connection:
                for item in prepared:
                    self.connection.execute(
                        """INSERT INTO correction_paragraphs(
                               id, correction_run_id, ordinal, start_ms, end_ms, speaker_id,
                               source_segment_ids_json, original_text, corrected_text,
                               quality_status, metadata_json, created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (*item, now, now),
                    )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法保存精校段落：{exc}") from exc
        return self.list_paragraphs(run.id)

    def get_paragraph(self, paragraph_id: str) -> CorrectionParagraphRecord | None:
        row = self.connection.execute(
            "SELECT * FROM correction_paragraphs WHERE id=?", (_clean(paragraph_id),)
        ).fetchone()
        return self._paragraph(row) if row is not None else None

    def list_paragraphs(self, run_id: str) -> list[CorrectionParagraphRecord]:
        return [
            self._paragraph(row)
            for row in self.connection.execute(
                """SELECT * FROM correction_paragraphs
                   WHERE correction_run_id=? ORDER BY ordinal, id""",
                (_clean(run_id),),
            ).fetchall()
        ]

    def propose_change(
        self,
        run_id: str,
        *,
        change_type: str,
        before_text: str,
        after_text: str,
        reason: str = "",
        paragraph_id: str | None = None,
        confidence: float | None = None,
        source_segment_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        change_id: str | None = None,
        actor: str = "model",
    ) -> CorrectionChangeRecord:
        run = self._require_run(run_id)
        if run.status not in {"running", "completed"}:
            raise ValueError("当前精校任务状态不能新增修改建议")
        paragraph = None
        if paragraph_id:
            paragraph = self.get_paragraph(paragraph_id)
            if paragraph is None or paragraph.correction_run_id != run.id:
                raise ValueError("精校段落不属于当前任务")
        segment_ids = _unique_strings(source_segment_ids)
        self._validate_source_segments(run, segment_ids)
        number = None if confidence is None else float(confidence)
        if number is not None and not 0 <= number <= 1:
            raise ValueError("修改建议置信度必须在 0 到 1 之间")
        clean_id = _clean(change_id or f"change-{uuid4().hex}")
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO correction_changes(
                           id, correction_run_id, paragraph_id, change_type, before_text,
                           after_text, reason, confidence, status, source_segment_ids_json,
                           metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?, ?, ?)""",
                    (clean_id, run.id, paragraph.id if paragraph else None,
                     _clean(change_type, required=True, field_name="修改类型"),
                     str(before_text), str(after_text), _clean(reason), number,
                     _dump(list(segment_ids), default=[]), _dump(dict(metadata or {}), default={}),
                     now, now),
                )
                self._insert_change_event(
                    clean_id, None, "proposed", actor=actor,
                    reason="生成修改建议", created_at=now,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法保存修改建议：{exc}") from exc
        return self._require_change(clean_id)

    def get_change(self, change_id: str) -> CorrectionChangeRecord | None:
        row = self.connection.execute(
            "SELECT * FROM correction_changes WHERE id=?", (_clean(change_id),)
        ).fetchone()
        return self._change(row) if row is not None else None

    def list_changes(
        self,
        run_id: str,
        *,
        statuses: Sequence[str] = (),
    ) -> list[CorrectionChangeRecord]:
        params: list[object] = [_clean(run_id)]
        query = "SELECT * FROM correction_changes WHERE correction_run_id=?"
        clean_statuses = tuple(dict.fromkeys(_clean(item) for item in statuses if _clean(item)))
        if clean_statuses:
            if any(item not in CORRECTION_CHANGE_STATUSES for item in clean_statuses):
                raise ValueError("包含不支持的修改建议状态")
            query += " AND status IN (" + ",".join("?" for _ in clean_statuses) + ")"
            params.extend(clean_statuses)
        query += " ORDER BY created_at, id"
        return [self._change(row) for row in self.connection.execute(query, params).fetchall()]

    def review_change(
        self,
        change_id: str,
        *,
        decision: str,
        actor: str = "user",
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> CorrectionChangeRecord:
        change = self._require_change(change_id)
        clean_decision = _clean(decision).casefold()
        if clean_decision not in {"accepted", "rejected"}:
            raise ValueError("decision 只能是 accepted 或 rejected")
        if change.status != "proposed":
            raise ValueError("修改建议已经完成审核，拒绝覆盖审核结论")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE correction_changes SET status=?, reviewed_by=?, reviewed_at=?,
                       updated_at=? WHERE id=?""",
                (clean_decision, _clean(actor) or "user", now, now, change.id),
            )
            self._insert_change_event(
                change.id, "proposed", clean_decision, actor=actor,
                reason=reason, metadata=metadata, created_at=now,
            )
        return self._require_change(change.id)

    def accept_change_and_apply(
        self,
        change_id: str,
        *,
        actor: str = "user",
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> CorrectionChangeRecord:
        """Accept one proposal and apply its complete corrected text atomically.

        The proposal must identify exactly one source segment. ``raw_text`` is
        never updated; only ``corrected_text`` is changed. A no-op acceptance
        still records the review decision, but deliberately avoids a duplicate
        transcript edit entry.
        """

        change = self._require_change(change_id)
        if change.status != "proposed":
            raise ValueError("修改建议已经完成审核，拒绝覆盖审核结论")
        if len(change.source_segment_ids) != 1:
            raise ValueError("接受并应用的修改必须恰好一个原转写片段")
        run = self._require_run(change.correction_run_id)
        segment_id = change.source_segment_ids[0]
        segment = self.connection.execute(
            """SELECT * FROM transcript_segments
               WHERE id=? AND run_id=?""",
            (segment_id, run.transcript_run_id),
        ).fetchone()
        if segment is None:
            raise ValueError("修改建议关联的原转写片段不存在")
        clean_actor = _clean(actor) or "user"
        clean_reason = _clean(reason)
        event_metadata = dict(metadata or {})
        audit_metadata = dict(event_metadata)
        # Provenance keys are authoritative and cannot be replaced by callers.
        audit_metadata.update({
            "correction_run_id": run.id,
            "change_id": change.id,
        })
        current_text = (
            str(segment["corrected_text"])
            if segment["corrected_text"] is not None
            else str(segment["raw_text"])
        )
        after_text = change.after_text
        now = utcnow_iso()
        transcript_repository = TranscriptRepository(self.database)
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE correction_changes SET status='accepted', reviewed_by=?,
                       reviewed_at=?, updated_at=? WHERE id=? AND status='proposed'""",
                (clean_actor, now, now, change.id),
            )
            if cursor.rowcount != 1:
                raise ValueError("修改建议状态已变化，拒绝重复应用")
            self._insert_change_event(
                change.id,
                "proposed",
                "accepted",
                actor=clean_actor,
                reason=clean_reason,
                metadata=event_metadata,
                created_at=now,
            )
            if after_text != current_text:
                self.connection.execute(
                    """UPDATE transcript_segments SET corrected_text=?, updated_at=?
                       WHERE id=? AND run_id=?""",
                    (after_text, now, segment_id, run.transcript_run_id),
                )
                transcript_repository._insert_edit(
                    run_id=run.transcript_run_id,
                    segment_id=segment_id,
                    target_type="segment_text",
                    target_id=segment_id,
                    before_text=current_text,
                    after_text=after_text,
                    edit_type="deep_correction_accept",
                    reason=clean_reason,
                    actor=clean_actor,
                    metadata=audit_metadata,
                    created_at=now,
                    confirmed=True,
                )
            transcript_repository._touch_run(run.transcript_run_id, now)
            self.connection.execute(
                "UPDATE correction_runs SET updated_at=? WHERE id=?", (now, run.id)
            )
        return self._require_change(change.id)

    def list_change_events(self, change_id: str) -> list[CorrectionChangeEventRecord]:
        return [
            self._event(row)
            for row in self.connection.execute(
                """SELECT * FROM correction_change_events
                   WHERE change_id=? ORDER BY created_at, rowid""",
                (_clean(change_id),),
            ).fetchall()
        ]

    def add_evidence(
        self,
        run_id: str,
        *,
        evidence_type: str,
        title: str = "",
        url: str | None = None,
        summary: str = "",
        paragraph_id: str | None = None,
        change_id: str | None = None,
        source_reference: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> CorrectionEvidenceRecord:
        run = self._require_run(run_id)
        clean_type = _clean(evidence_type)
        if clean_type == "web":
            clean_type = "external"
        if clean_type not in CORRECTION_EVIDENCE_TYPES:
            raise ValueError(f"不支持的证据类型：{clean_type}")
        if paragraph_id:
            paragraph = self.get_paragraph(paragraph_id)
            if paragraph is None or paragraph.correction_run_id != run.id:
                raise ValueError("证据关联段落不属于当前任务")
        if change_id:
            change = self.get_change(change_id)
            if change is None or change.correction_run_id != run.id:
                raise ValueError("证据关联修改不属于当前任务")
        clean_id = _clean(evidence_id or f"evidence-{uuid4().hex}")
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO correction_evidence(
                           id, correction_run_id, paragraph_id, change_id, evidence_type,
                           title, url, summary, source_reference_json, metadata_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (clean_id, run.id, _clean(paragraph_id) or None, _clean(change_id) or None,
                     clean_type, _clean(title), _clean(url) or None, _clean(summary),
                     _dump(dict(source_reference or {}), default={}),
                     _dump(dict(metadata or {}), default={}), now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法保存精校证据：{exc}") from exc
        result = self.get_evidence(clean_id)
        assert result is not None
        return result

    def get_evidence(self, evidence_id: str) -> CorrectionEvidenceRecord | None:
        row = self.connection.execute(
            "SELECT * FROM correction_evidence WHERE id=?", (_clean(evidence_id),)
        ).fetchone()
        return self._evidence(row) if row is not None else None

    def list_evidence(self, run_id: str) -> list[CorrectionEvidenceRecord]:
        return [
            self._evidence(row)
            for row in self.connection.execute(
                """SELECT * FROM correction_evidence
                   WHERE correction_run_id=? ORDER BY created_at, id""",
                (_clean(run_id),),
            ).fetchall()
        ]

    def snapshot(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        changes = self.list_changes(run.id)
        return {
            "run": run.to_dict(),
            "paragraphs": [item.to_dict() for item in self.list_paragraphs(run.id)],
            "changes": [item.to_dict() for item in changes],
            "change_events": {
                item.id: [event.to_dict() for event in self.list_change_events(item.id)]
                for item in changes
            },
            "evidence": [item.to_dict() for item in self.list_evidence(run.id)],
        }

    def _prepare_paragraph(
        self,
        run: DeepCorrectionRunRecord,
        value: Mapping[str, Any],
    ) -> tuple[object, ...]:
        paragraph_id = _clean(value.get("id") or f"paragraph-{uuid4().hex}")
        try:
            ordinal = int(value.get("ordinal", 0))
            start_ms = int(value.get("start_ms", 0))
            end_ms = int(value.get("end_ms", 0))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("精校段落序号和时间必须是整数") from exc
        if ordinal < 0 or start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"精校段落时间范围无效：{paragraph_id}")
        segment_ids = _unique_strings(value.get("source_segment_ids", ()))
        if not segment_ids:
            raise ValueError(f"精校段落必须关联原转写片段：{paragraph_id}")
        self._validate_source_segments(run, segment_ids)
        original = str(value.get("original_text") or "")
        corrected = str(value.get("corrected_text") if value.get("corrected_text") is not None else original)
        quality = _clean(value.get("quality_status") or "review").casefold()
        if quality not in {"pass", "review", "fail"}:
            raise ValueError(f"精校段落质量状态无效：{quality}")
        return (
            paragraph_id, run.id, ordinal, start_ms, end_ms,
            _clean(value.get("speaker_id")) or None,
            _dump(list(segment_ids), default=[]), original, corrected, quality,
            _dump(dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), Mapping) else {}, default={}),
        )

    def _validate_source_segments(
        self,
        run: DeepCorrectionRunRecord,
        segment_ids: Sequence[str],
    ) -> None:
        if not segment_ids:
            return
        placeholders = ",".join("?" for _ in segment_ids)
        found = {
            str(row["id"])
            for row in self.connection.execute(
                f"""SELECT id FROM transcript_segments
                    WHERE run_id=? AND id IN ({placeholders})""",
                (run.transcript_run_id, *segment_ids),
            ).fetchall()
        }
        missing = [item for item in segment_ids if item not in found]
        if missing:
            raise ValueError("关联的原转写片段不存在：" + ", ".join(missing))

    def _insert_change_event(
        self,
        change_id: str,
        from_status: str | None,
        to_status: str,
        *,
        actor: str,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
        created_at: str,
    ) -> None:
        self.connection.execute(
            """INSERT INTO correction_change_events(
                   id, change_id, from_status, to_status, actor, reason,
                   metadata_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"change-event-{uuid4().hex}", change_id, from_status, to_status,
             _clean(actor) or "system", _clean(reason),
             _dump(dict(metadata or {}), default={}), created_at),
        )

    def _require_run(self, run_id: str) -> DeepCorrectionRunRecord:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"深度精校任务不存在：{run_id}")
        return run

    def _require_change(self, change_id: str) -> CorrectionChangeRecord:
        change = self.get_change(change_id)
        if change is None:
            raise KeyError(f"精校修改不存在：{change_id}")
        return change

    @staticmethod
    def _run(row: sqlite3.Row) -> DeepCorrectionRunRecord:
        return DeepCorrectionRunRecord(
            id=str(row["id"]), transcript_run_id=str(row["transcript_run_id"]),
            provider=str(row["provider"]), model=str(row["model"]), status=str(row["status"]),
            attempt_count=int(row["attempt_count"]), max_attempts=int(row["max_attempts"]),
            cancel_requested=bool(row["cancel_requested"]), config=_mapping(row["config_json"]),
            result=_mapping(row["result_json"]), quality_summary=_mapping(row["quality_summary_json"]),
            result_checksum=row["result_checksum"], output_path=row["output_path"],
            output_checksum=row["output_checksum"], last_error=row["last_error"],
            created_at=str(row["created_at"]), started_at=row["started_at"],
            completed_at=row["completed_at"], updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _paragraph(row: sqlite3.Row) -> CorrectionParagraphRecord:
        return CorrectionParagraphRecord(
            id=str(row["id"]), correction_run_id=str(row["correction_run_id"]),
            ordinal=int(row["ordinal"]), start_ms=int(row["start_ms"]), end_ms=int(row["end_ms"]),
            speaker_id=row["speaker_id"],
            source_segment_ids=tuple(str(item) for item in _sequence(row["source_segment_ids_json"])),
            original_text=str(row["original_text"]), corrected_text=str(row["corrected_text"]),
            quality_status=str(row["quality_status"]), metadata=_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _change(row: sqlite3.Row) -> CorrectionChangeRecord:
        return CorrectionChangeRecord(
            id=str(row["id"]), correction_run_id=str(row["correction_run_id"]),
            paragraph_id=row["paragraph_id"], change_type=str(row["change_type"]),
            before_text=str(row["before_text"]), after_text=str(row["after_text"]),
            reason=str(row["reason"]), confidence=row["confidence"], status=str(row["status"]),
            source_segment_ids=tuple(str(item) for item in _sequence(row["source_segment_ids_json"])),
            metadata=_mapping(row["metadata_json"]), reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"], created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> CorrectionChangeEventRecord:
        return CorrectionChangeEventRecord(
            id=str(row["id"]), change_id=str(row["change_id"]),
            from_status=row["from_status"], to_status=str(row["to_status"]),
            actor=str(row["actor"]), reason=str(row["reason"]),
            metadata=_mapping(row["metadata_json"]), created_at=str(row["created_at"]),
        )

    @staticmethod
    def _evidence(row: sqlite3.Row) -> CorrectionEvidenceRecord:
        return CorrectionEvidenceRecord(
            id=str(row["id"]), correction_run_id=str(row["correction_run_id"]),
            paragraph_id=row["paragraph_id"], change_id=row["change_id"],
            evidence_type=str(row["evidence_type"]), title=str(row["title"]),
            url=row["url"], summary=str(row["summary"]),
            source_reference=_mapping(row["source_reference_json"]),
            metadata=_mapping(row["metadata_json"]), created_at=str(row["created_at"]),
        )


__all__ = [
    "CORRECTION_CHANGE_STATUSES",
    "CORRECTION_EVIDENCE_TYPES",
    "CORRECTION_RUN_STATUSES",
    "CorrectionChangeEventRecord",
    "CorrectionChangeRecord",
    "CorrectionEvidenceRecord",
    "CorrectionParagraphRecord",
    "DeepCorrectionRepository",
    "DeepCorrectionRunRecord",
]
