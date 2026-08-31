from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..models import utcnow_iso
from .database import KnowledgeDatabase
from .governance import KnowledgeGovernanceRepository, KnowledgeItem


PROPOSAL_STATUSES = frozenset({"proposed", "accepted", "rejected", "merged"})
SOURCE_CLASSES = frozenset(
    {"unassessed", "official", "primary", "research", "industry", "media", "community", "personal"}
)
SOURCE_RELIABILITIES = frozenset({"unassessed", "high", "medium", "low"})
CONFLICT_POLICIES = frozenset({"warn", "require-review", "manual"})

DEFAULT_POLICY: dict[str, object] = {
    "auto_propose": True,
    "conflict_policy": "warn",
    "default_source_reliability": "unassessed",
    "external_verification": False,
    "require_review": True,
    "routing": "source-to-proposal-to-knowledge",
}


def _text(value: object, *, required: bool = False, name: str = "文本") -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{name}不能为空")
    return result


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Sequence):
        candidates = list(value)
    else:
        candidates = []
    return tuple(dict.fromkeys(_text(item) for item in candidates if _text(item)))


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized(value: object) -> str:
    return " ".join(_text(value).casefold().split())


@dataclass(frozen=True, slots=True)
class KnowledgeSpacePolicy:
    id: str
    name: str
    policy: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KnowledgeProposal:
    id: str
    fingerprint: str
    proposed_type: str
    title: str
    summary: str
    body: str
    status: str
    source_document_id: str | None = None
    correction_run_id: str | None = None
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    source_segment_ids: tuple[str, ...] = ()
    confidence: float | None = None
    duplicate_item_id: str | None = None
    accepted_item_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    reviewed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["aliases"] = list(self.aliases)
        value["tags"] = list(self.tags)
        value["source_segment_ids"] = list(self.source_segment_ids)
        return value


@dataclass(frozen=True, slots=True)
class SourceAssessment:
    document_id: str
    source_class: str
    reliability: str
    extraction_completeness: float | None = None
    published_at: str | None = None
    valid_until: str | None = None
    notes: str = ""
    checked_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WorkflowTemplate:
    id: str
    name: str
    description: str
    trigger: dict[str, Any]
    steps: tuple[str, ...]
    model_policy: dict[str, Any]
    privacy: dict[str, Any]
    status: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["steps"] = list(self.steps)
        return value


class KnowledgeOperationsRepository:
    """Structured operational layer; no policy value is ever executed as code."""

    def __init__(self, database: KnowledgeDatabase):
        self.database = database
        self.connection = database.connection

    def get_policy(self, policy_id: str = "local-default") -> KnowledgeSpacePolicy:
        clean_id = _text(policy_id, required=True, name="知识空间 ID")
        row = self.connection.execute(
            "SELECT * FROM knowledge_space_policies WHERE id=?", (clean_id,)
        ).fetchone()
        if row is None:
            return self.upsert_policy(clean_id, clean_id, DEFAULT_POLICY)
        policy = {**DEFAULT_POLICY, **_mapping(row["policy_json"])}
        return KnowledgeSpacePolicy(
            id=str(row["id"]), name=str(row["name"]), policy=policy,
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    def upsert_policy(
        self, policy_id: str, name: str, policy: Mapping[str, object]
    ) -> KnowledgeSpacePolicy:
        clean_id = _text(policy_id, required=True, name="知识空间 ID")
        clean_name = _text(name, required=True, name="知识空间名称")
        merged = {**DEFAULT_POLICY, **dict(policy)}
        merged["auto_propose"] = bool(merged.get("auto_propose", True))
        merged["external_verification"] = bool(merged.get("external_verification", False))
        merged["require_review"] = bool(merged.get("require_review", True))
        conflict = _text(merged.get("conflict_policy") or "warn")
        if conflict not in CONFLICT_POLICIES:
            raise ValueError("冲突策略必须是 warn、require-review 或 manual")
        merged["conflict_policy"] = conflict
        reliability = _text(merged.get("default_source_reliability") or "unassessed")
        if reliability not in SOURCE_RELIABILITIES:
            raise ValueError("默认来源可靠性无效")
        merged["default_source_reliability"] = reliability
        # Explicit allow-list prevents arbitrary instructions from becoming executable policy.
        allowed = {
            "auto_propose", "conflict_policy", "default_source_reliability",
            "external_verification", "require_review", "routing",
            "answer_model", "transcription_profile", "glossary_ids",
        }
        safe = {key: value for key, value in merged.items() if key in allowed}
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """INSERT INTO knowledge_space_policies(id, name, policy_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                       policy_json=excluded.policy_json, updated_at=excluded.updated_at""",
                (clean_id, clean_name, _json(safe), now, now),
            )
        return self.get_policy(clean_id)

    def create_proposal(
        self,
        *,
        title: str,
        body: str,
        proposed_type: str = "topic",
        summary: str = "",
        source_document_id: str | None = None,
        correction_run_id: str | None = None,
        aliases: Sequence[str] = (),
        tags: Sequence[str] = (),
        source_segment_ids: Sequence[str] = (),
        confidence: float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> KnowledgeProposal:
        clean_title = _text(title, required=True, name="候选标题")
        clean_body = _text(body, required=True, name="候选内容")
        if proposed_type not in {"topic", "entity", "analysis", "decision"}:
            raise ValueError("候选知识类型无效")
        if confidence is not None and not 0 <= float(confidence) <= 1:
            raise ValueError("候选置信度必须在 0 到 1 之间")
        source_document_id = _text(source_document_id) or None
        correction_run_id = _text(correction_run_id) or None
        fingerprint = hashlib.sha256(
            "\0".join(
                (
                    correction_run_id or "",
                    source_document_id or "",
                    proposed_type,
                    _normalized(clean_title),
                    _normalized(clean_body),
                )
            ).encode("utf-8")
        ).hexdigest()
        existing = self.connection.execute(
            "SELECT * FROM knowledge_proposals WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if existing is not None:
            return self._proposal(existing)
        duplicate = self._find_duplicate(clean_title, aliases)
        now = utcnow_iso()
        proposal_id = f"proposal-{fingerprint[:24]}"
        with self.connection:
            self.connection.execute(
                """INSERT INTO knowledge_proposals(
                       id, fingerprint, source_document_id, correction_run_id,
                       proposed_type, title, summary, body, aliases_json, tags_json,
                       source_segment_ids_json, confidence, duplicate_item_id,
                       metadata_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    proposal_id, fingerprint, source_document_id, correction_run_id,
                    proposed_type, clean_title, _text(summary), clean_body,
                    _json(list(_sequence(aliases))), _json(list(_sequence(tags))),
                    _json(list(_sequence(source_segment_ids))), confidence,
                    duplicate.item_id if duplicate else None, _json(dict(metadata or {})),
                    now, now,
                ),
            )
            self._event(
                "proposal-created", f"创建候选知识：{clean_title}",
                metadata={"proposal_id": proposal_id, "duplicate_item_id": duplicate.item_id if duplicate else None},
            )
        return self.get_proposal(proposal_id)

    def list_proposals(
        self, *, statuses: Sequence[str] = ("proposed",), limit: int = 500
    ) -> list[KnowledgeProposal]:
        selected = tuple(dict.fromkeys(_text(item) for item in statuses if _text(item)))
        if any(item not in PROPOSAL_STATUSES for item in selected):
            raise ValueError("候选状态无效")
        where = ""
        parameters: list[object] = []
        if selected:
            where = f"WHERE status IN ({','.join('?' for _ in selected)})"
            parameters.extend(selected)
        parameters.append(max(1, min(2000, int(limit))))
        rows = self.connection.execute(
            f"SELECT * FROM knowledge_proposals {where} ORDER BY updated_at DESC, id LIMIT ?",
            parameters,
        ).fetchall()
        return [self._proposal(row) for row in rows]

    def get_proposal(self, proposal_id: str) -> KnowledgeProposal:
        row = self.connection.execute(
            "SELECT * FROM knowledge_proposals WHERE id=?", (_text(proposal_id),)
        ).fetchone()
        if row is None:
            raise ValueError("候选知识不存在")
        return self._proposal(row)

    def accept_proposal(self, proposal_id: str, *, merge_duplicate: bool = False) -> KnowledgeItem:
        proposal = self.get_proposal(proposal_id)
        if proposal.status != "proposed":
            raise ValueError("该候选已经处理")
        governance = KnowledgeGovernanceRepository(self.database)
        if merge_duplicate and proposal.duplicate_item_id:
            target = governance.get_item(proposal.duplicate_item_id)
            if target is None:
                raise ValueError("候选标记的重复知识已经不存在")
            combined = target.body.strip()
            if proposal.body not in combined:
                combined = "\n\n".join(value for value in (combined, proposal.body) if value)
            item = governance.update_item(
                target.item_id,
                body=combined,
                summary=target.summary or proposal.summary,
                status="needs-review",
                aliases=tuple(dict.fromkeys((*target.aliases, *proposal.aliases))),
                tags=tuple(dict.fromkeys((*target.tags, *proposal.tags))),
                metadata={**target.metadata, "last_merged_proposal_id": proposal.id},
            )
            final_status = "merged"
        else:
            item = governance.create_item(
                item_type=proposal.proposed_type,
                title=proposal.title,
                summary=proposal.summary or proposal.body[:240],
                body=proposal.body,
                status="needs-review",
                maturity="summarized",
                aliases=proposal.aliases,
                tags=proposal.tags,
                metadata={
                    **proposal.metadata,
                    "proposal_id": proposal.id,
                    "correction_run_id": proposal.correction_run_id,
                    "source_segment_ids": list(proposal.source_segment_ids),
                    "review_required": True,
                },
            )
            final_status = "accepted"
        if proposal.source_document_id:
            source_item = governance.get_item_for_document(proposal.source_document_id)
            if source_item is not None:
                try:
                    governance.create_relation(
                        source_item.item_id, item.item_id, "supports",
                        summary="原始资料支持该候选知识",
                        metadata={"proposal_id": proposal.id, "source_segment_ids": list(proposal.source_segment_ids)},
                    )
                except ValueError as exc:
                    if "UNIQUE constraint failed" not in str(exc):
                        raise
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE knowledge_proposals SET status=?, accepted_item_id=?,
                       updated_at=?, reviewed_at=? WHERE id=?""",
                (final_status, item.item_id, now, now, proposal.id),
            )
            self._event(
                f"proposal-{final_status}",
                f"{('合并' if final_status == 'merged' else '接受')}候选：{proposal.title}",
                item_id=item.item_id,
                metadata={"proposal_id": proposal.id},
            )
        return item

    def reject_proposal(self, proposal_id: str, *, reason: str = "人工判断不入库") -> KnowledgeProposal:
        proposal = self.get_proposal(proposal_id)
        if proposal.status != "proposed":
            raise ValueError("该候选已经处理")
        now = utcnow_iso()
        metadata = {**proposal.metadata, "review_reason": _text(reason)}
        with self.connection:
            self.connection.execute(
                """UPDATE knowledge_proposals SET status='rejected', metadata_json=?,
                       updated_at=?, reviewed_at=? WHERE id=?""",
                (_json(metadata), now, now, proposal.id),
            )
            self._event(
                "proposal-rejected", f"拒绝候选：{proposal.title}",
                metadata={"proposal_id": proposal.id, "reason": _text(reason)},
            )
        return self.get_proposal(proposal.id)

    def get_source_assessment(self, document_id: str) -> SourceAssessment:
        clean_id = _text(document_id, required=True, name="资料 ID")
        row = self.connection.execute(
            "SELECT * FROM source_assessments WHERE document_id=?", (clean_id,)
        ).fetchone()
        if row is None:
            return self.upsert_source_assessment(clean_id)
        return self._assessment(row)

    def upsert_source_assessment(
        self,
        document_id: str,
        *,
        source_class: str = "unassessed",
        reliability: str = "unassessed",
        extraction_completeness: float | None = None,
        published_at: str | None = None,
        valid_until: str | None = None,
        notes: str = "",
        checked: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> SourceAssessment:
        clean_id = _text(document_id, required=True, name="资料 ID")
        if self.database.get_document(clean_id) is None:
            raise ValueError("资料不存在")
        if source_class not in SOURCE_CLASSES or reliability not in SOURCE_RELIABILITIES:
            raise ValueError("来源类型或可靠性无效")
        if extraction_completeness is not None and not 0 <= float(extraction_completeness) <= 1:
            raise ValueError("解析完整度必须在 0 到 1 之间")
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """INSERT INTO source_assessments(
                       document_id, source_class, reliability, extraction_completeness,
                       published_at, valid_until, notes, checked_at, metadata_json,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                       source_class=excluded.source_class,
                       reliability=excluded.reliability,
                       extraction_completeness=excluded.extraction_completeness,
                       published_at=excluded.published_at, valid_until=excluded.valid_until,
                       notes=excluded.notes, checked_at=excluded.checked_at,
                       metadata_json=excluded.metadata_json, updated_at=excluded.updated_at""",
                (
                    clean_id, source_class, reliability, extraction_completeness,
                    _text(published_at) or None, _text(valid_until) or None, _text(notes),
                    now if checked else None, _json(dict(metadata or {})), now, now,
                ),
            )
        return self.get_source_assessment(clean_id)

    def source_quality_issues(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """SELECT d.id, d.title, a.* FROM documents d
               LEFT JOIN source_assessments a ON a.document_id=d.id
               ORDER BY d.updated_at DESC"""
        ).fetchall()
        today = date.today()
        issues: list[dict[str, object]] = []
        for row in rows:
            reliability = str(row["reliability"] or "unassessed")
            if reliability == "unassessed":
                issues.append({"code": "source_unassessed", "severity": "warning", "document_id": row["id"], "title": row["title"], "message": "尚未评估来源可靠性"})
            if reliability == "low":
                issues.append({"code": "source_low_reliability", "severity": "warning", "document_id": row["id"], "title": row["title"], "message": "来源可靠性为低，回答时应交叉核验"})
            completeness = row["extraction_completeness"]
            if completeness is not None and float(completeness) < 0.7:
                issues.append({"code": "source_incomplete", "severity": "error", "document_id": row["id"], "title": row["title"], "message": f"解析完整度仅 {float(completeness):.0%}"})
            valid_until = str(row["valid_until"] or "").strip()
            try:
                expired = bool(valid_until) and date.fromisoformat(valid_until[:10]) < today
            except ValueError:
                expired = False
                issues.append({"code": "source_invalid_date", "severity": "warning", "document_id": row["id"], "title": row["title"], "message": f"来源有效期格式无效：{valid_until}"})
            if expired:
                issues.append({"code": "source_expired", "severity": "warning", "document_id": row["id"], "title": row["title"], "message": f"来源有效期已过：{valid_until[:10]}"})
        return issues

    def upsert_workflow(
        self,
        *,
        name: str,
        description: str = "",
        trigger: Mapping[str, object] | None = None,
        steps: Sequence[str] = (),
        model_policy: Mapping[str, object] | None = None,
        privacy: Mapping[str, object] | None = None,
        status: str = "current",
        workflow_id: str | None = None,
    ) -> WorkflowTemplate:
        clean_name = _text(name, required=True, name="流程名称")
        if status not in {"current", "archived"}:
            raise ValueError("流程状态无效")
        clean_steps = _sequence(steps)
        if not clean_steps:
            raise ValueError("流程至少需要一个步骤")
        clean_id = _text(workflow_id) or f"workflow-{uuid4().hex}"
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """INSERT INTO workflow_templates(
                       id, name, description, trigger_json, steps_json,
                       model_policy_json, privacy_json, status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                       description=excluded.description, trigger_json=excluded.trigger_json,
                       steps_json=excluded.steps_json, model_policy_json=excluded.model_policy_json,
                       privacy_json=excluded.privacy_json, status=excluded.status,
                       updated_at=excluded.updated_at""",
                (
                    clean_id, clean_name, _text(description), _json(dict(trigger or {})),
                    _json(list(clean_steps)), _json(dict(model_policy or {})),
                    _json(dict(privacy or {})), status, now, now,
                ),
            )
            self._event("workflow-saved", f"保存 SOP：{clean_name}", metadata={"workflow_id": clean_id})
        return self.get_workflow(clean_id)

    def get_workflow(self, workflow_id: str) -> WorkflowTemplate:
        row = self.connection.execute(
            "SELECT * FROM workflow_templates WHERE id=?", (_text(workflow_id),)
        ).fetchone()
        if row is None:
            raise ValueError("SOP 流程不存在")
        return self._workflow(row)

    def list_workflows(self, *, include_archived: bool = False) -> list[WorkflowTemplate]:
        where = "" if include_archived else "WHERE status='current'"
        rows = self.connection.execute(
            f"SELECT * FROM workflow_templates {where} ORDER BY updated_at DESC, name, id"
        ).fetchall()
        return [self._workflow(row) for row in rows]

    def list_events(self, *, limit: int = 200) -> list[dict[str, object]]:
        rows = self.connection.execute(
            "SELECT * FROM knowledge_events ORDER BY created_at DESC, id LIMIT ?",
            (max(1, min(2000, int(limit))),),
        ).fetchall()
        return [
            {
                "id": row["id"], "item_id": row["item_id"],
                "event_type": row["event_type"], "summary": row["summary"],
                "metadata": _mapping(row["metadata_json"]), "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _find_duplicate(self, title: str, aliases: Sequence[str]) -> KnowledgeItem | None:
        terms = {_normalized(title), *(_normalized(item) for item in aliases)}
        terms.discard("")
        governance = KnowledgeGovernanceRepository(self.database)
        for term in sorted(terms):
            matches = governance.search(term, limit=10)
            for item in matches:
                names = {_normalized(item.title), *(_normalized(value) for value in item.aliases)}
                if term in names:
                    return item
        return None

    def _event(
        self,
        event_type: str,
        summary: str,
        *,
        item_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.connection.execute(
            """INSERT INTO knowledge_events(id, item_id, event_type, summary, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (f"event-{uuid4().hex}", item_id, event_type, _text(summary), _json(dict(metadata or {})), utcnow_iso()),
        )

    @staticmethod
    def _proposal(row: sqlite3.Row) -> KnowledgeProposal:
        return KnowledgeProposal(
            id=str(row["id"]), fingerprint=str(row["fingerprint"]),
            proposed_type=str(row["proposed_type"]), title=str(row["title"]),
            summary=str(row["summary"]), body=str(row["body"]), status=str(row["status"]),
            source_document_id=row["source_document_id"], correction_run_id=row["correction_run_id"],
            aliases=_sequence(json.loads(row["aliases_json"] or "[]")),
            tags=_sequence(json.loads(row["tags_json"] or "[]")),
            source_segment_ids=_sequence(json.loads(row["source_segment_ids_json"] or "[]")),
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            duplicate_item_id=row["duplicate_item_id"], accepted_item_id=row["accepted_item_id"],
            metadata=_mapping(row["metadata_json"]), created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]), reviewed_at=row["reviewed_at"],
        )

    @staticmethod
    def _assessment(row: sqlite3.Row) -> SourceAssessment:
        return SourceAssessment(
            document_id=str(row["document_id"]), source_class=str(row["source_class"]),
            reliability=str(row["reliability"]),
            extraction_completeness=float(row["extraction_completeness"]) if row["extraction_completeness"] is not None else None,
            published_at=row["published_at"], valid_until=row["valid_until"], notes=str(row["notes"]),
            checked_at=row["checked_at"], metadata=_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _workflow(row: sqlite3.Row) -> WorkflowTemplate:
        return WorkflowTemplate(
            id=str(row["id"]), name=str(row["name"]), description=str(row["description"]),
            trigger=_mapping(row["trigger_json"]),
            steps=_sequence(json.loads(row["steps_json"] or "[]")),
            model_policy=_mapping(row["model_policy_json"]), privacy=_mapping(row["privacy_json"]),
            status=str(row["status"]), created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )


__all__ = [
    "DEFAULT_POLICY", "KnowledgeOperationsRepository", "KnowledgeProposal",
    "KnowledgeSpacePolicy", "SourceAssessment", "WorkflowTemplate",
]
