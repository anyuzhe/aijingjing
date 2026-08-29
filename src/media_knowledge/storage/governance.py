from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from ..models import utcnow_iso
from .database import KnowledgeDatabase


KNOWLEDGE_ITEM_TYPES = frozenset(
    {"source", "topic", "entity", "analysis", "decision", "output"}
)
KNOWLEDGE_STATUSES = frozenset(
    {"draft", "current", "needs-review", "stale", "archived"}
)
KNOWLEDGE_MATURITIES = frozenset(
    {"unreviewed", "indexed", "summarized", "compiled", "low-value"}
)
KNOWLEDGE_RELATION_TYPES = frozenset(
    {"supports", "extends", "contradicts", "supersedes", "opens"}
)

_UNSET = object()
_SPACE_RE = re.compile(r"\s+")
_CANONICAL_TAG_RE = re.compile(r"^[a-z0-9\u3400-\u9fff]+(?:-[a-z0-9\u3400-\u9fff]+)*$")


def _clean_text(value: Any, *, required: bool = False, field_name: str = "文本") -> str:
    clean = str(value or "").strip()
    if required and not clean:
        raise ValueError(f"{field_name}不能为空")
    return clean


def _normalize(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip().casefold()


def _load_mapping(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dump_mapping(value: dict[str, Any] | None) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, dict):
        raise ValueError("metadata 必须是字典")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata 必须可以序列化为 JSON") from exc


def _validate_choice(value: str, allowed: frozenset[str], field_name: str) -> str:
    clean = _clean_text(value, required=True, field_name=field_name)
    if clean not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(f"不支持的{field_name}: {clean}；可选值：{options}")
    return clean


def _bounded_page(limit: int, offset: int, *, maximum: int = 1000) -> tuple[int, int]:
    try:
        safe_limit = int(limit)
        safe_offset = int(offset)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit 和 offset 必须是整数") from exc
    if safe_limit < 1:
        raise ValueError("limit 必须大于 0")
    if safe_offset < 0:
        raise ValueError("offset 不能小于 0")
    return min(safe_limit, maximum), safe_offset


@dataclass(slots=True, frozen=True)
class KnowledgeItem:
    item_id: str
    item_type: str
    status: str
    maturity: str
    title: str
    summary: str = ""
    body: str = ""
    document_id: str | None = None
    artifact_id: str | None = None
    high_value: bool = False
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def id(self) -> str:
        """Compatibility alias for callers that use database-style ``id`` names."""

        return self.item_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["id"] = value.pop("item_id")
        value["aliases"] = list(self.aliases)
        value["tags"] = list(self.tags)
        return value


@dataclass(slots=True, frozen=True)
class KnowledgeRelation:
    relation_id: str
    source_item_id: str
    target_item_id: str
    relation_type: str
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def id(self) -> str:
        return self.relation_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["id"] = value.pop("relation_id")
        return value


@dataclass(slots=True, frozen=True)
class KnowledgeRestoreResult:
    """Result of restoring one governed item from a repository snapshot."""

    item: KnowledgeItem
    restored_relations: tuple[KnowledgeRelation, ...] = ()
    skipped_relation_ids: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class RelatedKnowledgeItem:
    item: KnowledgeItem
    relation: KnowledgeRelation
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "relation": self.relation.to_dict(),
            "direction": self.direction,
        }


@dataclass(slots=True, frozen=True)
class KnowledgeHealthIssue:
    code: str
    severity: str
    item_id: str | None
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class KnowledgeHealthReport:
    generated_at: str
    total_items: int
    counts: dict[str, dict[str, int]]
    issues: tuple[KnowledgeHealthIssue, ...] = ()

    @property
    def healthy(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "healthy": self.healthy,
            "total_items": self.total_items,
            "counts": self.counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class KnowledgeGovernanceRepository:
    """CRUD, graph traversal and deterministic health checks for governed knowledge."""

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    @property
    def connection(self) -> sqlite3.Connection:
        return self.database.connection

    def create_item(
        self,
        *,
        item_type: str,
        title: str,
        status: str = "draft",
        maturity: str = "unreviewed",
        summary: str = "",
        body: str = "",
        document_id: str | None = None,
        artifact_id: str | None = None,
        high_value: bool = False,
        aliases: Sequence[str] = (),
        tags: Sequence[str] = (),
        metadata: dict[str, Any] | None = None,
        item_id: str | None = None,
    ) -> KnowledgeItem:
        item_type = _validate_choice(item_type, KNOWLEDGE_ITEM_TYPES, "知识类型")
        status = _validate_choice(status, KNOWLEDGE_STATUSES, "状态")
        maturity = _validate_choice(maturity, KNOWLEDGE_MATURITIES, "摄取成熟度")
        title = _clean_text(title, required=True, field_name="标题")
        clean_id = _clean_text(item_id or f"kg-{uuid4().hex}", required=True, field_name="知识 ID")
        document_id = _clean_text(document_id) or None
        artifact_id = _clean_text(artifact_id) or None
        self._validate_links(item_type, document_id, artifact_id)
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO knowledge_items(
                           id, item_type, status, maturity, title, summary, body,
                           document_id, artifact_id, high_value, metadata_json,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        clean_id,
                        item_type,
                        status,
                        maturity,
                        title,
                        _clean_text(summary),
                        _clean_text(body),
                        document_id,
                        artifact_id,
                        1 if high_value else 0,
                        _dump_mapping(metadata),
                        now,
                        now,
                    ),
                )
                self._replace_aliases(clean_id, aliases, now)
                self._replace_tags(clean_id, tags, now)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法创建知识条目：{exc}") from exc
        return self._require_item(clean_id)

    def get_item(self, item_id: str) -> KnowledgeItem | None:
        row = self.connection.execute(
            "SELECT * FROM knowledge_items WHERE id=?", (_clean_text(item_id),)
        ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def get_item_for_document(self, document_id: str) -> KnowledgeItem | None:
        row = self.connection.execute(
            """SELECT * FROM knowledge_items
               WHERE document_id=? AND item_type='source' LIMIT 1""",
            (_clean_text(document_id),),
        ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def get_item_for_artifact(self, artifact_id: str) -> KnowledgeItem | None:
        row = self.connection.execute(
            """SELECT * FROM knowledge_items
               WHERE artifact_id=? AND item_type='output' LIMIT 1""",
            (_clean_text(artifact_id),),
        ).fetchone()
        return self._item_from_row(row) if row is not None else None

    def update_item(
        self,
        item_id: str,
        *,
        item_type: str | object = _UNSET,
        title: str | object = _UNSET,
        status: str | object = _UNSET,
        maturity: str | object = _UNSET,
        summary: str | object = _UNSET,
        body: str | object = _UNSET,
        document_id: str | None | object = _UNSET,
        artifact_id: str | None | object = _UNSET,
        high_value: bool | object = _UNSET,
        aliases: Sequence[str] | object = _UNSET,
        tags: Sequence[str] | object = _UNSET,
        metadata: dict[str, Any] | object = _UNSET,
    ) -> KnowledgeItem:
        current = self._require_item(item_id)
        next_type = (
            current.item_type
            if item_type is _UNSET
            else _validate_choice(str(item_type), KNOWLEDGE_ITEM_TYPES, "知识类型")
        )
        next_document_id = (
            current.document_id
            if document_id is _UNSET
            else (_clean_text(document_id) or None)
        )
        next_artifact_id = (
            current.artifact_id
            if artifact_id is _UNSET
            else (_clean_text(artifact_id) or None)
        )
        self._validate_links(next_type, next_document_id, next_artifact_id)
        values: dict[str, Any] = {
            "item_type": next_type,
            "title": current.title
            if title is _UNSET
            else _clean_text(title, required=True, field_name="标题"),
            "status": current.status
            if status is _UNSET
            else _validate_choice(str(status), KNOWLEDGE_STATUSES, "状态"),
            "maturity": current.maturity
            if maturity is _UNSET
            else _validate_choice(str(maturity), KNOWLEDGE_MATURITIES, "摄取成熟度"),
            "summary": current.summary if summary is _UNSET else _clean_text(summary),
            "body": current.body if body is _UNSET else _clean_text(body),
            "document_id": next_document_id,
            "artifact_id": next_artifact_id,
            "high_value": int(current.high_value if high_value is _UNSET else bool(high_value)),
            "metadata_json": _dump_mapping(current.metadata if metadata is _UNSET else metadata),
            "updated_at": utcnow_iso(),
        }
        try:
            with self.connection:
                self.connection.execute(
                    """UPDATE knowledge_items SET
                           item_type=:item_type, title=:title, status=:status,
                           maturity=:maturity, summary=:summary, body=:body,
                           document_id=:document_id, artifact_id=:artifact_id,
                           high_value=:high_value, metadata_json=:metadata_json,
                           updated_at=:updated_at
                       WHERE id=:item_id""",
                    {**values, "item_id": current.item_id},
                )
                if aliases is not _UNSET:
                    self._replace_aliases(current.item_id, aliases, values["updated_at"])
                if tags is not _UNSET:
                    self._replace_tags(current.item_id, tags, values["updated_at"])
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法更新知识条目：{exc}") from exc
        return self._require_item(current.item_id)

    def delete_item(self, item_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM knowledge_items WHERE id=?", (_clean_text(item_id),)
            )
        return cursor.rowcount > 0

    def snapshot_item(self, item_id: str) -> dict[str, Any]:
        """Return all rows that would be removed by deleting one governed item.

        The snapshot deliberately contains the original alias/tag timestamps and every
        incoming or outgoing relation.  It is suitable for a durable tombstone, while
        filesystem-owned note information remains the controller's responsibility.
        """

        item = self._require_item(item_id)
        aliases = self.connection.execute(
            """SELECT alias, normalized_alias, created_at FROM knowledge_aliases
               WHERE item_id=? ORDER BY normalized_alias, alias""",
            (item.item_id,),
        ).fetchall()
        tags = self.connection.execute(
            """SELECT tag, normalized_tag, created_at FROM knowledge_item_tags
               WHERE item_id=? ORDER BY normalized_tag, tag""",
            (item.item_id,),
        ).fetchall()
        relations = self.connection.execute(
            """SELECT * FROM knowledge_relations
               WHERE source_item_id=? OR target_item_id=?
               ORDER BY updated_at DESC, id""",
            (item.item_id, item.item_id),
        ).fetchall()
        return {
            "item": item.to_dict(),
            "aliases": [dict(row) for row in aliases],
            "tags": [dict(row) for row in tags],
            "relations": [
                self._relation_from_row(row).to_dict() for row in relations
            ],
        }

    def restore_item_snapshot(
        self, snapshot: Mapping[str, Any]
    ) -> KnowledgeRestoreResult:
        """Atomically restore an item and relations whose other endpoints still exist."""

        raw_item = snapshot.get("item")
        if not isinstance(raw_item, Mapping):
            raise ValueError("知识回收站记录缺少 item")
        item_id = _clean_text(
            raw_item.get("id") or raw_item.get("item_id"),
            required=True,
            field_name="知识 ID",
        )
        if self.get_item(item_id) is not None:
            raise ValueError(f"知识条目已存在，不能重复恢复: {item_id}")
        item_type = _validate_choice(
            str(raw_item.get("item_type") or ""), KNOWLEDGE_ITEM_TYPES, "知识类型"
        )
        status = _validate_choice(
            str(raw_item.get("status") or ""), KNOWLEDGE_STATUSES, "状态"
        )
        maturity = _validate_choice(
            str(raw_item.get("maturity") or ""),
            KNOWLEDGE_MATURITIES,
            "摄取成熟度",
        )
        title = _clean_text(raw_item.get("title"), required=True, field_name="标题")
        document_id = _clean_text(raw_item.get("document_id")) or None
        artifact_id = _clean_text(raw_item.get("artifact_id")) or None
        self._validate_links(item_type, document_id, artifact_id)
        metadata = raw_item.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("知识回收站中的 metadata 必须是字典")
        created_at = _clean_text(raw_item.get("created_at")) or utcnow_iso()
        updated_at = _clean_text(raw_item.get("updated_at")) or created_at
        alias_rows = self._snapshot_values(
            snapshot.get("aliases", raw_item.get("aliases", ())),
            value_key="alias",
            fallback_created_at=created_at,
            field_name="别名",
        )
        tag_rows = self._snapshot_values(
            snapshot.get("tags", raw_item.get("tags", ())),
            value_key="tag",
            fallback_created_at=created_at,
            field_name="标签",
        )
        raw_relations = snapshot.get("relations", ())
        if not isinstance(raw_relations, (list, tuple)):
            raise ValueError("知识回收站中的 relations 必须是列表")
        restored_relation_ids: list[str] = []
        skipped_relation_ids: list[str] = []
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO knowledge_items(
                           id, item_type, status, maturity, title, summary, body,
                           document_id, artifact_id, high_value, metadata_json,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item_id,
                        item_type,
                        status,
                        maturity,
                        title,
                        _clean_text(raw_item.get("summary")),
                        _clean_text(raw_item.get("body")),
                        document_id,
                        artifact_id,
                        1 if bool(raw_item.get("high_value")) else 0,
                        _dump_mapping(metadata),
                        created_at,
                        updated_at,
                    ),
                )
                self.connection.executemany(
                    """INSERT INTO knowledge_aliases(
                           item_id, alias, normalized_alias, created_at
                       ) VALUES (?, ?, ?, ?)""",
                    ((item_id, value, normalized, timestamp) for value, normalized, timestamp in alias_rows),
                )
                self.connection.executemany(
                    """INSERT INTO knowledge_item_tags(
                           item_id, tag, normalized_tag, created_at
                       ) VALUES (?, ?, ?, ?)""",
                    ((item_id, value, normalized, timestamp) for value, normalized, timestamp in tag_rows),
                )
                for raw_relation in raw_relations:
                    if not isinstance(raw_relation, Mapping):
                        raise ValueError("知识回收站中存在无效关系记录")
                    relation_id = _clean_text(
                        raw_relation.get("id") or raw_relation.get("relation_id"),
                        required=True,
                        field_name="关系 ID",
                    )
                    source_id = _clean_text(
                        raw_relation.get("source_item_id"),
                        required=True,
                        field_name="源知识 ID",
                    )
                    target_id = _clean_text(
                        raw_relation.get("target_item_id"),
                        required=True,
                        field_name="目标知识 ID",
                    )
                    relation_type = _validate_choice(
                        str(raw_relation.get("relation_type") or ""),
                        KNOWLEDGE_RELATION_TYPES,
                        "关系类型",
                    )
                    if item_id not in {source_id, target_id}:
                        raise ValueError("知识回收站中的关系与待恢复条目无关")
                    endpoints = self.connection.execute(
                        "SELECT id FROM knowledge_items WHERE id IN (?, ?)",
                        (source_id, target_id),
                    ).fetchall()
                    if len(endpoints) != 2:
                        skipped_relation_ids.append(relation_id)
                        continue
                    relation_metadata = raw_relation.get("metadata")
                    if relation_metadata is not None and not isinstance(relation_metadata, dict):
                        raise ValueError("知识关系 metadata 必须是字典")
                    cursor = self.connection.execute(
                        """INSERT OR IGNORE INTO knowledge_relations(
                               id, source_item_id, target_item_id, relation_type,
                               summary, metadata_json, created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            relation_id,
                            source_id,
                            target_id,
                            relation_type,
                            _clean_text(raw_relation.get("summary")),
                            _dump_mapping(relation_metadata),
                            _clean_text(raw_relation.get("created_at")) or created_at,
                            _clean_text(raw_relation.get("updated_at")) or updated_at,
                        ),
                    )
                    if cursor.rowcount:
                        restored_relation_ids.append(relation_id)
                    else:
                        skipped_relation_ids.append(relation_id)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法恢复知识条目：{exc}") from exc
        restored = self._require_item(item_id)
        return KnowledgeRestoreResult(
            item=restored,
            restored_relations=tuple(
                self._require_relation(relation_id)
                for relation_id in restored_relation_ids
            ),
            skipped_relation_ids=tuple(skipped_relation_ids),
        )

    def list_items(
        self,
        *,
        item_types: Sequence[str] = (),
        statuses: Sequence[str] = (),
        maturities: Sequence[str] = (),
        tags: Sequence[str] = (),
        high_value: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[KnowledgeItem]:
        safe_limit, safe_offset = _bounded_page(limit, offset)
        where, parameters = self._item_filters(
            item_types=item_types,
            statuses=statuses,
            maturities=maturities,
            tags=tags,
            high_value=high_value,
        )
        rows = self.connection.execute(
            f"""SELECT ki.* FROM knowledge_items ki
                {where}
                ORDER BY ki.updated_at DESC, ki.title COLLATE NOCASE, ki.id
                LIMIT ? OFFSET ?""",
            (*parameters, safe_limit, safe_offset),
        ).fetchall()
        return [self._item_from_row(row) for row in rows]

    def search(
        self,
        query: str,
        *,
        item_types: Sequence[str] = (),
        statuses: Sequence[str] = (),
        maturities: Sequence[str] = (),
        tags: Sequence[str] = (),
        high_value: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeItem]:
        """Search titles, aliases, summaries, bodies and tags with deterministic ranking."""

        term = _normalize(query)
        if not term:
            return self.list_items(
                item_types=item_types,
                statuses=statuses,
                maturities=maturities,
                tags=tags,
                high_value=high_value,
                limit=limit,
                offset=offset,
            )
        safe_limit, safe_offset = _bounded_page(limit, offset)
        # Fetch the filtered governance corpus so Python's Unicode casefold semantics
        # are consistent for Chinese aliases and non-ASCII names across SQLite builds.
        where, parameters = self._item_filters(
            item_types=item_types,
            statuses=statuses,
            maturities=maturities,
            tags=tags,
            high_value=high_value,
        )
        candidate_rows = self.connection.execute(
            f"""SELECT ki.* FROM knowledge_items ki {where}
                ORDER BY ki.updated_at DESC, ki.id""",
            parameters,
        ).fetchall()
        candidates = [self._item_from_row(row) for row in candidate_rows]
        terms = tuple(part for part in term.split(" ") if part)

        def rank(item: KnowledgeItem) -> tuple[int, str, str]:
            title = _normalize(item.title)
            aliases = tuple(_normalize(value) for value in item.aliases)
            summary = _normalize(item.summary)
            body = _normalize(item.body)
            item_tags = tuple(_normalize(value) for value in item.tags)
            searchable = " ".join((title, *aliases, summary, body, *item_tags))
            if not all(part in searchable for part in terms):
                return (-1, item.updated_at, item.item_id)
            score = 0
            if title == term:
                score += 1000
            if term in aliases:
                score += 900
            if title.startswith(term):
                score += 500
            score += sum(220 for alias in aliases if alias.startswith(term))
            score += sum(140 for part in terms if part in title)
            score += sum(100 for part in terms if any(part in alias for alias in aliases))
            score += sum(45 for part in terms if part in item_tags)
            score += sum(25 for part in terms if part in summary)
            score += sum(8 for part in terms if part in body)
            return (score, item.updated_at, item.item_id)

        ranked = [(rank(item), item) for item in candidates]
        ranked = [pair for pair in ranked if pair[0][0] >= 0]
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[safe_offset : safe_offset + safe_limit]]

    def add_alias(self, item_id: str, alias: str) -> bool:
        self._require_item(item_id)
        clean = _clean_text(alias, required=True, field_name="别名")
        with self.connection:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO knowledge_aliases(
                       item_id, alias, normalized_alias, created_at
                   ) VALUES (?, ?, ?, ?)""",
                (item_id, clean, _normalize(clean), utcnow_iso()),
            )
        return cursor.rowcount > 0

    def remove_alias(self, item_id: str, alias: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM knowledge_aliases WHERE item_id=? AND normalized_alias=?",
                (_clean_text(item_id), _normalize(alias)),
            )
        return cursor.rowcount > 0

    def list_aliases(self, item_id: str) -> list[str]:
        rows = self.connection.execute(
            """SELECT alias FROM knowledge_aliases WHERE item_id=?
               ORDER BY normalized_alias, alias""",
            (_clean_text(item_id),),
        ).fetchall()
        return [str(row["alias"]) for row in rows]

    def add_tag(self, item_id: str, tag: str) -> bool:
        self._require_item(item_id)
        clean = _clean_text(tag, required=True, field_name="标签")
        with self.connection:
            cursor = self.connection.execute(
                """INSERT OR IGNORE INTO knowledge_item_tags(
                       item_id, tag, normalized_tag, created_at
                   ) VALUES (?, ?, ?, ?)""",
                (item_id, clean, _normalize(clean), utcnow_iso()),
            )
        return cursor.rowcount > 0

    def remove_tag(self, item_id: str, tag: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM knowledge_item_tags WHERE item_id=? AND normalized_tag=?",
                (_clean_text(item_id), _normalize(tag)),
            )
        return cursor.rowcount > 0

    def list_tags(self, item_id: str) -> list[str]:
        rows = self.connection.execute(
            """SELECT tag FROM knowledge_item_tags WHERE item_id=?
               ORDER BY normalized_tag, tag""",
            (_clean_text(item_id),),
        ).fetchall()
        return [str(row["tag"]) for row in rows]

    def create_relation(
        self,
        source_item_id: str,
        target_item_id: str,
        relation_type: str,
        *,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        relation_id: str | None = None,
    ) -> KnowledgeRelation:
        source_item_id = _clean_text(source_item_id, required=True, field_name="源知识 ID")
        target_item_id = _clean_text(target_item_id, required=True, field_name="目标知识 ID")
        relation_type = _validate_choice(
            relation_type, KNOWLEDGE_RELATION_TYPES, "关系类型"
        )
        if source_item_id == target_item_id:
            raise ValueError("知识关系不能指向自身")
        self._require_item(source_item_id)
        self._require_item(target_item_id)
        clean_id = _clean_text(
            relation_id or f"rel-{uuid4().hex}", required=True, field_name="关系 ID"
        )
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO knowledge_relations(
                           id, source_item_id, target_item_id, relation_type,
                           summary, metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        clean_id,
                        source_item_id,
                        target_item_id,
                        relation_type,
                        _clean_text(summary),
                        _dump_mapping(metadata),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法创建知识关系：{exc}") from exc
        return self._require_relation(clean_id)

    def get_relation(self, relation_id: str) -> KnowledgeRelation | None:
        row = self.connection.execute(
            "SELECT * FROM knowledge_relations WHERE id=?", (_clean_text(relation_id),)
        ).fetchone()
        return self._relation_from_row(row) if row is not None else None

    def update_relation(
        self,
        relation_id: str,
        *,
        relation_type: str | object = _UNSET,
        summary: str | object = _UNSET,
        metadata: dict[str, Any] | object = _UNSET,
    ) -> KnowledgeRelation:
        relation = self._require_relation(relation_id)
        next_type = (
            relation.relation_type
            if relation_type is _UNSET
            else _validate_choice(
                str(relation_type), KNOWLEDGE_RELATION_TYPES, "关系类型"
            )
        )
        try:
            with self.connection:
                self.connection.execute(
                    """UPDATE knowledge_relations SET relation_type=?, summary=?,
                           metadata_json=?, updated_at=? WHERE id=?""",
                    (
                        next_type,
                        relation.summary if summary is _UNSET else _clean_text(summary),
                        _dump_mapping(
                            relation.metadata if metadata is _UNSET else metadata
                        ),
                        utcnow_iso(),
                        relation.relation_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法更新知识关系：{exc}") from exc
        return self._require_relation(relation.relation_id)

    def delete_relation(self, relation_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM knowledge_relations WHERE id=?", (_clean_text(relation_id),)
            )
        return cursor.rowcount > 0

    def list_relations(
        self,
        item_id: str | None = None,
        *,
        direction: str = "both",
        relation_types: Sequence[str] = (),
        limit: int = 200,
        offset: int = 0,
    ) -> list[KnowledgeRelation]:
        safe_limit, safe_offset = _bounded_page(limit, offset)
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("direction 必须是 outgoing、incoming 或 both")
        clauses: list[str] = []
        parameters: list[Any] = []
        if item_id is not None:
            clean_id = _clean_text(item_id, required=True, field_name="知识 ID")
            if direction == "outgoing":
                clauses.append("source_item_id=?")
                parameters.append(clean_id)
            elif direction == "incoming":
                clauses.append("target_item_id=?")
                parameters.append(clean_id)
            else:
                clauses.append("(source_item_id=? OR target_item_id=?)")
                parameters.extend((clean_id, clean_id))
        types = self._validated_values(
            relation_types, KNOWLEDGE_RELATION_TYPES, "关系类型"
        )
        if types:
            clauses.append(f"relation_type IN ({','.join('?' for _ in types)})")
            parameters.extend(types)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"""SELECT * FROM knowledge_relations {where}
                ORDER BY updated_at DESC, id LIMIT ? OFFSET ?""",
            (*parameters, safe_limit, safe_offset),
        ).fetchall()
        return [self._relation_from_row(row) for row in rows]

    def related_items(
        self,
        item_id: str,
        *,
        direction: str = "both",
        relation_types: Sequence[str] = (),
        limit: int = 200,
    ) -> list[RelatedKnowledgeItem]:
        clean_id = _clean_text(item_id, required=True, field_name="知识 ID")
        self._require_item(clean_id)
        results: list[RelatedKnowledgeItem] = []
        for relation in self.list_relations(
            clean_id,
            direction=direction,
            relation_types=relation_types,
            limit=limit,
        ):
            is_outgoing = relation.source_item_id == clean_id
            other_id = relation.target_item_id if is_outgoing else relation.source_item_id
            item = self.get_item(other_id)
            if item is not None:
                results.append(
                    RelatedKnowledgeItem(
                        item=item,
                        relation=relation,
                        direction="outgoing" if is_outgoing else "incoming",
                    )
                )
        return results

    def health_report(
        self,
        *,
        stale_after_days: int = 120,
        as_of: datetime | None = None,
    ) -> KnowledgeHealthReport:
        try:
            days = int(stale_after_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("stale_after_days 必须是整数") from exc
        if days < 1:
            raise ValueError("stale_after_days 必须大于 0")
        now = as_of or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        else:
            now = now.astimezone(UTC)
        cutoff = now - timedelta(days=days)
        item_rows = self.connection.execute(
            "SELECT * FROM knowledge_items ORDER BY updated_at DESC, id"
        ).fetchall()
        items = [self._item_from_row(row) for row in item_rows]
        issues: list[KnowledgeHealthIssue] = []
        relation_counts = {
            str(row["item_id"]): int(row["relation_count"])
            for row in self.connection.execute(
                """SELECT item_id, COUNT(*) AS relation_count FROM (
                       SELECT source_item_id AS item_id FROM knowledge_relations
                       UNION ALL SELECT target_item_id AS item_id FROM knowledge_relations
                   ) GROUP BY item_id"""
            ).fetchall()
        }
        sourced_items = {
            str(row["item_id"])
            for row in self.connection.execute(
                """WITH RECURSIVE provenance(item_id, ancestor_id) AS (
                       SELECT target_item_id, source_item_id
                       FROM knowledge_relations
                       WHERE relation_type IN ('supports', 'extends')
                       UNION
                       SELECT provenance.item_id, relation.source_item_id
                       FROM provenance
                       JOIN knowledge_relations relation
                         ON relation.target_item_id=provenance.ancestor_id
                       WHERE relation.relation_type IN ('supports', 'extends')
                   )
                   SELECT DISTINCT provenance.item_id
                   FROM provenance
                   JOIN knowledge_items ancestor ON ancestor.id=provenance.ancestor_id
                   WHERE ancestor.item_type='source'"""
            ).fetchall()
        }
        for item in items:
            active = item.status != "archived"
            relation_count = relation_counts.get(item.item_id, 0)
            if active and not item.metadata:
                issues.append(self._issue(item, "missing_metadata", "info", "缺少结构化元数据"))
            if active and item.maturity in {"summarized", "compiled"} and not item.summary:
                issues.append(self._issue(item, "missing_summary", "warning", "成熟知识缺少摘要"))
            if (
                active
                and item.item_type in {"topic", "entity", "analysis", "decision"}
                and not item.body
            ):
                issues.append(self._issue(item, "missing_body", "warning", "知识正文为空"))
            if active and item.item_type == "source" and not self._has_source_evidence(item):
                issues.append(
                    self._issue(item, "source_without_evidence", "error", "来源知识没有原始资料或来源地址")
                )
            if active and item.item_type != "source" and relation_count == 0:
                issues.append(self._issue(item, "orphan_item", "warning", "知识条目未连接到知识图谱"))
            if (
                active
                and item.item_type == "source"
                and item.maturity in {"summarized", "compiled"}
                and relation_count == 0
            ):
                issues.append(self._issue(item, "isolated_source", "info", "已处理来源尚未支持任何知识"))
            if item.status == "current" and self._older_than(item.updated_at, cutoff):
                issues.append(
                    self._issue(
                        item,
                        "stale_current",
                        "warning",
                        f"当前知识超过 {days} 天未更新",
                        {"stale_after_days": days, "updated_at": item.updated_at},
                    )
                )
            if item.status == "stale":
                issues.append(self._issue(item, "marked_stale", "warning", "知识已标记为过期"))
            if item.item_type == "source" and item.high_value and item.maturity != "compiled":
                issues.append(
                    self._issue(
                        item,
                        "high_value_uncompiled",
                        "warning",
                        "高价值来源尚未完成知识编译",
                        {"maturity": item.maturity},
                    )
                )
            if (
                active
                and item.item_type != "source"
                and item.maturity == "compiled"
                and item.item_id not in sourced_items
            ):
                issues.append(
                    self._issue(item, "compiled_without_source", "warning", "已编译知识缺少直接来源支持")
                )
            for tag in item.tags:
                if not _CANONICAL_TAG_RE.fullmatch(tag):
                    issues.append(
                        self._issue(
                            item,
                            "noncanonical_tag",
                            "info",
                            f"标签“{tag}”不符合小写 kebab-case 规范",
                            {"tag": tag, "suggested": self._suggest_tag(tag)},
                        )
                    )

        collisions = self.connection.execute(
            """SELECT normalized_alias, COUNT(DISTINCT item_id) AS item_count,
                      GROUP_CONCAT(DISTINCT item_id) AS item_ids
               FROM knowledge_aliases GROUP BY normalized_alias
               HAVING COUNT(DISTINCT item_id) > 1"""
        ).fetchall()
        for row in collisions:
            item_ids = sorted(str(row["item_ids"] or "").split(","))
            issues.append(
                KnowledgeHealthIssue(
                    code="ambiguous_alias",
                    severity="warning",
                    item_id=None,
                    title=str(row["normalized_alias"]),
                    message="同一别名指向多个知识条目",
                    details={"item_ids": item_ids},
                )
            )
        severity_order = {"error": 0, "warning": 1, "info": 2}
        issues.sort(
            key=lambda issue: (
                severity_order.get(issue.severity, 3),
                issue.code,
                issue.item_id or "",
            )
        )
        counts = {
            "by_type": self._count_values(items, "item_type"),
            "by_status": self._count_values(items, "status"),
            "by_maturity": self._count_values(items, "maturity"),
            "issues_by_code": self._count_values(issues, "code"),
            "issues_by_severity": self._count_values(issues, "severity"),
        }
        return KnowledgeHealthReport(
            generated_at=now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            total_items=len(items),
            counts=counts,
            issues=tuple(issues),
        )

    def _item_filters(
        self,
        *,
        item_types: Sequence[str],
        statuses: Sequence[str],
        maturities: Sequence[str],
        tags: Sequence[str],
        high_value: bool | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, values, allowed, name in (
            ("item_type", item_types, KNOWLEDGE_ITEM_TYPES, "知识类型"),
            ("status", statuses, KNOWLEDGE_STATUSES, "状态"),
            ("maturity", maturities, KNOWLEDGE_MATURITIES, "摄取成熟度"),
        ):
            clean_values = self._validated_values(values, allowed, name)
            if clean_values:
                clauses.append(f"ki.{column} IN ({','.join('?' for _ in clean_values)})")
                parameters.extend(clean_values)
        for tag in dict.fromkeys(_normalize(value) for value in tags if _normalize(value)):
            clauses.append(
                """EXISTS (SELECT 1 FROM knowledge_item_tags kit
                           WHERE kit.item_id=ki.id AND kit.normalized_tag=?)"""
            )
            parameters.append(tag)
        if high_value is not None:
            clauses.append("ki.high_value=?")
            parameters.append(1 if high_value else 0)
        return (f"WHERE {' AND '.join(clauses)}" if clauses else "", parameters)

    @staticmethod
    def _validated_values(
        values: Sequence[str], allowed: frozenset[str], field_name: str
    ) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(_validate_choice(value, allowed, field_name) for value in values)
        )

    def _validate_links(
        self, item_type: str, document_id: str | None, artifact_id: str | None
    ) -> None:
        if document_id and item_type != "source":
            raise ValueError("只有 source 类型可以关联 document_id")
        if artifact_id and item_type != "output":
            raise ValueError("只有 output 类型可以关联 artifact_id")
        if document_id and self.connection.execute(
            "SELECT 1 FROM documents WHERE id=?", (document_id,)
        ).fetchone() is None:
            raise ValueError(f"文档不存在: {document_id}")
        if artifact_id and self.connection.execute(
            "SELECT 1 FROM artifacts WHERE id=?", (artifact_id,)
        ).fetchone() is None:
            raise ValueError(f"产物不存在: {artifact_id}")

    def _replace_aliases(
        self, item_id: str, aliases: Iterable[str], created_at: str
    ) -> None:
        self.connection.execute("DELETE FROM knowledge_aliases WHERE item_id=?", (item_id,))
        unique: dict[str, str] = {}
        for alias in aliases:
            clean = _clean_text(alias, required=True, field_name="别名")
            unique.setdefault(_normalize(clean), clean)
        self.connection.executemany(
            """INSERT INTO knowledge_aliases(item_id, alias, normalized_alias, created_at)
               VALUES (?, ?, ?, ?)""",
            ((item_id, alias, normalized, created_at) for normalized, alias in unique.items()),
        )

    def _replace_tags(self, item_id: str, tags: Iterable[str], created_at: str) -> None:
        self.connection.execute("DELETE FROM knowledge_item_tags WHERE item_id=?", (item_id,))
        unique: dict[str, str] = {}
        for tag in tags:
            clean = _clean_text(tag, required=True, field_name="标签")
            unique.setdefault(_normalize(clean), clean)
        self.connection.executemany(
            """INSERT INTO knowledge_item_tags(item_id, tag, normalized_tag, created_at)
               VALUES (?, ?, ?, ?)""",
            ((item_id, tag, normalized, created_at) for normalized, tag in unique.items()),
        )

    @staticmethod
    def _snapshot_values(
        raw_values: Any,
        *,
        value_key: str,
        fallback_created_at: str,
        field_name: str,
    ) -> list[tuple[str, str, str]]:
        if not isinstance(raw_values, (list, tuple)):
            raise ValueError(f"知识回收站中的{field_name}必须是列表")
        unique: dict[str, tuple[str, str]] = {}
        for raw in raw_values:
            if isinstance(raw, Mapping):
                value = _clean_text(
                    raw.get(value_key), required=True, field_name=field_name
                )
                created_at = _clean_text(raw.get("created_at")) or fallback_created_at
            else:
                value = _clean_text(raw, required=True, field_name=field_name)
                created_at = fallback_created_at
            normalized = _normalize(value)
            unique.setdefault(normalized, (value, created_at))
        return [
            (value, normalized, timestamp)
            for normalized, (value, timestamp) in unique.items()
        ]

    def _item_from_row(self, row: sqlite3.Row) -> KnowledgeItem:
        item_id = str(row["id"])
        return KnowledgeItem(
            item_id=item_id,
            item_type=str(row["item_type"]),
            status=str(row["status"]),
            maturity=str(row["maturity"]),
            title=str(row["title"]),
            summary=str(row["summary"] or ""),
            body=str(row["body"] or ""),
            document_id=str(row["document_id"]) if row["document_id"] is not None else None,
            artifact_id=str(row["artifact_id"]) if row["artifact_id"] is not None else None,
            high_value=bool(row["high_value"]),
            aliases=tuple(self.list_aliases(item_id)),
            tags=tuple(self.list_tags(item_id)),
            metadata=_load_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _relation_from_row(row: sqlite3.Row) -> KnowledgeRelation:
        return KnowledgeRelation(
            relation_id=str(row["id"]),
            source_item_id=str(row["source_item_id"]),
            target_item_id=str(row["target_item_id"]),
            relation_type=str(row["relation_type"]),
            summary=str(row["summary"] or ""),
            metadata=_load_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _require_item(self, item_id: str) -> KnowledgeItem:
        item = self.get_item(item_id)
        if item is None:
            raise ValueError(f"知识条目不存在: {item_id}")
        return item

    def _require_relation(self, relation_id: str) -> KnowledgeRelation:
        relation = self.get_relation(relation_id)
        if relation is None:
            raise ValueError(f"知识关系不存在: {relation_id}")
        return relation

    @staticmethod
    def _has_source_evidence(item: KnowledgeItem) -> bool:
        if item.document_id:
            return True
        evidence_fields = {
            "original_uri", "source_uri", "uri", "url", "local_path", "path", "archive_path"
        }
        return any(_clean_text(item.metadata.get(key)) for key in evidence_fields)

    @staticmethod
    def _older_than(value: str, cutoff: datetime) -> bool:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC) < cutoff

    @staticmethod
    def _suggest_tag(tag: str) -> str:
        clean = _normalize(tag)
        clean = re.sub(r"[\s_/]+", "-", clean)
        clean = re.sub(r"[^a-z0-9\u3400-\u9fff-]+", "", clean)
        return re.sub(r"-+", "-", clean).strip("-")

    @staticmethod
    def _issue(
        item: KnowledgeItem,
        code: str,
        severity: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> KnowledgeHealthIssue:
        return KnowledgeHealthIssue(
            code=code,
            severity=severity,
            item_id=item.item_id,
            title=item.title,
            message=message,
            details=details or {},
        )

    @staticmethod
    def _count_values(values: Iterable[Any], attribute: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            key = str(getattr(value, attribute))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


__all__ = [
    "KNOWLEDGE_ITEM_TYPES",
    "KNOWLEDGE_STATUSES",
    "KNOWLEDGE_MATURITIES",
    "KNOWLEDGE_RELATION_TYPES",
    "KnowledgeItem",
    "KnowledgeRelation",
    "KnowledgeRestoreResult",
    "RelatedKnowledgeItem",
    "KnowledgeHealthIssue",
    "KnowledgeHealthReport",
    "KnowledgeGovernanceRepository",
]
