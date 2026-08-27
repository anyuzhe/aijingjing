from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def estimate_tokens(value: str) -> int:
    """Cheap, deterministic estimate suitable for local chunk sizing."""
    units = re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", value)
    punctuation_allowance = max(0, len(value) - sum(len(unit) for unit in units)) // 8
    return max(1, len(units) + punctuation_allowance) if value.strip() else 0


@dataclass(slots=True)
class SourceReference:
    source_id: str
    media_type: str
    title: str
    document_id: str | None = None
    chunk_id: str | None = None
    original_uri: str | None = None
    local_path: str | None = None
    obsidian_path: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    section: str | None = None
    image_path: str | None = None
    text_start: int | None = None
    text_end: int | None = None
    checksum: str | None = None

    def with_chunk(self, document_id: str, chunk_id: str) -> "SourceReference":
        return replace(self, document_id=document_id, chunk_id=chunk_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceReference":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: data.get(key) for key in allowed})


@dataclass(slots=True)
class ContentSegment:
    id: str
    sequence: float
    modality: str
    text: str = ""
    description: str = ""
    location: dict[str, Any] = field(default_factory=dict)
    heading_path: list[str] = field(default_factory=list)
    asset: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_text(self) -> str:
        values = [self.text.strip(), self.description.strip()]
        if self.metadata.get("speaker"):
            values.append(f"Speaker: {self.metadata['speaker']}")
        return "\n".join(value for value in values if value)


@dataclass(slots=True)
class KnowledgeDocument:
    source_id: str
    title: str
    media_type: str
    segments: list[ContentSegment]
    source: SourceReference
    document_id: str | None = None
    collections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    def content_hash(self) -> str:
        payload = [
            {
                "sequence": segment.sequence,
                "modality": segment.modality,
                "text": segment.text,
                "description": segment.description,
                "location": segment.location,
            }
            for segment in sorted(self.segments, key=lambda item: (item.sequence, item.id))
        ]
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


@dataclass(slots=True)
class KnowledgeChunk:
    id: str
    document_id: str
    chunk_key: str
    ordinal: int
    content: str
    heading_path: list[str]
    source_reference: SourceReference
    token_count: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_status: str = "pending"
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)


@dataclass(slots=True)
class SearchFilters:
    collections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None


@dataclass(slots=True)
class SearchCandidate:
    chunk_id: str
    content: str
    title: str
    source_reference: SourceReference
    fused_score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    lexical_overlap: float | None = None


@dataclass(slots=True)
class SearchResult:
    score: float
    content: str
    title: str
    source: SourceReference
    page: int | None
    slide: int | None
    timestamp_start: float | None
    timestamp_end: float | None
    document_id: str
    chunk_id: str
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "content": self.content,
            "title": self.title,
            "source": self.source.to_dict(),
            "page": self.page,
            "slide": self.slide,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "debug": self.debug,
        }


@dataclass(slots=True)
class IndexReport:
    document_id: str
    source_id: str
    status: str
    created_chunks: int = 0
    updated_chunks: int = 0
    unchanged_chunks: int = 0
    deleted_chunks: int = 0
    embedded_chunks: int = 0
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
