from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..models import SourceReference, utcnow_iso


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


@dataclass(slots=True)
class QuestionAnalysis:
    original_question: str
    normalized_question: str
    is_follow_up: bool
    task_type: str
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class Evidence:
    evidence_id: str
    content: str
    title: str
    score: float
    source: SourceReference
    source_kind: str = "knowledge"

    @property
    def document_id(self) -> str | None:
        return self.source.document_id

    @property
    def chunk_id(self) -> str | None:
        return self.source.chunk_id

    def locator(self) -> str:
        parts = []
        if self.source.page_number is not None:
            parts.append(f"page {self.source.page_number}")
        if self.source.slide_number is not None:
            parts.append(f"slide {self.source.slide_number}")
        if self.source.timestamp_start is not None:
            end = self.source.timestamp_end
            parts.append(
                f"{self.source.timestamp_start:g}s"
                + (f"-{end:g}s" if end is not None else "")
            )
        if self.source.section:
            parts.append(self.source.section)
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "title": self.title,
            "score": self.score,
            "source_kind": self.source_kind,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source.to_dict(),
        }


@dataclass(slots=True)
class Citation:
    citation_id: str
    evidence_id: str
    source_kind: str
    document_id: str | None
    chunk_id: str | None
    media_type: str
    title: str
    original_uri: str | None = None
    local_path: str | None = None
    obsidian_path: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    section: str | None = None

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> "Citation":
        source = evidence.source
        return cls(
            citation_id=evidence.evidence_id,
            evidence_id=evidence.evidence_id,
            source_kind=evidence.source_kind,
            document_id=source.document_id,
            chunk_id=source.chunk_id,
            media_type=source.media_type,
            title=source.title,
            original_uri=source.original_uri,
            local_path=source.local_path,
            obsidian_path=source.obsidian_path,
            page_number=source.page_number,
            slide_number=source.slide_number,
            timestamp_start=source.timestamp_start,
            timestamp_end=source.timestamp_end,
            section=source.section,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KnowledgeAnswer:
    answer_id: str
    conversation_id: str
    markdown: str
    citations: list[Citation]
    evidence: list[Evidence]
    model: str
    provider: str
    token_usage: TokenUsage
    retrieval_info: dict[str, Any]
    confidence: float
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "conversation_id": self.conversation_id,
            "markdown": self.markdown,
            "citations": [citation.to_dict() for citation in self.citations],
            "evidence": [item.to_dict() for item in self.evidence],
            "model": self.model,
            "provider": self.provider,
            "token_usage": self.token_usage.to_dict(),
            "retrieval_info": self.retrieval_info,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ConversationMessage:
    message_id: str
    conversation_id: str
    ordinal: int
    role: str
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationContext:
    conversation_id: str
    summary: str
    recent_messages: list[ConversationMessage]

    def as_prompt(self, max_characters: int = 4000) -> str:
        sections = []
        if self.summary.strip():
            sections.append("Conversation summary:\n" + self.summary.strip())
        if self.recent_messages:
            lines = [f"{message.role}: {message.content}" for message in self.recent_messages]
            sections.append("Recent context:\n" + "\n".join(lines))
        return "\n\n".join(sections)[-max_characters:]

    def subject_candidates(self) -> list[str]:
        text = "\n".join([self.summary, *(message.content for message in self.recent_messages)])
        named = re.findall(r"\b(?:[A-Z][A-Z0-9]*)(?:-[A-Z0-9]+)+\b|\b[A-Z]{2,}[0-9]*\b", text)
        return list(dict.fromkeys(named))[-6:]

    def latest_image_attachments(self, limit: int = 4) -> list["ImageAttachment"]:
        """Return the newest usable user images so follow-ups can refer to “this image”."""

        for message in reversed(self.recent_messages):
            if message.role != "user":
                continue
            raw_values = message.metadata.get("image_attachments", [])
            if not isinstance(raw_values, list):
                continue
            attachments: list[ImageAttachment] = []
            for raw in raw_values:
                if not isinstance(raw, dict):
                    continue
                try:
                    attachment = ImageAttachment.from_dict(raw)
                except (TypeError, ValueError):
                    continue
                if Path(attachment.local_path).is_file():
                    attachments.append(attachment)
            if attachments:
                return attachments[:limit]
        return []


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    """A normalized local image supplied with a chat message."""

    local_path: str
    filename: str
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageAttachment":
        return cls(
            local_path=str(value["local_path"]),
            filename=str(value.get("filename") or Path(str(value["local_path"])).name),
            mime_type=str(value.get("mime_type") or "image/png"),
            width=int(value["width"]) if value.get("width") is not None else None,
            height=int(value["height"]) if value.get("height") is not None else None,
        )


@dataclass(slots=True)
class AnswerRequest:
    question: str
    system_prompt: str
    user_prompt: str
    evidence: list[Evidence]
    response_language: str | None = None
    image_attachments: list[ImageAttachment] = field(default_factory=list)


@dataclass(slots=True)
class AnswerResponse:
    markdown: str
    model: str
    provider: str
    token_usage: TokenUsage
