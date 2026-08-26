from __future__ import annotations

import json
import re
from typing import Any

from ..models import SourceReference, utcnow_iso
from ..qa.models import (
    ConversationContext,
    ConversationMessage,
    KnowledgeAnswer,
    new_id,
)
from .database import KnowledgeDatabase


class ConversationRepository:
    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    def ensure_conversation(self, conversation_id: str | None = None, title: str | None = None) -> str:
        if conversation_id:
            row = self.database.connection.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row:
                return conversation_id
        conversation_id = conversation_id or new_id("conv")
        now = utcnow_iso()
        with self.database.connection:
            self.database.connection.execute(
                """INSERT INTO conversations(id, title, summary, summary_through_ordinal, created_at, updated_at)
                   VALUES (?, ?, '', 0, ?, ?)""",
                (conversation_id, title, now, now),
            )
        return conversation_id

    def _next_ordinal(self, conversation_id: str) -> int:
        row = self.database.connection.execute(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 AS value FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row["value"])

    def _insert_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            message_id=new_id("msg"),
            conversation_id=conversation_id,
            ordinal=self._next_ordinal(conversation_id),
            role=role,
            content=content,
            metadata=metadata or {},
            created_at=utcnow_iso(),
        )
        self.database.connection.execute(
            """INSERT INTO messages(id, conversation_id, ordinal, role, content, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                message.message_id,
                message.conversation_id,
                message.ordinal,
                message.role,
                message.content,
                json.dumps(message.metadata, ensure_ascii=False, sort_keys=True),
                message.created_at,
            ),
        )
        self.database.connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (message.created_at, conversation_id),
        )
        return message

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"unsupported conversation role: {role}")
        with self.database.connection:
            return self._insert_message(conversation_id, role, content, metadata)

    def context(self, conversation_id: str, recent_limit: int = 6) -> ConversationContext:
        conversation = self.database.connection.execute(
            "SELECT summary FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            raise ValueError(f"conversation does not exist: {conversation_id}")
        rows = self.database.connection.execute(
            """SELECT * FROM messages WHERE conversation_id = ?
               ORDER BY ordinal DESC LIMIT ?""",
            (conversation_id, recent_limit),
        ).fetchall()
        messages = [
            ConversationMessage(
                message_id=row["id"],
                conversation_id=row["conversation_id"],
                ordinal=row["ordinal"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in reversed(rows)
        ]
        return ConversationContext(conversation_id, conversation["summary"], messages)

    def refresh_summary(self, conversation_id: str, recent_limit: int = 6) -> str:
        conversation = self.database.connection.execute(
            "SELECT summary, summary_through_ordinal FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conversation is None:
            raise ValueError(f"conversation does not exist: {conversation_id}")
        maximum = self.database.connection.execute(
            "SELECT COALESCE(MAX(ordinal), 0) AS value FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()["value"]
        cutoff = max(0, int(maximum) - recent_limit)
        if cutoff <= conversation["summary_through_ordinal"]:
            return conversation["summary"]
        rows = self.database.connection.execute(
            """SELECT ordinal, role, content FROM messages
               WHERE conversation_id = ? AND ordinal > ? AND ordinal <= ? ORDER BY ordinal""",
            (conversation_id, conversation["summary_through_ordinal"], cutoff),
        ).fetchall()
        additions = []
        for row in rows:
            compact = re.sub(r"\s+", " ", row["content"]).strip()
            additions.append(f"{row['role']}: {compact[:360]}")
        summary = "\n".join(part for part in [conversation["summary"], *additions] if part).strip()
        if len(summary) > 4000:
            summary = summary[-4000:]
        with self.database.connection:
            self.database.connection.execute(
                """UPDATE conversations SET summary = ?, summary_through_ordinal = ?, updated_at = ?
                   WHERE id = ?""",
                (summary, cutoff, utcnow_iso(), conversation_id),
            )
        return summary

    def chunk_belongs_to_document(self, chunk_id: str, document_id: str) -> bool:
        row = self.database.connection.execute(
            "SELECT 1 FROM chunks WHERE id = ? AND document_id = ?", (chunk_id, document_id)
        ).fetchone()
        return row is not None

    def save_answer(self, answer: KnowledgeAnswer, question_message_id: str) -> str:
        with self.database.connection:
            assistant_message = self._insert_message(
                answer.conversation_id,
                "assistant",
                answer.markdown,
                {"answer_id": answer.answer_id},
            )
            self.database.connection.execute(
                """INSERT INTO answers(
                       id, conversation_id, question_message_id, answer_message_id, markdown,
                       model, provider, token_usage_json, retrieval_info_json, confidence, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    answer.answer_id,
                    answer.conversation_id,
                    question_message_id,
                    assistant_message.message_id,
                    answer.markdown,
                    answer.model,
                    answer.provider,
                    json.dumps(answer.token_usage.to_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(answer.retrieval_info, ensure_ascii=False, sort_keys=True),
                    answer.confidence,
                    answer.created_at,
                ),
            )
            for evidence in answer.evidence:
                self.database.connection.execute(
                    """INSERT INTO answer_evidence(
                           answer_id, evidence_id, source_kind, document_id, chunk_id, title,
                           content, score, source_reference_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        answer.answer_id,
                        evidence.evidence_id,
                        evidence.source_kind,
                        evidence.document_id,
                        evidence.chunk_id,
                        evidence.title,
                        evidence.content,
                        evidence.score,
                        json.dumps(evidence.source.to_dict(), ensure_ascii=False, sort_keys=True),
                    ),
                )
            for citation in answer.citations:
                self.database.connection.execute(
                    """INSERT INTO citations(
                           answer_id, citation_id, evidence_id, source_kind, document_id, chunk_id,
                           media_type, title, original_uri, local_path, obsidian_path, page_number,
                           slide_number, timestamp_start, timestamp_end, section
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        answer.answer_id,
                        citation.citation_id,
                        citation.evidence_id,
                        citation.source_kind,
                        citation.document_id,
                        citation.chunk_id,
                        citation.media_type,
                        citation.title,
                        citation.original_uri,
                        citation.local_path,
                        citation.obsidian_path,
                        citation.page_number,
                        citation.slide_number,
                        citation.timestamp_start,
                        citation.timestamp_end,
                        citation.section,
                    ),
                )
        return assistant_message.message_id

    def conversation_record(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.database.connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            raise ValueError(f"conversation does not exist: {conversation_id}")
        messages = self.database.connection.execute(
            "SELECT id, ordinal, role, content, metadata_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY ordinal",
            (conversation_id,),
        ).fetchall()
        answer_ids = [
            row["id"]
            for row in self.database.connection.execute(
                "SELECT id FROM answers WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        ]
        return {
            "conversation_id": conversation["id"],
            "title": conversation["title"],
            "summary": conversation["summary"],
            "summary_through_ordinal": conversation["summary_through_ordinal"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "messages": [
                {
                    "message_id": row["id"],
                    "ordinal": row["ordinal"],
                    "role": row["role"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                }
                for row in messages
            ],
            "answers": [self.answer_record(answer_id) for answer_id in answer_ids],
        }

    def answer_record(self, answer_id: str) -> dict[str, Any]:
        answer = self.database.connection.execute(
            """SELECT a.*, q.content AS question
               FROM answers a JOIN messages q ON q.id=a.question_message_id WHERE a.id = ?""",
            (answer_id,),
        ).fetchone()
        if answer is None:
            raise ValueError(f"answer does not exist: {answer_id}")
        evidence_rows = self.database.connection.execute(
            """SELECT * FROM answer_evidence WHERE answer_id = ?
               ORDER BY CAST(SUBSTR(evidence_id, 2) AS INTEGER)""",
            (answer_id,),
        ).fetchall()
        citation_rows = self.database.connection.execute(
            """SELECT * FROM citations WHERE answer_id = ?
               ORDER BY CAST(SUBSTR(citation_id, 2) AS INTEGER)""",
            (answer_id,),
        ).fetchall()
        return {
            "answer_id": answer["id"],
            "conversation_id": answer["conversation_id"],
            "question_message_id": answer["question_message_id"],
            "answer_message_id": answer["answer_message_id"],
            "question": answer["question"],
            "markdown": answer["markdown"],
            "model": answer["model"],
            "provider": answer["provider"],
            "token_usage": json.loads(answer["token_usage_json"]),
            "retrieval_info": json.loads(answer["retrieval_info_json"]),
            "confidence": answer["confidence"],
            "created_at": answer["created_at"],
            "evidence": [
                {
                    "evidence_id": row["evidence_id"],
                    "source_kind": row["source_kind"],
                    "document_id": row["document_id"],
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "content": row["content"],
                    "score": row["score"],
                    "source": json.loads(row["source_reference_json"]),
                }
                for row in evidence_rows
            ],
            "citations": [
                {
                    "citation_id": row["citation_id"],
                    "evidence_id": row["evidence_id"],
                    "source_kind": row["source_kind"],
                    "document_id": row["document_id"],
                    "chunk_id": row["chunk_id"],
                    "media_type": row["media_type"],
                    "title": row["title"],
                    "original_uri": row["original_uri"],
                    "local_path": row["local_path"],
                    "obsidian_path": row["obsidian_path"],
                    "page_number": row["page_number"],
                    "slide_number": row["slide_number"],
                    "timestamp_start": row["timestamp_start"],
                    "timestamp_end": row["timestamp_end"],
                    "section": row["section"],
                }
                for row in citation_rows
            ],
        }
