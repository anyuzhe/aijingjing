from __future__ import annotations

import json
import re
from typing import Any

from ..models import utcnow_iso
from ..qa.models import (
    ConversationContext,
    ConversationMessage,
    KnowledgeAnswer,
    new_id,
)
from .database import KnowledgeDatabase


class ConversationRepository:
    MAX_PAGE_SIZE = 500

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    @classmethod
    def _page(cls, limit: int, offset: int) -> tuple[int, int]:
        try:
            safe_limit = int(limit)
            safe_offset = int(offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit 和 offset 必须是整数") from exc
        return max(1, min(cls.MAX_PAGE_SIZE, safe_limit)), max(0, safe_offset)

    def list_conversations(
        self,
        query: str = "",
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return a bounded, newest-first page suitable for conversation history UIs."""

        safe_limit, safe_offset = self._page(limit, offset)
        term = re.sub(r"\s+", " ", str(query or "")).strip()[:200]
        where = ""
        parameters: list[Any] = []
        if term:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            where = """WHERE (
                COALESCE(c.title, '') LIKE ? ESCAPE '\\' COLLATE NOCASE
                OR c.summary LIKE ? ESCAPE '\\' COLLATE NOCASE
                OR EXISTS (
                    SELECT 1 FROM messages search_message
                    WHERE search_message.conversation_id=c.id
                      AND search_message.content LIKE ? ESCAPE '\\' COLLATE NOCASE
                )
            )"""
            parameters.extend((pattern, pattern, pattern))
        rows = self.database.connection.execute(
            f"""SELECT c.*,
                       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id)
                           AS message_count,
                       (SELECT COUNT(*) FROM answers a WHERE a.conversation_id=c.id)
                           AS answer_count,
                       (SELECT m.content FROM messages m WHERE m.conversation_id=c.id
                        ORDER BY m.ordinal DESC LIMIT 1) AS last_message
                FROM conversations c
                {where}
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ? OFFSET ?""",
            (*parameters, safe_limit, safe_offset),
        ).fetchall()
        return [
            {
                "conversation_id": str(row["id"]),
                "title": str(row["title"] or "").strip() or "新对话",
                "summary": str(row["summary"] or ""),
                "message_count": int(row["message_count"] or 0),
                "answer_count": int(row["answer_count"] or 0),
                "last_message": str(row["last_message"] or ""),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in rows
        ]

    def search_conversations(
        self, query: str, *, limit: int = 200, offset: int = 0
    ) -> list[dict[str, Any]]:
        return self.list_conversations(query, limit=limit, offset=offset)

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
        if not clean_title:
            raise ValueError("对话标题不能为空")
        if len(clean_title) > 200:
            raise ValueError("对话标题不能超过 200 个字符")
        with self.database.connection:
            cursor = self.database.connection.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (clean_title, utcnow_iso(), conversation_id),
            )
        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.database.connection:
            cursor = self.database.connection.execute(
                "DELETE FROM conversations WHERE id=?", (conversation_id,)
            )
        return cursor.rowcount > 0

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

    def conversation_markdown(self, conversation_id: str) -> str:
        """Build a portable Markdown transcript with answer evidence and feedback."""

        record = self.conversation_record(conversation_id)
        answers_by_message = {
            str(answer["answer_message_id"]): answer for answer in record["answers"]
        }
        title = str(record.get("title") or "新对话").strip() or "新对话"
        lines = [
            "---",
            f"conversation_id: {json.dumps(conversation_id, ensure_ascii=False)}",
            f"created_at: {json.dumps(record['created_at'], ensure_ascii=False)}",
            f"updated_at: {json.dumps(record['updated_at'], ensure_ascii=False)}",
            'tags: ["AI静静/对话"]',
            "---",
            "",
            f"# {title}",
            "",
        ]
        role_titles = {"user": "用户", "assistant": "AI静静", "system": "系统"}
        for message in record["messages"]:
            role = role_titles.get(str(message["role"]), str(message["role"]))
            lines.extend(
                [
                    f"## {role} · {message['created_at']}",
                    "",
                    str(message["content"]).strip(),
                    "",
                ]
            )
            metadata = message.get("metadata") or {}
            attachments = metadata.get("image_attachments", []) if isinstance(metadata, dict) else []
            if isinstance(attachments, list) and attachments:
                lines.extend(["### 提问图片", ""])
                for attachment in attachments:
                    if not isinstance(attachment, dict):
                        continue
                    path = str(attachment.get("local_path") or "").strip()
                    label = str(attachment.get("filename") or "图片").strip() or "图片"
                    if path:
                        lines.append(f"- [{label}]({path})")
                lines.append("")
            answer = answers_by_message.get(str(message["message_id"]))
            if not answer:
                continue
            citations = answer.get("citations") or []
            if citations:
                lines.extend(["### 证据来源", ""])
                for citation in citations:
                    location = []
                    if citation.get("page_number") is not None:
                        location.append(f"P{citation['page_number']}")
                    if citation.get("slide_number") is not None:
                        location.append(f"S{citation['slide_number']}")
                    if citation.get("timestamp_start") is not None:
                        location.append(f"{citation['timestamp_start']:g}s")
                    target = citation.get("original_uri") or citation.get("local_path") or "本地知识库"
                    suffix = f"（{' / '.join(location)}）" if location else ""
                    lines.append(
                        f"- [{citation['citation_id']}] {citation['title']}{suffix} — {target}"
                    )
                lines.append("")
            feedback = answer.get("feedback")
            if isinstance(feedback, dict):
                label = "有帮助" if feedback.get("rating") == "up" else "需要改进"
                reason = str(feedback.get("reason") or "").strip()
                lines.extend(["### 回答反馈", "", f"- 评价：{label}"])
                if reason:
                    lines.append(f"- 原因：{reason}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def save_answer_feedback(
        self, answer_id: str, rating: str, reason: str = ""
    ) -> dict[str, Any]:
        normalized = str(rating or "").strip().casefold()
        if normalized not in {"up", "down"}:
            raise ValueError("反馈只能是 up 或 down")
        clean_reason = str(reason or "").strip()
        if len(clean_reason) > 2000:
            raise ValueError("反馈原因不能超过 2000 个字符")
        if self.database.connection.execute(
            "SELECT 1 FROM answers WHERE id=?", (answer_id,)
        ).fetchone() is None:
            raise ValueError(f"answer does not exist: {answer_id}")
        now = utcnow_iso()
        with self.database.connection:
            self.database.connection.execute(
                """INSERT INTO answer_feedback(answer_id, rating, reason, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(answer_id) DO UPDATE SET
                       rating=excluded.rating, reason=excluded.reason,
                       updated_at=excluded.updated_at""",
                (answer_id, normalized, clean_reason, now, now),
            )
        row = self.database.connection.execute(
            "SELECT * FROM answer_feedback WHERE answer_id=?", (answer_id,)
        ).fetchone()
        return dict(row)

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
        feedback = self.database.connection.execute(
            "SELECT rating, reason, created_at, updated_at FROM answer_feedback WHERE answer_id=?",
            (answer_id,),
        ).fetchone()
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
            "feedback": dict(feedback) if feedback is not None else None,
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
