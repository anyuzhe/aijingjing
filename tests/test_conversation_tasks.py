from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from media_knowledge.desktop.controller import DesktopController
from media_knowledge.ingestion import IngestionResult, IngestionSummary, ProgressEvent
from media_knowledge.models import SourceReference, utcnow_iso
from media_knowledge.qa.models import Citation, Evidence, KnowledgeAnswer, TokenUsage
from media_knowledge.storage import (
    ConversationRepository,
    IngestionJobRepository,
    KnowledgeDatabase,
)


class ConversationTaskPersistenceTests(unittest.TestCase):
    def test_explicit_migrations_upgrade_an_existing_version_eight_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (8, 'legacy')"
            )
            connection.commit()
            connection.close()

            with KnowledgeDatabase(path) as database:
                versions = {
                    int(row["version"])
                    for row in database.connection.execute(
                        "SELECT version FROM schema_migrations"
                    ).fetchall()
                }
                tables = {
                    str(row["name"])
                    for row in database.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            self.assertTrue({8, 9, 10}.issubset(versions))
            self.assertIn("answer_feedback", tables)
            self.assertIn("ingestion_jobs", tables)
            self.assertIn("ingestion_job_items", tables)

    def test_conversation_search_pagination_rename_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with KnowledgeDatabase(Path(temporary) / "knowledge.db") as database:
                repository = ConversationRepository(database)
                first = repository.ensure_conversation(title="FDE 行业资料")
                repository.add_message(first, "user", "高质量 Skills 有哪些来源？", {"scope": "FDE"})
                second = repository.ensure_conversation(title="冬天计划")
                repository.add_message(second, "user", "准备冬季旅行")

                result = repository.search_conversations("Skills", limit=20, offset=-10)
                self.assertEqual([item["conversation_id"] for item in result], [first])
                self.assertEqual(result[0]["message_count"], 1)
                self.assertIn("Skills", result[0]["last_message"])
                self.assertEqual(repository.search_conversations("%' OR 1=1 --"), [])
                self.assertEqual(len(repository.list_conversations(limit=10_000)), 2)
                with self.assertRaises(ValueError):
                    repository.list_conversations(limit="bad")  # type: ignore[arg-type]

                self.assertTrue(repository.rename_conversation(first, "FDE 知识问答"))
                self.assertEqual(repository.conversation_record(first)["title"], "FDE 知识问答")
                self.assertTrue(repository.delete_conversation(second))
                self.assertFalse(repository.delete_conversation(second))

    def test_controller_feedback_is_updatable_and_export_contains_citations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            controller = DesktopController(data, migrate_legacy=False)
            with KnowledgeDatabase(controller.paths.database) as database:
                repository = ConversationRepository(database)
                conversation_id = repository.ensure_conversation(title="可追溯回答")
                question = repository.add_message(
                    conversation_id,
                    "user",
                    "结论是什么？",
                    {"image_attachments": [{"filename": "图.png", "local_path": "/tmp/图.png"}]},
                )
                source = SourceReference(
                    source_id="source-1",
                    media_type="pdf",
                    title="测试资料",
                    original_uri="https://example.test/source.pdf",
                    page_number=6,
                )
                evidence = Evidence("S1", "证据内容", "测试资料", 0.93, source)
                answer = KnowledgeAnswer(
                    answer_id="answer-feedback",
                    conversation_id=conversation_id,
                    markdown="结论来自测试资料。[S1]",
                    citations=[Citation.from_evidence(evidence)],
                    evidence=[evidence],
                    model="test-model",
                    provider="test-provider",
                    token_usage=TokenUsage(),
                    retrieval_info={"strategy": "focused"},
                    confidence=0.9,
                )
                repository.save_answer(answer, question.message_id)

            first = controller.save_answer_feedback(answer.answer_id, "up", "引用清楚")
            second = controller.save_answer_feedback(answer.answer_id, "down", "需要更详细")
            self.assertEqual(first["created_at"], second["created_at"])
            self.assertEqual(second["rating"], "down")
            self.assertEqual(second["reason"], "需要更详细")
            record = controller.conversation_record(conversation_id)
            self.assertEqual(record["messages"][0]["metadata"]["image_attachments"][0]["filename"], "图.png")
            self.assertEqual(record["answers"][0]["citations"][0]["page_number"], 6)
            self.assertEqual(record["answers"][0]["feedback"]["rating"], "down")
            self.assertEqual(
                controller.conversations("可追溯", limit=5)[0]["conversation_id"],
                conversation_id,
            )

            target = controller.export_conversation(
                conversation_id, Path(temporary) / "exports" / "conversation.md"
            )
            markdown = target.read_text(encoding="utf-8")
            self.assertIn("# 可追溯回答", markdown)
            self.assertIn("[S1] 测试资料（P6）", markdown)
            self.assertIn("需要更详细", markdown)
            self.assertTrue(controller.rename_conversation(conversation_id, "已重命名"))
            self.assertTrue(controller.delete_conversation(conversation_id))

    def test_stopped_partial_answer_remains_in_conversation_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(Path(temporary) / "data", migrate_legacy=False)
            with KnowledgeDatabase(controller.paths.database) as database:
                conversation_id = ConversationRepository(database).ensure_conversation(
                    title="停止生成测试"
                )
                ConversationRepository(database).add_message(
                    conversation_id, "user", "请生成一个很长的回答"
                )
            message_id = controller.save_partial_answer(
                conversation_id, "已经生成的部分内容。\n\n*回答已由用户停止。*"
            )
            record = controller.conversation_record(conversation_id)
            self.assertEqual(record["messages"][-1]["message_id"], message_id)
            self.assertTrue(record["messages"][-1]["metadata"]["partial"])
            self.assertIn("已经生成", record["messages"][-1]["content"])

    def test_ingestion_jobs_survive_restart_and_controller_records_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            controller = DesktopController(data, migrate_legacy=False)
            queued = controller.create_ingestion_job(["first.md", "second.md"])
            job_id = str(queued["id"])
            with KnowledgeDatabase(controller.paths.database) as database:
                jobs = IngestionJobRepository(database)
                jobs.begin_job(job_id)
                jobs.record_progress(job_id, "first.md", "extracting", 30, "正在解析")

            restarted = DesktopController(data, migrate_legacy=False)
            self.assertEqual(restarted.recovered_ingestion_jobs, 1)
            recovered = restarted.ingestion_job(job_id)
            self.assertEqual(recovered["status"], "queued")
            self.assertEqual(recovered["items"][0]["status"], "queued")

            def fake_ingest(service, values, *, progress=None, cancellation=None):
                summary = IngestionSummary()
                for source in values:
                    if progress:
                        progress(ProgressEvent(source, "extracting", 30, "正在解析"))
                        progress(ProgressEvent(source, "complete", 100, "完成"))
                    summary.results.append(
                        IngestionResult(
                            item=source,
                            title=Path(source).stem,
                            media_type="markdown",
                            status="created",
                            document_id=f"document-{Path(source).stem}",
                        )
                    )
                summary.completed_at = utcnow_iso()
                return summary

            with patch(
                "media_knowledge.desktop.controller.IngestionService.ingest",
                autospec=True,
                side_effect=fake_ingest,
            ):
                summary = restarted.resume_ingestion_job(job_id)

            self.assertEqual(summary.job_id, job_id)
            completed = restarted.ingestion_job(job_id)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["completed_items"], 2)
            self.assertEqual(completed["succeeded_items"], 2)
            self.assertTrue(all(item["result"].get("document_id") for item in completed["items"]))
            self.assertEqual(restarted.ingestion_jobs(limit=5)[0]["id"], job_id)

    def test_ingestion_jobs_preserve_failed_and_cancelled_terminal_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with KnowledgeDatabase(Path(temporary) / "knowledge.db") as database:
                jobs = IngestionJobRepository(database)
                failed_id = str(jobs.create_job(["broken.pdf"])["id"])
                jobs.begin_job(failed_id)
                jobs.record_result(
                    failed_id,
                    "broken.pdf",
                    {"item": "broken.pdf", "status": "failed", "error": "无法解析"},
                )
                self.assertEqual(jobs.finalize_job(failed_id)["status"], "failed")
                self.assertEqual(jobs.job_record(failed_id)["items"][0]["error"], "无法解析")

                cancelled_id = str(jobs.create_job(["large.mp4"])["id"])
                jobs.begin_job(cancelled_id)
                self.assertTrue(jobs.cancel_job(cancelled_id))
                cancelled = jobs.job_record(cancelled_id)
                self.assertEqual(cancelled["status"], "cancelled")
                self.assertEqual(cancelled["items"][0]["status"], "cancelled")
                self.assertIsNotNone(cancelled["completed_at"])

                self.assertEqual(jobs.reset_failed_items(failed_id), 1)
                self.assertEqual(jobs.job_record(failed_id)["status"], "queued")

    def test_controller_catastrophic_ingestion_failure_remains_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(Path(temporary) / "data", migrate_legacy=False)
            with patch(
                "media_knowledge.desktop.controller.IngestionService.ingest",
                side_effect=RuntimeError("解析服务异常退出"),
            ):
                with self.assertRaisesRegex(RuntimeError, "解析服务异常退出"):
                    controller.ingest(["first.pdf", "second.pdf"])
            job = controller.ingestion_jobs(limit=1)[0]
            self.assertEqual(job["status"], "failed")
            record = controller.ingestion_job(str(job["id"]))
            self.assertEqual(
                [item["status"] for item in record["items"]],
                ["failed", "failed"],
            )
            self.assertTrue(all(item["error"] == "解析服务异常退出" for item in record["items"]))


if __name__ == "__main__":
    unittest.main()
