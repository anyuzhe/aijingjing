from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.transcripts import (
    DeepCorrectionRepository,
    DeepCorrectionMarkdownExporter,
    TranscriptQuality,
    TranscriptRepository,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptSpeaker,
    TranscriptV2,
)


class DeepCorrectionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = KnowledgeDatabase(self.root / "knowledge.db")
        self.transcripts = TranscriptRepository(self.database)
        self.transcripts.save_transcript(_transcript())
        self.repository = DeepCorrectionRepository(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_schema_13_adds_deep_correction_fact_and_audit_tables(self) -> None:
        versions = {
            int(row["version"])
            for row in self.database.connection.execute(
                "SELECT version FROM schema_migrations"
            )
        }
        self.assertIn(13, versions)
        for table in (
            "correction_runs",
            "correction_paragraphs",
            "correction_changes",
            "correction_change_events",
            "correction_evidence",
        ):
            self.assertEqual(self.database.status()[table], 0)

    def test_run_failure_retry_and_cancellation_are_explicit(self) -> None:
        run = self.repository.create_run(
            "asr-run-001",
            provider="deepseek",
            model="deepseek-chat",
            config={"profile": "deep-proofread", "temperature": 0},
            max_attempts=2,
            run_id="correction-001",
        )
        self.assertEqual(run.status, "queued")
        self.assertEqual(self.repository.start_run(run.id).attempt_count, 1)
        failed = self.repository.fail_run(run.id, "模型返回无效 JSON")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.last_error, "模型返回无效 JSON")
        self.assertEqual(self.repository.retry_run(run.id).status, "queued")
        self.assertEqual(self.repository.start_run(run.id).attempt_count, 2)
        requested = self.repository.request_cancel(run.id)
        self.assertTrue(requested.cancel_requested)
        cancelled = self.repository.mark_cancelled(run.id, reason="用户取消")
        self.assertEqual(cancelled.status, "cancelled")
        with self.assertRaises(ValueError):
            self.repository.retry_run(run.id)

    def test_paragraph_batch_rolls_back_and_existing_facts_cannot_be_overwritten(self) -> None:
        run = self._create_running()
        with self.assertRaises(ValueError):
            self.repository.save_paragraphs(
                run.id,
                [
                    _paragraph("p1", 0, 0, 1000, ["seg_0001"]),
                    _paragraph("p2", 1, 2000, 1500, ["seg_0002"]),
                ],
            )
        self.assertEqual(self.repository.list_paragraphs(run.id), [])

        saved = self.repository.save_paragraphs(
            run.id,
            [
                _paragraph("p1", 0, 0, 1500, ["seg_0001"]),
                _paragraph("p2", 1, 1500, 3000, ["seg_0002"]),
            ],
        )
        self.assertEqual(len(saved), 2)
        with self.assertRaises(ValueError):
            self.repository.save_paragraphs(
                run.id,
                [_paragraph("p1", 0, 0, 1500, ["seg_0001"], corrected="被覆盖")],
            )
        self.assertEqual(self.repository.list_paragraphs(run.id)[0].corrected_text, "围压需要提高。")

    def test_sql_failure_rolls_back_the_whole_paragraph_batch(self) -> None:
        first = self._create_running()
        self.repository.save_paragraphs(
            first.id, [_paragraph("shared-id", 0, 0, 1500, ["seg_0001"])]
        )
        second = self.repository.create_run(
            "asr-run-001", provider="deepseek", model="deepseek-chat",
            run_id="correction-002",
        )
        second = self.repository.start_run(second.id)
        with self.assertRaises(ValueError):
            self.repository.save_paragraphs(
                second.id,
                [
                    _paragraph("new-id", 0, 0, 1500, ["seg_0001"]),
                    _paragraph("shared-id", 1, 1500, 3000, ["seg_0002"]),
                ],
            )
        self.assertEqual(self.repository.list_paragraphs(second.id), [])

    def test_change_review_and_evidence_are_audited_without_touching_raw(self) -> None:
        run = self._create_running()
        self.repository.save_paragraphs(
            run.id, [_paragraph("p1", 0, 0, 1500, ["seg_0001"])]
        )
        change = self.repository.propose_change(
            run.id,
            paragraph_id="p1",
            change_type="terminology",
            before_text="微压",
            after_text="围压",
            reason="命中岩体力学词库，等待人工核验",
            confidence=0.86,
            source_segment_ids=["seg_0001"],
            change_id="change-001",
        )
        self.assertEqual(change.status, "proposed")
        accepted = self.repository.review_change(
            change.id, decision="accepted", actor="reviewer", reason="已回听原录音"
        )
        self.assertEqual(accepted.status, "accepted")
        with self.assertRaises(ValueError):
            self.repository.review_change(change.id, decision="rejected")
        events = self.repository.list_change_events(change.id)
        self.assertEqual([event.to_status for event in events], ["proposed", "accepted"])

        safe = self.repository.add_evidence(
            run.id,
            change_id=change.id,
            evidence_type="external",
            title="术语标准",
            url="https://example.test/spec?q=rock",
            summary="定义了围压。",
        )
        unsafe = self.repository.add_evidence(
            run.id,
            paragraph_id="p1",
            evidence_type="external",
            title="不安全链接",
            url="javascript:alert(1)",
        )
        self.assertEqual(len(self.repository.list_evidence(run.id)), 2)
        self.assertEqual(safe.url, "https://example.test/spec?q=rock")
        self.assertEqual(unsafe.url, "javascript:alert(1)")
        segment = self.transcripts.get_segment("seg_0001")
        assert segment is not None
        self.assertEqual(segment.raw_text, "微压需要提高")

    def test_accept_change_and_apply_is_atomic_and_keeps_raw_immutable(self) -> None:
        run = self._create_running()
        self.repository.save_paragraphs(
            run.id, [_paragraph("p1", 0, 0, 1500, ["seg_0001"])]
        )
        change = self.repository.propose_change(
            run.id,
            paragraph_id="p1",
            change_type="terminology",
            before_text="围压需要提高。",
            after_text="围压需要进一步提高。",
            source_segment_ids=["seg_0001"],
            change_id="atomic-change",
        )

        from unittest.mock import patch

        with patch.object(
            self.repository, "_insert_change_event", side_effect=RuntimeError("audit failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "audit failed"):
                self.repository.accept_change_and_apply(change.id, reason="人工回听确认")

        rolled_back = self.repository.get_change(change.id)
        assert rolled_back is not None
        self.assertEqual(rolled_back.status, "proposed")
        segment = self.transcripts.get_segment("seg_0001")
        assert segment is not None
        self.assertEqual(segment.corrected_text, "围压需要提高。")
        self.assertEqual(segment.raw_text, "微压需要提高")
        self.assertEqual(
            [item.to_status for item in self.repository.list_change_events(change.id)],
            ["proposed"],
        )

        accepted = self.repository.accept_change_and_apply(
            change.id,
            actor="reviewer",
            reason="人工回听确认",
            metadata={"review_session": "session-1", "change_id": "cannot-override"},
        )
        self.assertEqual(accepted.status, "accepted")
        segment = self.transcripts.get_segment("seg_0001")
        assert segment is not None
        self.assertEqual(segment.corrected_text, "围压需要进一步提高。")
        self.assertEqual(segment.raw_text, "微压需要提高")
        edits = self.transcripts.list_edits(run_id="asr-run-001")
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0].edit_type, "deep_correction_accept")
        self.assertEqual(edits[0].metadata["correction_run_id"], run.id)
        self.assertEqual(edits[0].metadata["change_id"], change.id)
        self.assertEqual(edits[0].metadata["review_session"], "session-1")

        unchanged = self.repository.propose_change(
            run.id,
            paragraph_id="p1",
            change_type="punctuation",
            before_text="围压需要进一步提高。",
            after_text="围压需要进一步提高。",
            source_segment_ids=["seg_0001"],
            change_id="no-op-change",
        )
        self.repository.accept_change_and_apply(unchanged.id)
        self.assertEqual(len(self.transcripts.list_edits(run_id="asr-run-001")), 1)
        self.assertEqual(
            [item.to_status for item in self.repository.list_change_events(unchanged.id)],
            ["proposed", "accepted"],
        )

        ambiguous = self.repository.propose_change(
            run.id,
            change_type="terminology",
            before_text="旧",
            after_text="新",
            source_segment_ids=["seg_0001", "seg_0002"],
            change_id="ambiguous-change",
        )
        with self.assertRaisesRegex(ValueError, "恰好一个"):
            self.repository.accept_change_and_apply(ambiguous.id)
        self.assertEqual(self.repository.get_change(ambiguous.id).status, "proposed")

    def test_complete_and_export_full_safe_markdown_atomically(self) -> None:
        run = self._create_running()
        self.repository.save_paragraphs(
            run.id,
            [
                _paragraph("p1", 0, 0, 1500, ["seg_0001"]),
                _paragraph(
                    "p2", 1, 1500, 3000, ["seg_0002"],
                    corrected="## 伪标题 <script>alert(1)</script>",
                ),
            ],
        )
        change = self.repository.propose_change(
            run.id,
            paragraph_id="p1",
            change_type="terminology",
            before_text="微压",
            after_text="围压",
            reason="术语核验",
            source_segment_ids=["seg_0001"],
        )
        self.repository.review_change(change.id, decision="accepted", reason="人工确认")
        self.repository.add_evidence(
            run.id,
            change_id=change.id,
            evidence_type="external",
            title="围压定义 [权威]",
            url="https://example.test/spec?q=a%20b",
            summary="可公开查证",
        )
        self.repository.add_evidence(
            run.id,
            paragraph_id="p2",
            evidence_type="external",
            title="危险",
            url="javascript:alert(1)",
        )
        result = {
            "processing_boundaries": ["仅修正术语与标点，不改变原意"],
            "uncertain_items": ["Speaker 2 的真实身份尚未确认"],
            "knowledge_cards": [
                {"title": "围压", "summary": "工程参数", "tags": ["岩体力学"]},
                {"title": "对照计算", "summary": "待办事项"},
            ],
            "relations": [{"source": "围压", "target": "对照计算", "label": "影响"}],
        }
        completed = self.repository.complete_run(run.id, result=result)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(
            completed.result_checksum,
            hashlib.sha256(
                json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )

        target = self.root / "exports" / "meeting.md"
        exported = DeepCorrectionMarkdownExporter(self.repository).export(
            run.id, target, allowed_root=self.root / "exports"
        )
        self.assertEqual(exported.path, target)
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), exported.sha256)
        markdown = target.read_text(encoding="utf-8")
        for heading in (
            "处理元数据", "处理边界", "说话人", "术语校正表", "完整精校正文",
            "不确定项", "外部证据", "知识卡", "知识关系图", "审计摘要",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("[围压定义 \\[权威\\]](https://example.test/spec?q=a%20b)", markdown)
        self.assertNotIn("](javascript:", markdown)
        self.assertNotIn("<script>", markdown)
        self.assertIn("\\#\\# 伪标题", markdown)
        self.assertIn("```mermaid", markdown)
        refreshed = self.repository.get_run(run.id)
        assert refreshed is not None
        self.assertEqual(refreshed.output_path, str(target))
        self.assertEqual(refreshed.output_checksum, exported.sha256)

        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            DeepCorrectionMarkdownExporter(self.repository).export(run.id, target)
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaises(ValueError):
            DeepCorrectionMarkdownExporter(self.repository).export(
                run.id, self.root / "outside.md", allowed_root=self.root / "exports"
            )

    def test_latest_v2_export_uses_database_corrections_and_compare_and_swap(self) -> None:
        self.transcripts.update_corrected_text("seg_0001", "围压需要进一步提高。")
        target = self.root / "transcripts" / "meeting.v2.json"
        target.parent.mkdir()
        target.write_text('{"stale":true}\n', encoding="utf-8")
        stale_checksum = hashlib.sha256(target.read_bytes()).hexdigest()

        with self.assertRaises(FileExistsError):
            self.transcripts.write_latest_v2("asr-run-001", target)
        with self.assertRaises(RuntimeError):
            self.transcripts.write_latest_v2(
                "asr-run-001", target, overwrite=True,
                expected_existing_checksum="0" * 64,
            )
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), stale_checksum)

        written = self.transcripts.write_latest_v2(
            "asr-run-001", target, overwrite=True,
            expected_existing_checksum=stale_checksum,
            allowed_root=self.root / "transcripts",
        )
        payload = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(payload["segments"][0]["raw_text"], "微压需要提高")
        self.assertEqual(payload["segments"][0]["corrected_text"], "围压需要进一步提高。")

    def test_persist_result_bundle_rolls_back_everything_on_midway_failure(self) -> None:
        run = self._create_running()
        other = self.repository.create_run(
            "asr-run-001", provider="deepseek", model="deepseek-chat",
            run_id="correction-other",
        )
        other = self.repository.start_run(other.id)
        self.repository.add_evidence(
            other.id,
            evidence_type="model",
            title="占用全局 ID",
            evidence_id="taken-evidence",
        )

        paragraphs = [_paragraph("bundle-p1", 0, 0, 1500, ["seg_0001"])]
        changes = [{
            "id": "bundle-change",
            "paragraph_id": "bundle-p1",
            "change_type": "terminology",
            "before_text": "围压需要提高。",
            "after_text": "围压需要进一步提高。",
            "source_segment_ids": ["seg_0001"],
        }]
        evidence = [
            {"id": "bundle-e1", "change_id": "bundle-change", "evidence_type": "source"},
            {"id": "taken-evidence", "evidence_type": "model"},
        ]
        with self.assertRaises(ValueError):
            self.repository.persist_result_bundle(
                run.id,
                paragraphs=paragraphs,
                changes=changes,
                evidence=evidence,
                result={"knowledge_cards": []},
                quality_summary={"status": "pass"},
            )

        self.assertEqual(self.repository.get_run(run.id).status, "running")
        self.assertEqual(self.repository.list_paragraphs(run.id), [])
        self.assertEqual(self.repository.list_changes(run.id), [])
        self.assertEqual(self.repository.list_evidence(run.id), [])

        completed = self.repository.persist_result_bundle(
            run.id,
            paragraphs=paragraphs,
            changes=changes,
            evidence=[{"id": "bundle-e1", "change_id": "bundle-change", "evidence_type": "source"}],
            result={"knowledge_cards": []},
            quality_summary={"status": "pass"},
        )
        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(self.repository.list_paragraphs(run.id)), 1)
        self.assertEqual(len(self.repository.list_changes(run.id)), 1)
        self.assertEqual(len(self.repository.list_evidence(run.id)), 1)

    def _create_running(self):
        run = self.repository.create_run(
            "asr-run-001", provider="deepseek", model="deepseek-chat",
            config={"mode": "deep"}, run_id="correction-001",
        )
        return self.repository.start_run(run.id)


def _paragraph(
    paragraph_id: str,
    ordinal: int,
    start_ms: int,
    end_ms: int,
    source_segment_ids: list[str],
    *,
    corrected: str = "围压需要提高。",
) -> dict[str, object]:
    return {
        "id": paragraph_id,
        "ordinal": ordinal,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "speaker_id": "spk_00" if ordinal == 0 else "spk_01",
        "source_segment_ids": source_segment_ids,
        "original_text": "微压需要提高" if ordinal == 0 else "先做对照计算",
        "corrected_text": corrected,
        "quality_status": "pass",
    }


def _transcript() -> TranscriptV2:
    return TranscriptV2(
        TranscriptSource("项目会议.wav", "a" * 64, 3000),
        TranscriptRun("asr-run-001", "accuracy", "qwen3-mlx", "Qwen3-ASR-1.7B", "Chinese"),
        [
            TranscriptSpeaker("spk_00", "张工", "manual"),
            TranscriptSpeaker("spk_01", "Speaker 2", "automatic"),
        ],
        [
            TranscriptSegment("seg_0001", 0, 0, 1500, "spk_00", "微压需要提高", "围压需要提高。"),
            TranscriptSegment("seg_0002", 1, 1500, 3000, "spk_01", "先做对照计算", "先做对照计算。"),
        ],
        TranscriptQuality("pass"),
    )


if __name__ == "__main__":
    unittest.main()
