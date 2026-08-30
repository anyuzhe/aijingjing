from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from media_knowledge.desktop.transcript_editor import (  # noqa: E402
    QT_WIDGETS_AVAILABLE,
    TranscriptEditorDialog,
    format_editor_timestamp,
    speaker_label,
    transcript_route_text,
)
from media_knowledge.storage.database import KnowledgeDatabase  # noqa: E402
from media_knowledge.transcripts import (  # noqa: E402
    TranscriptQuality,
    TranscriptRepository,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptSpeaker,
    TranscriptV2,
)


def _sample_transcript() -> TranscriptV2:
    return TranscriptV2(
        source=TranscriptSource(
            "冬季技术会议.m4a",
            "e" * 64,
            5_500,
            original_uri="/tmp/winter-meeting.m4a",
        ),
        run=TranscriptRun(
            id="run-editor-001",
            profile="chinese-accuracy",
            provider="qwen3-mlx",
            model="Qwen3-ASR-1.7B",
            language="Chinese",
            word_timestamps=True,
            diarization_provider="pyannote",
        ),
        speakers=[
            TranscriptSpeaker("spk_00", "讲师", "manual"),
            TranscriptSpeaker("spk_01", "学员", "manual"),
        ],
        segments=[
            TranscriptSegment(
                "seg-editor-1",
                0,
                0,
                2_500,
                "spk_00",
                "微压需要提高",
                "围压需要提高。",
                0.82,
                ("terminology_review",),
            ),
            TranscriptSegment(
                "seg-editor-2",
                1,
                2_500,
                5_500,
                "spk_01",
                "微压需要复查",
                None,
                0.91,
            ),
        ],
        quality=TranscriptQuality("review", ("首段专业术语需要人工确认",)),
    )


class TranscriptEditorHelperTests(unittest.TestCase):
    def test_route_timestamp_and_speaker_helpers(self) -> None:
        transcript = _sample_transcript()
        self.assertEqual(format_editor_timestamp(65_999), "01:05")
        self.assertEqual(format_editor_timestamp("bad"), "00:00")
        self.assertIn("qwen3-mlx", transcript_route_text(transcript))
        self.assertIn("说话人：pyannote", transcript_route_text(transcript))
        self.assertEqual(speaker_label(transcript.speakers[0]), "讲师")
        self.assertEqual(speaker_label(None, "spk_x"), "spk_x")


@unittest.skipUnless(QT_WIDGETS_AVAILABLE, "当前环境未安装 PySide6")
class TranscriptEditorDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = KnowledgeDatabase(Path(self.temporary.name) / "knowledge.db")
        self.repository = TranscriptRepository(self.database)
        self.repository.save_transcript(_sample_transcript())
        self.dialog = TranscriptEditorDialog(
            self.repository,
            "run-editor-001",
            media_path="/tmp/winter-meeting.m4a",
            actor="editor-test",
        )
        self.application.processEvents()

    def tearDown(self) -> None:
        # Avoid a confirmation box in failed tests while still exercising normal
        # closeEvent behavior whenever there are no pending drafts.
        for draft in self.dialog._drafts.values():
            draft.baseline = draft.text
        self.dialog.close()
        self.dialog.deleteLater()
        self.application.processEvents()
        self.database.close()
        self.temporary.cleanup()

    def test_dialog_displays_route_quality_raw_text_and_accessible_controls(self) -> None:
        self.assertIn("Qwen3-ASR-1.7B", self.dialog.route_label.text())
        self.assertEqual(self.dialog.quality_label.text(), "需要复核")
        self.assertEqual(self.dialog.warning_list.count(), 1)
        self.assertTrue(self.dialog.raw_editor.isReadOnly())
        self.assertEqual(self.dialog.raw_editor.toPlainText(), "微压需要提高")
        self.assertEqual(self.dialog.corrected_editor.toPlainText(), "围压需要提高。")
        self.assertEqual(self.dialog.segment_list.count(), 2)
        self.assertEqual(self.dialog.save_button.accessibleName(), "保存全部待处理校订")
        self.assertEqual(self.dialog.play_button.accessibleName(), "从当前片段时间点播放")

    def test_save_is_per_segment_audited_and_never_overwrites_raw_text(self) -> None:
        saved_events: list[tuple[str, object]] = []
        self.dialog.transcriptSaved.connect(lambda run_id, ids: saved_events.append((run_id, ids)))
        self.dialog.corrected_editor.setPlainText("围压需要显著提高。")
        self.assertEqual(self.dialog.dirty_segment_ids, ("seg-editor-1",))

        saved = self.dialog.save_changes(reason="核对录音确认术语", actor="reviewer")

        self.assertEqual(saved, ("seg-editor-1",))
        first = self.repository.get_segment("seg-editor-1")
        second = self.repository.get_segment("seg-editor-2")
        assert first is not None and second is not None
        self.assertEqual(first.raw_text, "微压需要提高")
        self.assertEqual(first.corrected_text, "围压需要显著提高。")
        # The same phrase in another segment is deliberately untouched: this is
        # a fact-level edit, never a global string replacement.
        self.assertEqual(second.raw_text, "微压需要复查")
        self.assertIsNone(second.corrected_text)
        edit = self.repository.list_edits(run_id="run-editor-001")[-1]
        self.assertEqual(edit.reason, "核对录音确认术语")
        self.assertEqual(edit.actor, "reviewer")
        self.assertEqual(edit.target_id, "seg-editor-1")
        self.assertEqual(saved_events, [("run-editor-001", ("seg-editor-1",))])
        self.assertIn("原始识别文字保持不变", self.dialog.feedback_label.text())

    def test_drafts_survive_navigation_until_they_are_saved(self) -> None:
        self.dialog.corrected_editor.setPlainText("第一段待保存")
        self.dialog.segment_list.setCurrentRow(1)
        self.dialog.corrected_editor.setPlainText("第二段待保存")
        self.dialog.segment_list.setCurrentRow(0)
        self.assertEqual(self.dialog.corrected_editor.toPlainText(), "第一段待保存")
        self.assertEqual(
            self.dialog.dirty_segment_ids,
            ("seg-editor-1", "seg-editor-2"),
        )

    def test_speaker_operations_use_repository_audit_methods(self) -> None:
        events: list[tuple[str, object]] = []
        self.dialog.speakerChanged.connect(lambda run_id, ids: events.append((run_id, ids)))

        renamed = self.dialog.rename_speaker("spk_00", "张老师")
        reassigned = self.dialog.reassign_segment("seg-editor-2", "spk_00")
        affected = self.dialog.merge_speakers("spk_01", "spk_00")

        self.assertEqual(renamed.display_name, "张老师")
        self.assertEqual(reassigned.speaker_id, "spk_00")
        self.assertEqual(affected, 0)
        self.assertEqual(self.dialog.rename_speaker_combo.count(), 1)
        edit_types = {
            edit.edit_type for edit in self.repository.list_edits(run_id="run-editor-001")
        }
        self.assertEqual(
            edit_types,
            {"speaker_rename", "speaker_assignment", "speaker_merge"},
        )
        self.assertEqual(len(events), 3)

    def test_play_request_emits_exact_segment_bounds_and_invokes_callback(self) -> None:
        callbacks: list[tuple[str, int, int]] = []
        signals: list[tuple[str, int, int]] = []
        self.dialog.play_callback = lambda path, start, end: callbacks.append((path, start, end))
        self.dialog.playRequested.connect(
            lambda path, start, end: signals.append((path, start, end))
        )
        self.dialog.segment_list.setCurrentRow(1)

        self.assertTrue(self.dialog.request_play_current())
        expected = ("/tmp/winter-meeting.m4a", 2_500, 5_500)
        self.assertEqual(callbacks, [expected])
        self.assertEqual(signals, [expected])

    def test_empty_audit_reason_shows_clear_error_from_button_handler(self) -> None:
        self.dialog.corrected_editor.setPlainText("待保存")
        self.dialog.reason_edit.clear()
        self.dialog._save_clicked()
        self.assertIn("请填写修改原因", self.dialog.last_error)
        self.assertEqual(self.dialog.dirty_segment_ids, ("seg-editor-1",))


if __name__ == "__main__":
    unittest.main()
