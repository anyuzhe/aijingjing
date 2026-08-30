from __future__ import annotations

import os
import unittest
from dataclasses import replace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from media_knowledge.desktop.deep_correction_dialog import (  # noqa: E402
    DEEP_CORRECTION_STEPS,
    QT_WIDGETS_AVAILABLE,
    CorrectionChange,
    CorrectionEvidence,
    DeepCorrectionDialog,
    change_is_bulk_eligible,
    confidence_text,
    format_elapsed,
    format_time_range,
    is_navigable_evidence_url,
)


def _sample_changes() -> tuple[CorrectionChange, ...]:
    return (
        CorrectionChange(
            id="term-obsidian",
            start_ms=15_460,
            end_ms=17_400,
            speaker="S1｜悟鸣",
            raw_text="奥格森林可以存 markdown。" * 20,
            corrected_text="Obsidian 可以保存 Markdown。" * 20,
            confidence=0.96,
            evidence=(
                CorrectionEvidence("Obsidian 官方帮助", "https://help.obsidian.md/"),
            ),
            rationale="上下文明确讨论 Markdown 知识管理软件。",
        ),
        CorrectionChange(
            id="term-unknown",
            start_ms=25_000,
            end_ms=25_800,
            speaker="S1｜悟鸣",
            raw_text="Darling Gump Base",
            corrected_text="现场材料包目录名",
            confidence=0.93,
            rationale="缺少屏幕画面，只有文本推断。",
        ),
        CorrectionChange(
            id="term-cli",
            start_ms=24_290,
            end_ms=27_150,
            speaker="S1｜悟鸣",
            raw_text="打开 CNN 开关。",
            corrected_text="打开 Command line interface / CLI。",
            confidence=0.92,
            uncertain=True,
            uncertainty_reason="需要回听按钮名称",
            evidence=(
                CorrectionEvidence("Obsidian CLI 说明", "https://help.obsidian.md/cli"),
            ),
            rationale="上下文与命令行流程一致，但音频仍需人工复核。",
        ),
        CorrectionChange(
            id="term-rag",
            start_ms=18_000,
            end_ms=18_500,
            speaker="S1｜悟鸣",
            raw_text="RED",
            corrected_text="RAG",
            confidence=0.89,
            evidence=(CorrectionEvidence("本地证据", "file:///tmp/raw-transcript.md"),),
            rationale="后文明确出现向量数据库和检索片段。",
        ),
    )


class DeepCorrectionHelperTests(unittest.TestCase):
    def test_steps_time_confidence_and_evidence_helpers_are_stable(self) -> None:
        self.assertEqual(len(DEEP_CORRECTION_STEPS), 11)
        self.assertEqual(format_elapsed(65), "01:05")
        self.assertEqual(format_elapsed(3_661), "01:01:01")
        self.assertEqual(format_elapsed("bad"), "00:00")
        self.assertEqual(format_time_range(65_000, 125_000), "01:05–02:05")
        self.assertEqual(confidence_text(0.92), "高置信度 · 92%")
        self.assertIn("中置信度", confidence_text(0.7))
        self.assertTrue(is_navigable_evidence_url("https://example.com/source"))
        self.assertTrue(is_navigable_evidence_url("file:///tmp/source.md"))
        self.assertFalse(is_navigable_evidence_url("javascript:alert(1)"))
        self.assertFalse(is_navigable_evidence_url("只是资料名称"))

    def test_bulk_acceptance_requires_all_safety_conditions(self) -> None:
        evidenced, no_evidence, uncertain, _other_evidenced = _sample_changes()
        self.assertTrue(change_is_bulk_eligible(evidenced))
        self.assertFalse(change_is_bulk_eligible(no_evidence))
        self.assertFalse(change_is_bulk_eligible(uncertain))
        self.assertFalse(
            change_is_bulk_eligible(
                CorrectionChange(
                    id="medium",
                    start_ms=0,
                    end_ms=1_000,
                    speaker="S1",
                    raw_text="a",
                    corrected_text="b",
                    confidence=0.7,
                    evidence=(CorrectionEvidence("证据", "https://example.com"),),
                )
            )
        )
        self.assertFalse(change_is_bulk_eligible(replace(evidenced, status="rejected")))


@unittest.skipUnless(QT_WIDGETS_AVAILABLE, "当前环境未安装 PySide6")
class DeepCorrectionDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.dialog = DeepCorrectionDialog(
            source_name="悟鸣-知识库.mp3",
            raw_text="完整原始转写",
            corrected_text="完整精校转写",
            changes=_sample_changes(),
        )
        self.application.processEvents()

    def tearDown(self) -> None:
        self.dialog.mark_idle()
        self.dialog.close()
        self.dialog.deleteLater()
        self.application.processEvents()

    def test_dialog_exposes_all_steps_parallel_text_and_accessible_metadata(self) -> None:
        from PySide6.QtWidgets import QPushButton

        self.assertEqual(self.dialog.progress_bar.maximum(), 11)
        self.assertEqual(len(self.dialog.step_labels), 11)
        self.assertEqual(self.dialog.change_list.count(), 4)
        self.assertTrue(self.dialog.raw_editor.isReadOnly())
        self.assertTrue(self.dialog.corrected_editor.isReadOnly())
        self.assertIn("奥格森林", self.dialog.raw_editor.toPlainText())
        self.assertIn("Obsidian", self.dialog.corrected_editor.toPlainText())
        self.assertEqual(self.dialog.time_range_label.text(), "时间：00:15–00:17")
        self.assertIn("S1｜悟鸣", self.dialog.speaker_label.text())
        self.assertIn("高置信度", self.dialog.confidence_label.text())
        self.assertEqual(self.dialog.uncertainty_label.text(), "不确定标记：无")
        self.assertIn("https://help.obsidian.md/", self.dialog.evidence_browser.toHtml())
        self.assertEqual(self.dialog.change_list.accessibleName(), "深度精校变更列表")
        self.assertEqual(self.dialog.progress_bar.accessibleName(), "深度精校总进度")
        self.assertEqual(self.dialog.raw_editor.accessibleName(), "当前变更原始识别文字，只读")
        self.assertTrue(self.dialog.start_button.accessibleName())
        self.assertTrue(self.dialog.cancel_button.accessibleName())
        self.assertTrue(self.dialog.retry_button.accessibleName())
        self.assertTrue(self.dialog.export_button.accessibleName())
        self.assertTrue(all(button.accessibleName() for button in self.dialog.findChildren(QPushButton)))

    def test_start_cancel_failure_retry_completion_and_export_signals(self) -> None:
        events: list[str] = []
        self.dialog.startRequested.connect(lambda: events.append("start"))
        self.dialog.cancelRequested.connect(lambda: events.append("cancel"))
        self.dialog.retryRequested.connect(lambda: events.append("retry"))
        self.dialog.exportRequested.connect(lambda: events.append("export"))

        self.dialog.start_button.click()
        self.assertEqual(self.dialog.state, "running")
        self.assertEqual(events, ["start"])
        self.dialog.set_progress(2, current_step=3, detail="正在合并碎片")
        self.assertEqual(self.dialog.progress_bar.value(), 2)
        self.assertIn("第 3/11 步", self.dialog.current_step_label.text())
        self.assertIn("进行中", self.dialog.step_labels[2].text())

        self.dialog.cancel_button.click()
        self.assertEqual(self.dialog.state, "cancelling")
        self.assertEqual(events, ["start", "cancel"])

        self.dialog.mark_failed("外部核验服务超时", "检查网络后重试，或关闭外部核验。")
        self.assertFalse(self.dialog.error_frame.isHidden())
        self.assertIn("外部核验服务超时", self.dialog.error_label.text())
        self.assertIn("检查网络后重试", self.dialog.recovery_label.text())
        self.assertTrue(self.dialog.retry_button.isEnabled())
        self.dialog.retry_button.click()
        self.assertEqual(self.dialog.state, "running")
        self.assertEqual(events[-1], "retry")

        self.dialog.mark_completed()
        self.assertEqual(self.dialog.progress_bar.value(), 11)
        self.assertIn("全部 11 步已完成", self.dialog.current_step_label.text())
        self.assertTrue(self.dialog.export_button.isEnabled())
        self.dialog.export_button.click()
        self.assertEqual(events[-1], "export")

    def test_decisions_and_bulk_acceptance_are_explicit_and_safe(self) -> None:
        accepted: list[str] = []
        rejected: list[str] = []
        self.dialog.acceptRequested.connect(accepted.append)
        self.dialog.rejectRequested.connect(rejected.append)
        self.dialog.mark_completed()

        self.assertEqual(
            self.dialog.eligible_bulk_change_ids(),
            ("term-obsidian", "term-rag"),
        )
        self.assertEqual(
            self.dialog.accept_eligible_bulk(),
            ("term-obsidian", "term-rag"),
        )
        self.assertEqual(accepted, ["term-obsidian", "term-rag"])
        self.assertEqual(self.dialog.eligible_bulk_change_ids(), ())
        self.assertEqual(self.dialog.changes[0].status, "accepted")
        self.assertEqual(self.dialog.changes[1].status, "pending")
        self.assertEqual(self.dialog.changes[2].status, "pending")

        self.dialog.change_list.setCurrentRow(1)
        self.assertEqual(self.dialog.reject_selected(), "term-unknown")
        self.assertEqual(rejected, ["term-unknown"])
        self.assertEqual(self.dialog.changes[1].status, "rejected")

        self.dialog.change_list.setCurrentRow(2)
        self.assertIn("需要回听按钮名称", self.dialog.uncertainty_label.text())
        self.assertTrue(self.dialog.accept_button.isEnabled())
        self.assertEqual(self.dialog.accept_selected(), "term-cli")
        self.assertEqual(accepted[-1], "term-cli")

    def test_long_text_expands_and_escape_requests_safe_cancel(self) -> None:
        initial_maximum = self.dialog.raw_editor.maximumHeight()
        self.assertLessEqual(initial_maximum, 230)
        self.dialog.toggle_text_button.click()
        self.assertGreater(self.dialog.raw_editor.maximumHeight(), 1_000_000)
        self.assertEqual(self.dialog.toggle_text_button.text(), "收起长文本")

        cancelled: list[bool] = []
        self.dialog.cancelRequested.connect(lambda: cancelled.append(True))
        self.dialog.mark_running()
        self.dialog.reject()
        self.assertEqual(self.dialog.state, "cancelling")
        self.assertEqual(cancelled, [True])

    def test_empty_change_list_still_shows_full_document_text(self) -> None:
        empty = DeepCorrectionDialog(
            source_name="完整精校稿.md",
            raw_text="整份原始文本",
            corrected_text="整份深度精校文本",
        )
        try:
            self.assertEqual(empty.change_list.count(), 0)
            self.assertEqual(empty.raw_editor.toPlainText(), "整份原始文本")
            self.assertEqual(empty.corrected_editor.toPlainText(), "整份深度精校文本")
            self.assertIn("当前没有变更项", empty.evidence_browser.toPlainText())
            self.assertIn("等待", empty.step_labels[0].text())
        finally:
            empty.close()
            empty.deleteLater()
            self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
