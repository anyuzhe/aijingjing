from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from media_knowledge.desktop.app import create_application
except (ImportError, RuntimeError):  # pragma: no cover - desktop extra is optional
    create_application = None


def _document() -> dict[str, object]:
    return {
        "id": "document-audio-1",
        "title": "冬季技术分享",
        "media_type": "audio",
    }


def _latest_transcript() -> dict[str, object]:
    return {
        "run": {"id": "transcript-run-1"},
        "transcript": {
            "speakers": [{"id": "S1", "display_name": "悟鸣"}],
            "segments": [
                {
                    "id": "segment-1",
                    "ordinal": 0,
                    "start_ms": 0,
                    "end_ms": 4_000,
                    "speaker_id": "S1",
                    "raw_text": "奥格森林可以存 markdown。",
                    "corrected_text": "奥格森林可以存 Markdown。",
                }
            ],
        },
    }


def _snapshot(*, status: str = "completed", decision: str = "proposed") -> dict[str, object]:
    return {
        "run": {
            "id": "correction-run-1",
            "status": status,
            "result": {
                "transcript": {
                    "speakers": [{"id": "S1", "display_name": "悟鸣"}]
                }
            },
        },
        "paragraphs": [
            {
                "id": "paragraph-1",
                "ordinal": 0,
                "start_ms": 0,
                "end_ms": 4_000,
                "speaker_id": "S1",
                "original_text": "奥格森林可以存 markdown。",
                "corrected_text": "Obsidian 可以保存 Markdown。",
                "quality_status": "review",
            }
        ],
        "changes": [
            {
                "id": "change-1",
                "paragraph_id": "paragraph-1",
                "change_type": "terminology",
                "before_text": "奥格森林",
                "after_text": "Obsidian",
                "reason": "上下文明确讨论 Markdown 知识管理软件。",
                "confidence": 0.96,
                "status": decision,
                "metadata": {"uncertain": False},
            }
        ],
        "evidence": [
            {
                "id": "evidence-1",
                "change_id": "change-1",
                "evidence_type": "external",
                "title": "Obsidian 官方帮助",
                "url": "https://help.obsidian.md/",
                "summary": "Markdown 知识管理软件说明",
            }
        ],
    }


@unittest.skipIf(create_application is None, "PySide6 desktop components are unavailable")
class DeepCorrectionAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.application, self.window = create_application(
            Path(self.temporary.name) / "data"
        )
        self.dialogs = []

    def tearDown(self) -> None:
        for dialog in self.dialogs:
            if dialog in self.window._deep_correction_contexts:
                dialog.mark_idle()
                dialog.close()
        self.window._active_db_operation_token = None
        self.window.close()
        self.application.processEvents()
        self.temporary.cleanup()

    def _open_dialog(self):
        with (
            patch.object(
                self.window,
                "_selected_source_context",
                return_value=("document-audio-1", None, None),
            ),
            patch.object(self.window, "_document_record", return_value=_document()),
            patch.object(
                self.window.controller,
                "latest_transcript",
                return_value=_latest_transcript(),
            ),
        ):
            dialog = self.window.open_deep_correction_dialog()
        assert dialog is not None
        if dialog not in self.dialogs:
            self.dialogs.append(dialog)
        self.application.processEvents()
        return dialog

    def test_menu_entry_opens_only_one_workbench_for_the_same_transcript(self) -> None:
        knowledge_menu = next(
            action.menu()
            for action in self.window.menuBar().actions()
            if action.text() == "知识库"
        )
        self.assertIsNotNone(knowledge_menu)
        assert knowledge_menu is not None
        self.assertIn(
            "深度精校与证据复核…",
            [action.text() for action in knowledge_menu.actions()],
        )

        first = self._open_dialog()
        second = self._open_dialog()

        self.assertIs(first, second)
        self.assertEqual(len(self.window._deep_correction_dialogs), 1)
        self.assertIn("奥格森林", first.raw_editor.toPlainText())
        self.assertEqual(first.state, "idle")

    def test_worker_lifecycle_review_and_export_release_the_busy_guard(self) -> None:
        dialog = self._open_dialog()
        snapshot = _snapshot()
        review_state = {"decision": "proposed"}

        def current_snapshot(_run_id):
            value = copy.deepcopy(snapshot)
            value["changes"][0]["status"] = review_state["decision"]
            return value

        test_case = self

        class FakeWorkflow:
            def run(
                self,
                run_id,
                *,
                progress=None,
                cancellation=None,
                correction_run_id=None,
                run_created=None,
            ):
                test_case.assertEqual(run_id, "transcript-run-1")
                test_case.assertIsNotNone(cancellation)
                test_case.assertIsNone(correction_run_id)
                assert progress is not None
                assert run_created is not None
                run_created("correction-run-1")
                progress("semantic_correction", 5, 11, "正在恢复专业术语")
                return {
                    "correction_run_id": "correction-run-1",
                    "output_path": "/tmp/冬季技术分享-完整精校.md",
                }

            def review_change(self, change_id, *, decision):
                test_case.assertEqual(change_id, "change-1")
                review_state["decision"] = decision
                return {"change": {"id": change_id, "status": decision}}

            def export(self, correction_run_id):
                test_case.assertEqual(correction_run_id, "correction-run-1")
                return {"path": "/tmp/冬季技术分享-完整精校.md"}

        workflow = FakeWorkflow()

        with (
            patch.object(
                self.window.controller,
                "_deep_correction_workflow",
                return_value=workflow,
            ),
            patch.object(
                self.window.controller,
                "deep_correction_snapshot",
                side_effect=current_snapshot,
            ),
            patch.object(
                self.window.thread_pool,
                "start",
                side_effect=lambda worker: worker.run(),
            ),
            patch("media_knowledge.desktop.app.QMessageBox.information") as information,
        ):
            dialog.start_button.click()
            self.application.processEvents()
            self.assertEqual(dialog.state, "completed")
            self.assertEqual(dialog.change_list.count(), 1)
            self.assertIn("Obsidian", dialog.corrected_editor.toPlainText())
            self.assertIsNone(self.window._active_db_operation_token)
            self.assertIsNone(
                self.window._deep_correction_contexts[dialog]["worker"]
            )

            dialog.accept_button.click()
            self.application.processEvents()
            self.assertEqual(dialog.changes[0].status, "accepted")
            self.assertIsNone(self.window._active_db_operation_token)

            dialog.export_button.click()
            self.application.processEvents()
            self.assertIn("/tmp/冬季技术分享-完整精校.md", dialog.feedback_label.text())
            self.assertIsNone(self.window._active_db_operation_token)
            information.assert_called_once()
            self.assertIn(
                "/tmp/冬季技术分享-完整精校.md",
                information.call_args.args[2],
            )

    def test_cancel_and_failure_retry_are_identity_safe_and_reuse_run_id(self) -> None:
        dialog = self._open_dialog()
        started = []
        calls: list[str | None] = []

        def deep_correct(
            _run_id,
            *,
            progress=None,
            cancellation=None,
            correction_run_id=None,
        ):
            calls.append(correction_run_id)
            return {
                "correction_run_id": "correction-run-1",
                "snapshot": _snapshot(),
                "output_path": "/tmp/冬季技术分享-完整精校.md",
            }

        with (
            patch.object(
                self.window.controller,
                "deep_correct_transcript",
                side_effect=deep_correct,
                create=True,
            ),
            patch.object(
                self.window.controller,
                "deep_correction_snapshot",
                return_value=_snapshot(),
                create=True,
            ),
            patch.object(self.window.thread_pool, "start", side_effect=started.append),
        ):
            dialog.start_button.click()
            self.assertEqual(len(started), 1)
            first_worker = started[0]
            first_context = self.window._deep_correction_contexts[dialog]
            token = first_context["cancellation"]
            dialog.cancel_button.click()
            self.assertTrue(token.cancelled)
            first_worker.signals.error.emit("任务已取消")
            first_worker.signals.finished.emit()
            self.application.processEvents()
            self.assertEqual(dialog.state, "cancelled")
            self.assertIsNone(self.window._active_db_operation_token)

            dialog.start_button.click()
            self.assertEqual(len(started), 2)
            failing_worker = started[1]
            failing_worker.signals.progress.emit(
                {
                    "correction_run_id": "correction-run-1",
                    "stage": "web_verification",
                    "message": "核验公开证据",
                }
            )
            failing_worker.signals.error.emit("外部核验服务超时")
            failing_worker.signals.finished.emit()
            self.application.processEvents()
            self.assertEqual(dialog.state, "failed")
            self.assertTrue(dialog.retry_button.isEnabled())
            self.assertIsNone(self.window._active_db_operation_token)

            dialog.retry_button.click()
            self.assertEqual(len(started), 3)
            retry_worker = started[2]
            retry_worker.run()
            self.application.processEvents()
            self.assertEqual(calls, ["correction-run-1"])
            self.assertEqual(dialog.state, "completed")
            self.assertIsNone(self.window._active_db_operation_token)

    def test_bulk_review_requests_are_serialized_instead_of_dropped(self) -> None:
        dialog = self._open_dialog()
        snapshot = _snapshot()
        snapshot["paragraphs"].append(
            {
                "id": "paragraph-2",
                "ordinal": 1,
                "start_ms": 4_000,
                "end_ms": 8_000,
                "speaker_id": "S1",
                "original_text": "RED 会检索片段。",
                "corrected_text": "RAG 会检索片段。",
                "quality_status": "review",
            }
        )
        snapshot["changes"].append(
            {
                "id": "change-2",
                "paragraph_id": "paragraph-2",
                "change_type": "terminology",
                "before_text": "RED",
                "after_text": "RAG",
                "reason": "上下文明确讨论检索增强生成。",
                "confidence": 0.91,
                "status": "proposed",
                "metadata": {"uncertain": False},
            }
        )
        snapshot["evidence"].append(
            {
                "id": "evidence-2",
                "change_id": "change-2",
                "evidence_type": "external",
                "title": "RAG 说明",
                "url": "https://example.test/rag",
                "summary": "检索增强生成",
            }
        )
        context = self.window._deep_correction_contexts[dialog]
        context["correction_run_id"] = "correction-run-1"
        context["snapshot"] = copy.deepcopy(snapshot)
        self.window._apply_deep_correction_snapshot(dialog, snapshot)
        dialog.mark_completed()
        decisions = {"change-1": "proposed", "change-2": "proposed"}
        started = []

        def review(change_id, decision):
            decisions[change_id] = decision
            return {
                "change": {"id": change_id, "status": decision},
                "index": {"updated": decision == "accepted"},
            }

        def current_snapshot(_run_id):
            value = copy.deepcopy(snapshot)
            for change in value["changes"]:
                change["status"] = decisions[change["id"]]
            return value

        with (
            patch.object(
                self.window.controller,
                "review_deep_correction_change",
                side_effect=review,
                create=True,
            ),
            patch.object(
                self.window.controller,
                "deep_correction_snapshot",
                side_effect=current_snapshot,
                create=True,
            ),
            patch.object(self.window.thread_pool, "start", side_effect=started.append),
        ):
            self.assertEqual(
                dialog.accept_eligible_bulk(),
                ("change-1", "change-2"),
            )
            self.assertEqual(len(started), 1)
            self.assertEqual(context["pending_reviews"], [("change-2", "accepted")])

            started[0].run()
            self.application.processEvents()
            self.assertEqual(len(started), 2)
            started[1].run()
            self.application.processEvents()

        self.assertEqual(decisions, {"change-1": "accepted", "change-2": "accepted"})
        self.assertEqual([change.status for change in dialog.changes], ["accepted", "accepted"])
        self.assertEqual(context["pending_reviews"], [])
        self.assertIsNone(self.window._active_db_operation_token)


if __name__ == "__main__":
    unittest.main()
