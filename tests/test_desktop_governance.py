from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from media_knowledge.desktop.controller import DesktopController
from media_knowledge.desktop.privacy import verify_share_copy
from media_knowledge.product import DesktopSettings


class DesktopGovernanceTests(unittest.TestCase):
    @staticmethod
    def _controller(root: Path) -> DesktopController:
        controller = DesktopController(root, migrate_legacy=False)
        controller.save_settings(
            DesktopSettings(
                default_model="local-extractive",
                embedding_provider="hash",
                embedding_model="hash-384-v1",
                auto_synthesize_notes=False,
                enable_cloud_vision=False,
            )
        )
        return controller

    def test_answer_capture_creates_note_source_relation_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self._controller(root / "data")
            source = root / "evidence.md"
            source.write_text("# FDE\n\nFDE 是一种可复核的工程知识。", encoding="utf-8")
            self.assertEqual(controller.ingest([source]).succeeded, 1)
            document_id = str(controller.documents()[0]["id"])

            captured = controller.capture_answer_as_knowledge(
                markdown="FDE 的核心价值是保留可复核证据。",
                question="FDE 的价值是什么？",
                title="FDE 的可复核价值",
                item_type="analysis",
                status="needs-review",
                summary="说明 FDE 为什么需要证据溯源。",
                aliases=["FDE evidence"],
                tags=["fde", "evidence"],
                conversation_id="conversation-test",
                answer_id="answer-test",
                evidence_document_ids=[document_id, "missing-document"],
            )

            self.assertEqual(captured["maturity"], "compiled")
            detail = controller.knowledge_item(str(captured["id"]))
            self.assertEqual(detail["status"], "needs-review")
            self.assertTrue(
                any(
                    relation["relation_type"] == "supports"
                    and relation["related_type"] == "source"
                    for relation in detail["relations"]
                )
            )
            note = controller.paths.notes / detail["metadata"]["note_relative_path"]
            self.assertTrue(note.is_file())
            note_text = note.read_text(encoding="utf-8")
            self.assertIn("knowledge_type: \"analysis\"", note_text)
            self.assertIn("## 来源证据", note_text)
            self.assertIn(document_id, note_text)
            self.assertNotIn("missing-document", note_text)

            report = controller.knowledge_health()
            self.assertGreaterEqual(report["counts"]["items"], 2)
            self.assertEqual(report["counts"]["needs_review"], 1)
            updated = controller.update_knowledge_item(str(captured["id"]), status="current")
            self.assertEqual(updated["status"], "current")
            self.assertIn('status: "current"', note.read_text(encoding="utf-8"))

            self.assertTrue(controller.delete_knowledge_item(str(captured["id"])))
            self.assertFalse(note.exists())
            trash_items = controller.knowledge_trash_items()
            self.assertEqual(len(trash_items), 1)
            self.assertEqual(trash_items[0]["item_id"], captured["id"])
            tombstone = (
                controller.paths.trash
                / "knowledge-items"
                / str(trash_items[0]["tombstone_id"])
                / "tombstone.json"
            )
            payload = json.loads(tombstone.read_text(encoding="utf-8"))
            self.assertEqual(payload["item"]["id"], captured["id"])
            self.assertEqual(payload["item"]["aliases"], ["FDE evidence"])
            self.assertEqual(set(payload["item"]["tags"]), {"fde", "evidence"})
            self.assertEqual(len(payload["relations"]), 1)
            self.assertEqual(payload["note"]["original_relative_path"], detail["metadata"]["note_relative_path"])
            self.assertTrue(tombstone.with_name("note.md").is_file())

            restored = controller.restore_knowledge_item(str(trash_items[0]["tombstone_id"]))
            self.assertEqual(restored["id"], captured["id"])
            self.assertEqual(restored["aliases"], ["FDE evidence"])
            self.assertEqual(set(restored["tags"]), {"fde", "evidence"})
            self.assertEqual(restored["restored_relation_count"], 1)
            self.assertEqual(controller.knowledge_trash_items(), [])
            restored_detail = controller.knowledge_item(str(captured["id"]))
            restored_note = controller.paths.notes / restored_detail["metadata"]["note_relative_path"]
            self.assertEqual(restored_note, note)
            self.assertTrue(restored_note.is_file())
            self.assertIn("FDE 的核心价值", restored_note.read_text(encoding="utf-8"))

    def test_restore_uses_safe_note_name_when_original_path_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(Path(temporary) / "data")
            captured = controller.capture_answer_as_knowledge(
                markdown="需要恢复的正文",
                question="恢复什么？",
                title="冲突恢复测试",
                aliases=["restore-alias"],
                tags=["restore"],
            )
            original = controller.paths.notes / captured["metadata"]["note_relative_path"]
            self.assertTrue(controller.delete_knowledge_item(str(captured["id"])))
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_text("# 用户新建的同名笔记\n", encoding="utf-8")
            trash_item = controller.knowledge_trash_items()[0]

            restored = controller.restore_knowledge_item(str(trash_item["tombstone_id"]))

            restored_note = controller.paths.notes / restored["metadata"]["note_relative_path"]
            self.assertNotEqual(restored_note, original)
            self.assertIn("--restored-", restored_note.name)
            self.assertEqual(original.read_text(encoding="utf-8"), "# 用户新建的同名笔记\n")
            self.assertIn("需要恢复的正文", restored_note.read_text(encoding="utf-8"))

    def test_tombstone_failure_never_deletes_database_item_or_note(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(Path(temporary) / "data")
            captured = controller.capture_answer_as_knowledge(
                markdown="必须保留",
                question="失败时如何？",
                title="删除回滚测试",
            )
            note = controller.paths.notes / captured["metadata"]["note_relative_path"]

            with mock.patch(
                "media_knowledge.desktop.controller._atomic_json",
                side_effect=OSError("simulated tombstone failure"),
            ):
                with self.assertRaises(OSError):
                    controller.delete_knowledge_item(str(captured["id"]))

            self.assertEqual(controller.knowledge_item(str(captured["id"]))["id"], captured["id"])
            self.assertTrue(note.is_file())
            self.assertEqual(controller.knowledge_trash_items(), [])

    def test_note_move_failure_rolls_database_deletion_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = self._controller(Path(temporary) / "data")
            captured = controller.capture_answer_as_knowledge(
                markdown="数据库和笔记必须一起回滚",
                question="如何保持一致？",
                title="文件失败回滚测试",
                aliases=["rollback"],
                tags=["recovery"],
            )
            note = controller.paths.notes / captured["metadata"]["note_relative_path"]
            original_unlink = Path.unlink

            def fail_original_note(path: Path, *args, **kwargs):
                if path == note:
                    raise OSError("simulated note removal failure")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", new=fail_original_note):
                with self.assertRaises(OSError):
                    controller.delete_knowledge_item(str(captured["id"]))

            restored = controller.knowledge_item(str(captured["id"]))
            self.assertEqual(restored["aliases"], ["rollback"])
            self.assertEqual(restored["tags"], ["recovery"])
            self.assertTrue(note.is_file())
            self.assertEqual(controller.knowledge_trash_items(), [])

    def test_privacy_scan_is_redacted_and_share_copy_is_local_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = self._controller(root / "data")
            safe_note = controller.paths.notes / "正式知识" / "analysis" / "public.md"
            safe_note.parent.mkdir(parents=True, exist_ok=True)
            safe_note.write_text("# 可分享知识\n\n这是经过复核的公开说明。\n", encoding="utf-8")

            destination = root / "safe-share"
            share = controller.create_safe_share_copy(destination, include_notes=True)
            self.assertEqual(share["status"], "created")
            self.assertEqual(Path(str(share["destination"])), destination.resolve())
            self.assertEqual(verify_share_copy(destination)["status"], "verified")
            self.assertTrue((destination / "share_manifest.json").is_file())

            secret = "unit-test-redaction-value-1234567890"
            safe_note.write_text(f"password={secret}\n", encoding="utf-8")
            report = controller.privacy_scan()
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertEqual(report["status"], "blocked")
            self.assertNotIn(secret, serialized)
            self.assertNotIn(str(controller.paths.root), serialized)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from media_knowledge.desktop.app import KnowledgeTrashDialog, create_application
except (ImportError, RuntimeError):  # pragma: no cover - desktop extra is optional
    create_application = None
    KnowledgeTrashDialog = None


@unittest.skipIf(create_application is None, "PySide6 desktop components are unavailable")
class DesktopGovernanceUITests(unittest.TestCase):
    def test_knowledge_and_safety_actions_are_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            application, window = create_application(Path(temporary) / "data")
            try:
                self.assertEqual(window.left_tabs.tabText(1), "知识")
                self.assertEqual(window.knowledge_list.accessibleName(), "正式知识列表")
                self.assertEqual(window.knowledge_trash_button.accessibleName(), "打开知识回收站")
                menu_labels = [action.text() for action in window.menuBar().actions()]
                self.assertIn("数据安全", menu_labels)
            finally:
                window.close()
                application.processEvents()

    def test_trash_dialog_selects_and_restores_a_deleted_item(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            application, window = create_application(Path(temporary) / "data")
            dialog = None
            try:
                captured = window.controller.capture_answer_as_knowledge(
                    markdown="UI 恢复正文",
                    question="如何恢复？",
                    title="UI 回收站测试",
                    aliases=["ui-restore"],
                    tags=["ui"],
                )
                self.assertTrue(window.controller.delete_knowledge_item(str(captured["id"])))
                assert KnowledgeTrashDialog is not None
                dialog = KnowledgeTrashDialog(window.controller, window)
                self.assertEqual(dialog.trash_list.accessibleName(), "知识回收站列表")
                self.assertEqual(dialog.trash_list.count(), 1)
                self.assertFalse(dialog.restore_button.isEnabled())
                dialog.trash_list.setCurrentRow(0)
                application.processEvents()
                self.assertTrue(dialog.restore_button.isEnabled())

                dialog._restore_selected()

                self.assertEqual(dialog.trash_list.count(), 0)
                self.assertIn(str(captured["id"]), dialog.restored_ids)
                self.assertEqual(
                    window.controller.knowledge_item(str(captured["id"]))["aliases"],
                    ["ui-restore"],
                )
            finally:
                if dialog is not None:
                    dialog.close()
                window.close()
                application.processEvents()
