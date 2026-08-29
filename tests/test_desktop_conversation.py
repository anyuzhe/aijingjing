from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QImage
    from media_knowledge.desktop.app import SettingsDialog, create_application
    from media_knowledge.desktop.controller import DesktopController, ProviderConfigStore
    from media_knowledge.product import DesktopSettings
except (ImportError, RuntimeError):  # pragma: no cover - desktop extra is optional
    create_application = None


@unittest.skipIf(create_application is None, "PySide6 desktop components are unavailable")
class DesktopConversationTests(unittest.TestCase):
    def test_persisted_import_job_is_visible_and_resumable_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            controller = DesktopController(data, migrate_legacy=False)
            job = controller.create_ingestion_job(["one.md", "two.md"])
            application, window = create_application(data)
            try:
                window.refresh_ingestion_jobs()
                self.assertEqual(window.task_list.count(), 1)
                item = window.task_list.item(0)
                self.assertEqual(item.data(Qt.UserRole), job["id"])
                self.assertEqual(item.data(Qt.UserRole + 1), "queued")
                self.assertTrue(window.retry_button.isEnabled())
                self.assertEqual(window.retry_button.text(), "继续任务")
            finally:
                window.close()
                application.processEvents()

    def test_pasting_clipboard_image_adds_a_sendable_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(Path(temporary) / "data", migrate_legacy=False)
            application, window = create_application(controller.paths.root)
            try:
                image = QImage(120, 80, QImage.Format_RGB32)
                image.fill(QColor("#6cb6da"))
                application.clipboard().setImage(image)
                window.prompt.setFocus()
                window.prompt.paste()
                application.processEvents()
                self.assertEqual(len(window._pending_images), 1)
                self.assertEqual(window.attachment_list.count(), 1)
                self.assertFalse(window.attachment_list.isHidden())
                attachment = window._pending_images[0]
                self.assertTrue(Path(attachment.local_path).is_file())
                self.assertEqual((attachment.width, attachment.height), (120, 80))
                window.remove_selected_image()
                self.assertEqual(window._pending_images, [])
            finally:
                window.close()
                application.processEvents()

    def test_first_deepseek_key_keeps_vision_as_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(Path(temporary) / "data", migrate_legacy=False)
            application, window = create_application(controller.paths.root)
            dialog = SettingsDialog(controller, window)
            try:
                self.assertEqual(dialog.model.currentData(), "local-extractive")
                dialog.deepseek_key.setText("temporary-test-key")
                with patch.object(ProviderConfigStore, "_set_secret", return_value=False):
                    dialog.persist()
                self.assertEqual(
                    controller.settings.default_model,
                    "compatible::deepseek::deepseek-v4-flash-vision-exp",
                )
                self.assertIn(
                    "compatible::deepseek::deepseek-v4-flash-vision-exp",
                    {str(item["id"]) for item in controller.model_choices()},
                )
            finally:
                dialog.close()
                window.close()
                application.processEvents()

    def test_two_turns_restore_send_button_and_reuse_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            source = root / "knowledge.md"
            source.write_text(
                "# 连续对话\n\nAI静静支持连续对话，后续问题会沿用同一个对话上下文。",
                encoding="utf-8",
            )
            controller = DesktopController(data, migrate_legacy=False)
            controller.save_settings(
                DesktopSettings(
                    default_model="local-extractive",
                    embedding_provider="hash",
                    embedding_model="hash-384-v1",
                    auto_synthesize_notes=False,
                    enable_cloud_vision=False,
                )
            )
            self.assertEqual(controller.ingest([source]).succeeded, 1)
            application, window = create_application(data)
            window.show()
            application.processEvents()
            try:
                conversation_id = None
                answer_ids = []
                for question in ("它支持什么？", "第二轮会继续吗？"):
                    window.prompt.setPlainText(question)
                    window.send_button.click()
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline and window._answer_busy:
                        application.processEvents()
                        time.sleep(0.01)
                    application.processEvents()
                    self.assertFalse(window._answer_busy)
                    self.assertTrue(window.send_button.isEnabled())
                    self.assertEqual(window.send_button.text(), "发送 ↗")
                    self.assertIsNotNone(window.last_answer)
                    answer_ids.append(window.last_answer.answer_id)
                    if conversation_id is None:
                        conversation_id = window.conversation_id
                    self.assertEqual(window.conversation_id, conversation_id)
                self.assertEqual(len(set(answer_ids)), 2)
                self.assertIn("实际引用", window.answer_status.text())
                self.assertIn("份资料", window.answer_status.text())
                window.refresh_conversations()
                self.assertGreaterEqual(window.history_list.count(), 1)
                window.new_chat()
                self.assertIsNone(window.conversation_id)
                for index in range(window.history_list.count()):
                    item = window.history_list.item(index)
                    if item.data(Qt.UserRole) == conversation_id:
                        window.history_list.setCurrentItem(item)
                        break
                window.open_selected_conversation()
                self.assertEqual(window.conversation_id, conversation_id)
                self.assertEqual(len(window._chat_entries), 4)
                self.assertEqual(window.last_answer_id, answer_ids[-1])
                window.copy_last_answer()
                self.assertEqual(application.clipboard().text(), window.last_answer_markdown)
                window.save_answer_feedback("up")
                restored = controller.conversation_record(conversation_id)
                self.assertEqual(restored["answers"][-1]["feedback"]["rating"], "up")
            finally:
                window.close()
                application.processEvents()
