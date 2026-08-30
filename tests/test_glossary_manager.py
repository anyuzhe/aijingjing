from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from media_knowledge.ingestion import IngestionResult, IngestionService
from media_knowledge.product import DesktopSettings, ProductPaths
from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.transcripts import TranscriptRepository

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox
    from media_knowledge.desktop.app import GlossaryManagerDialog, SettingsDialog
    from media_knowledge.desktop.controller import DesktopController
except (ImportError, RuntimeError):  # pragma: no cover - desktop extra is optional
    QApplication = None


class GlossarySettingsTests(unittest.TestCase):
    def test_knowledge_space_id_defaults_and_loads_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            self.assertEqual(DesktopSettings.load(path).asr_knowledge_space_id, "本地知识库")
            path.write_text(
                json.dumps({"asr_knowledge_space_id": "  岩土项目  "}),
                encoding="utf-8",
            )
            self.assertEqual(DesktopSettings.load(path).asr_knowledge_space_id, "岩土项目")
            path.write_text(
                json.dumps({"asr_knowledge_space_id": "   "}),
                encoding="utf-8",
            )
            self.assertEqual(DesktopSettings.load(path).asr_knowledge_space_id, "本地知识库")

    def test_ingestion_queries_configured_knowledge_space_without_network_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "meeting.txt"
            source.write_text("会议记录", encoding="utf-8")
            paths = ProductPaths.resolve(root / "data").ensure()
            settings = DesktopSettings(
                embedding_provider="hash",
                embedding_model="hash-384-v1",
                enable_cloud_vision=False,
                auto_synthesize_notes=False,
                asr_knowledge_space_id="岩土项目",
            )
            service = IngestionService(paths, settings=settings)
            captured_settings: list[DesktopSettings] = []

            def ingest_one(item: str, *_args: object, **kwargs: object) -> IngestionResult:
                captured_settings.append(kwargs["settings"])  # type: ignore[arg-type]
                return IngestionResult(item=item, status="succeeded")

            with (
                patch.object(socket, "create_connection") as create_connection,
                patch.object(urllib.request, "urlopen") as urlopen,
                patch.object(
                    TranscriptRepository,
                    "context_terms",
                    return_value=("围压",),
                ) as context_terms,
                patch.object(service, "_ingest_one", side_effect=ingest_one),
            ):
                summary = service.ingest([source])

            self.assertEqual(summary.succeeded, 1)
            context_terms.assert_called_once()
            self.assertEqual(
                context_terms.call_args.kwargs["knowledge_space_id"],
                "岩土项目",
            )
            self.assertTrue(context_terms.call_args.kwargs["source_id"].startswith("desktop-"))
            self.assertEqual(captured_settings[0].asr_context_terms, ["围压"])
            create_connection.assert_not_called()
            urlopen.assert_not_called()


@unittest.skipIf(QApplication is None, "PySide6 desktop components are unavailable")
class GlossaryManagerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_three_scope_crud_enable_and_delete_stay_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(Path(temporary) / "data", migrate_legacy=False)
            dialog = GlossaryManagerDialog(
                controller,
                knowledge_space_id="岩土项目",
            )
            try:
                with (
                    patch.object(socket, "create_connection") as create_connection,
                    patch.object(urllib.request, "urlopen") as urlopen,
                    patch.object(controller.local_models, "download") as download_model,
                ):
                    global_id = dialog.create_glossary("通用术语", "global")
                    space_id = dialog.create_glossary(
                        "岩体力学",
                        "knowledge_space",
                        "岩土项目",
                    )
                    source_id = dialog.create_glossary(
                        "项目会议",
                        "source",
                        "file:meeting",
                    )
                    term_id = dialog.add_term(space_id, "围压")

                    dialog.refresh(select_glossary_id=space_id)
                    dialog._toggle_glossary()
                    with KnowledgeDatabase(controller.paths.database) as database:
                        repository = TranscriptRepository(database)
                        self.assertFalse(repository.get_glossary(space_id).enabled)  # type: ignore[union-attr]
                        self.assertIsNotNone(repository.get_glossary(global_id))
                        self.assertIsNotNone(repository.get_glossary(source_id))

                    dialog.refresh(select_glossary_id=space_id)
                    self.assertEqual(dialog.term_list.count(), 1)
                    self.assertEqual(dialog.term_list.item(0).data(Qt.UserRole), term_id)
                    with patch.object(QMessageBox, "question", return_value=QMessageBox.Yes):
                        dialog._delete_term()
                        dialog._delete_glossary()

                    with KnowledgeDatabase(controller.paths.database) as database:
                        repository = TranscriptRepository(database)
                        self.assertIsNone(repository.get_glossary(space_id))
                        self.assertIsNone(repository.get_glossary_term(term_id))
                    create_connection.assert_not_called()
                    urlopen.assert_not_called()
                    download_model.assert_not_called()
            finally:
                dialog.close()
                self.application.processEvents()

    def test_settings_exposes_and_persists_knowledge_space_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller = DesktopController(Path(temporary) / "data", migrate_legacy=False)
            dialog = SettingsDialog(controller)
            try:
                dialog.asr_knowledge_space_id.setText("  工程知识空间  ")
                dialog.persist()
                self.assertEqual(controller.settings.asr_knowledge_space_id, "工程知识空间")
                self.assertEqual(
                    DesktopSettings.load(controller.paths.settings).asr_knowledge_space_id,
                    "工程知识空间",
                )
            finally:
                dialog.close()
                self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
