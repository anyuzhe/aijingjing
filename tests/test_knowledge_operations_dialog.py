from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication
    from media_knowledge.desktop.controller import DesktopController
    from media_knowledge.desktop.knowledge_operations_dialog import KnowledgeOperationsDialog
except (ImportError, RuntimeError):  # pragma: no cover
    QApplication = None
    DesktopController = None
    KnowledgeOperationsDialog = None


@unittest.skipIf(KnowledgeOperationsDialog is None, "PySide6 desktop components are unavailable")
class KnowledgeOperationsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.controller = DesktopController(
            Path(self.temporary.name) / "data", migrate_legacy=False
        )
        self.dialog = KnowledgeOperationsDialog(self.controller)
        self.application.processEvents()

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()
        self.application.processEvents()
        self.temporary.cleanup()

    def test_all_operational_surfaces_are_visible_and_accessible(self) -> None:
        self.assertEqual(self.dialog.tabs.count(), 5)
        self.assertEqual(
            [self.dialog.tabs.tabText(index) for index in range(5)],
            ["候选审核", "来源与冲突", "黄金评测", "SOP 流程库", "便携 Wiki"],
        )
        self.assertEqual(self.dialog.proposal_list.accessibleName(), "待审核知识候选列表")
        self.assertFalse(self.dialog.accept_proposal.isEnabled())
        self.assertIn("待审核 0 条", self.dialog.proposal_summary.text())

    def test_portable_wiki_can_compile_empty_database_with_navigation(self) -> None:
        self.dialog._compile_wiki()
        deadline = time.monotonic() + 3
        while "已编译" not in self.dialog.wiki_status.text() and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.assertIn("已编译 0 条知识", self.dialog.wiki_status.text())
        self.assertTrue(
            (self.controller.paths.notes / "LLM-Wiki" / "wiki" / "index.md").is_file()
        )


if __name__ == "__main__":
    unittest.main()
