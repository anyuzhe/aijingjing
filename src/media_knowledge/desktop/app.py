from __future__ import annotations

import argparse
import html
import os
import sys
from dataclasses import replace
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Callable

try:
    from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPalette, QPixmap, QTextDocument
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QStatusBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QTextBrowser,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised by the launcher without desktop extras
    raise RuntimeError("桌面界面组件未安装，请安装项目的 desktop 依赖") from exc

from ..ingestion import CancellationToken, ProgressEvent
from ..product import DesktopSettings, PRODUCT_NAME
from .. import __version__
from .controller import DesktopController
from .diagnostics import run_diagnostics


APP_STYLE = """
QWidget { color: #294457; font-family: "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 14px; }
QMainWindow, QWidget#root {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
    stop:0 #e8f4fa, stop:0.48 #f5fafd, stop:1 #dcecf5);
}
QFrame#panel { background: #f9fcfe; border: 1px solid #c9dfea; border-radius: 14px; }
QFrame#brand {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    stop:0 #153b59, stop:0.56 #1d5578, stop:1 #286f91);
  border: 1px solid #73acc7;
  border-radius: 16px;
}
QFrame#brand QLabel { color: white; }
QLabel#muted { color: #718b9d; font-size: 12px; }
QLabel#section { color: #527187; font-size: 12px; font-weight: 600; }
QPushButton {
  background: #fbfdff;
  border: 1px solid #c7dce8;
  border-radius: 9px;
  padding: 8px 12px;
}
QPushButton:hover { background: #e8f5fb; border-color: #82b9d2; color: #174b6d; }
QPushButton:pressed { background: #d8ebf5; border-color: #6aa8c5; }
QPushButton:disabled { background: #edf3f6; color: #9babb5; border-color: #d8e4ea; }
QPushButton#primary {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2f82a8, stop:1 #1d587a);
  color: white;
  border: 1px solid #75b7d2;
  font-weight: 600;
}
QPushButton#primary:hover {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #45a0c3, stop:1 #286f95);
  border-color: #a6d7e8;
  color: white;
}
QPushButton#danger { color: #a95060; }
QLineEdit, QPlainTextEdit, QComboBox, QTextBrowser, QListWidget {
  background: #fcfeff;
  border: 1px solid #c8dce7;
  border-radius: 9px;
  padding: 7px;
  selection-background-color: #b9deed;
  selection-color: #183f59;
}
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QTextBrowser:focus, QListWidget:focus {
  border: 1px solid #70b3d0;
  background: #ffffff;
}
QListWidget::item { padding: 8px; border-radius: 8px; }
QListWidget::item:hover { background: #edf7fb; }
QListWidget::item:selected { background: #d8edf7; color: #174a6b; }
QTabWidget::pane { border: none; }
QTabBar::tab { padding: 8px 12px; color: #6e8798; }
QTabBar::tab:hover { color: #2b6f94; }
QTabBar::tab:selected { color: #185174; font-weight: 600; border-bottom: 2px solid #62adca; }
QProgressBar { border: none; background: #dfeaf0; border-radius: 4px; height: 7px; text-align: center; }
QProgressBar::chunk {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #80c8df, stop:1 #438eaf);
  border-radius: 4px;
}
QSplitter::handle { background: transparent; width: 5px; }
QStatusBar { background: #eff7fb; color: #678295; border-top: 1px solid #d3e5ee; }
QMenuBar { background: #edf6fa; color: #345469; }
QMenuBar::item:selected { background: #dceef6; border-radius: 5px; }
QMenu { background: #fbfdff; color: #29495d; border: 1px solid #c8dce8; }
QMenu::item:selected { background: #dceff7; color: #174a6b; }
QToolTip { background: #173f5b; color: #f5fcff; border: 1px solid #83bdd5; padding: 5px; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar:horizontal { background: transparent; height: 9px; margin: 2px; }
QScrollBar::handle { background: #b7d6e5; border-radius: 4px; min-height: 26px; min-width: 26px; }
QScrollBar::handle:hover { background: #7fb8d1; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
"""


def _markdown_html(markdown: str) -> str:
    document = QTextDocument()
    document.setMarkdown(markdown)
    return document.toHtml()


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(object)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, function: Callable[[WorkerSignals], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.result.emit(self.function(self.signals))
        except Exception as exc:  # worker boundary: the UI receives a safe, actionable message
            detail = str(exc).strip() or type(exc).__name__
            self.signals.error.emit(detail)
        finally:
            self.signals.finished.emit()


class PromptEdit(QPlainTextEdit):
    submit = Signal()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in {Qt.Key_Return, Qt.Key_Enter} and event.modifiers() & Qt.ControlModifier:
            self.submit.emit()
            return
        super().keyPressEvent(event)


class QualityCenterDialog(QDialog):
    def __init__(self, controller: DesktopController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("入库质检中心")
        self.resize(940, 590)
        layout = QVBoxLayout(self)
        intro = QLabel("每份资料必须通过真实性、正文完整性和来源范围检查；只抓到封面或说明的链接不会入库。")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["状态", "评分", "等级", "资料", "类型", "说明"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table, 1)
        rows = controller.quality_overview()
        self.table.setRowCount(len(rows))
        for row_index, report in enumerate(rows):
            checks = report.get("checks") or []
            failed = [str(item.get("detail", "")) for item in checks if item.get("status") == "fail"]
            warnings = [str(value) for value in report.get("warnings") or []]
            values = [
                "✓ 已通过" if report["accepted"] else "✕ 未通过",
                str(report["score"] or "—"), str(report["grade"]), str(report["title"]),
                str(report["media_type"]), "；".join(failed or warnings) or "完整性检查通过",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class SourceReaderDialog(QDialog):
    def __init__(
        self,
        controller: DesktopController,
        document: dict[str, object],
        *,
        chunk_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.document = document
        self.chunks = controller.document_chunks(str(document["id"]))
        self.annotations = controller.annotations(str(document["id"]))
        self.setWindowTitle(f"原文阅读 · {document['title']}")
        self.resize(1080, 720)
        layout = QVBoxLayout(self)
        info = QLabel(
            f"{document['media_type']} · {len(self.chunks)} 个知识块 · "
            f"{document.get('original_uri') or document.get('local_path') or '本地资料'}"
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        layout.addWidget(info)
        splitter = QSplitter(Qt.Horizontal)
        self.segments = QListWidget()
        self.segments.setMaximumWidth(340)
        for chunk in self.chunks:
            location = self._location(chunk)
            label = location or f"知识块 {int(chunk.get('ordinal', 0)) + 1}"
            item = QListWidgetItem(f"{label}\n{str(chunk.get('content', ''))[:70]}")
            item.setData(Qt.UserRole, chunk)
            self.segments.addItem(item)
            if chunk_id and chunk.get("id") == chunk_id:
                self.segments.setCurrentItem(item)
        self.preview = QTextBrowser()
        self.preview.setOpenExternalLinks(True)
        splitter.addWidget(self.segments)
        splitter.addWidget(self.preview)
        splitter.setSizes([300, 760])
        layout.addWidget(splitter, 1)
        row = QHBoxLayout()
        annotate = QPushButton("添加批注 / 制成卡片")
        annotate.clicked.connect(self._annotate)
        original = QPushButton("用系统播放器/原应用打开")
        original.clicked.connect(self._open_original)
        row.addWidget(annotate)
        row.addStretch()
        row.addWidget(original)
        layout.addLayout(row)
        self.segments.currentItemChanged.connect(self._show_chunk)
        if self.segments.currentItem() is None and self.segments.count():
            self.segments.setCurrentRow(0)

    @staticmethod
    def _location(chunk: dict[str, object]) -> str:
        values = []
        if chunk.get("page_number") is not None:
            values.append(f"第 {chunk['page_number']} 页")
        if chunk.get("slide_number") is not None:
            values.append(f"第 {chunk['slide_number']} 张")
        if chunk.get("timestamp_start") is not None:
            values.append(f"{float(chunk['timestamp_start']):g} 秒")
        if chunk.get("section"):
            values.append(str(chunk["section"]))
        return " · ".join(values)

    def _show_chunk(self, item: QListWidgetItem | None) -> None:
        if not item:
            return
        chunk = item.data(Qt.UserRole)
        if not isinstance(chunk, dict):
            return
        location = self._location(chunk)
        content = str(chunk.get("content") or "")
        related_notes = [
            str(note.get("content") or "") for note in self.annotations
            if note.get("chunk_id") == chunk.get("id")
        ]
        note_markdown = "\n\n---\n\n### 我的批注\n\n" + "\n\n".join(
            f"- {value}" for value in related_notes
        ) if related_notes else ""
        target = str(self.document.get("local_path") or "")
        image_path = str(chunk.get("image_path") or "")
        preview_path = image_path if image_path and Path(image_path).is_file() else target
        suffix = Path(preview_path).suffix.casefold()
        if suffix == ".pdf" and Path(preview_path).is_file():
            try:
                import pymupdf
                page_number = max(0, int(chunk.get("page_number") or 1) - 1)
                with pymupdf.open(preview_path) as pdf:
                    page_number = min(page_number, max(0, pdf.page_count - 1))
                    png = pdf[page_number].get_pixmap(matrix=pymupdf.Matrix(1.45, 1.45), alpha=False).tobytes("png")
                pixmap = QPixmap()
                pixmap.loadFromData(png)
                data_url = "data:image/png;base64," + __import__("base64").b64encode(png).decode("ascii")
                self.preview.setHtml(
                    f"<h3>{html.escape(location)}</h3><img src='{data_url}' style='max-width:100%'><hr>"
                    f"<pre style='white-space:pre-wrap'>{html.escape(content)}</pre>"
                )
                return
            except (OSError, RuntimeError, ValueError, ImportError):
                pass
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"} and Path(preview_path).is_file():
            self.preview.setHtml(
                f"<h3>{html.escape(location)}</h3><img src='{QUrl.fromLocalFile(preview_path).toString()}' style='max-width:100%'><hr>"
                f"<pre style='white-space:pre-wrap'>{html.escape(content)}</pre>"
            )
            return
        self.preview.setMarkdown(f"## {location or '原始知识块'}\n\n{content}{note_markdown}")

    def _annotate(self) -> None:
        item = self.segments.currentItem()
        if not item:
            return
        chunk = item.data(Qt.UserRole)
        value, accepted = QInputDialog.getMultiLineText(
            self, "添加批注", "写下理解、疑问或卡片答案："
        )
        if accepted and value.strip():
            self.controller.add_annotation(
                str(self.document["id"]), value,
                chunk_id=str(chunk.get("id")), locator={"label": self._location(chunk)},
            )
            self.annotations = self.controller.annotations(str(self.document["id"]))
            self._show_chunk(item)
            QMessageBox.information(self, "批注已保存", "批注已与当前原文位置绑定。")

    def _open_original(self) -> None:
        target = self.document.get("original_uri") or self.document.get("local_path")
        if not target:
            return
        value = str(target)
        QDesktopServices.openUrl(QUrl(value) if value.startswith(("http://", "https://")) else QUrl.fromLocalFile(value))


class SettingsDialog(QDialog):
    def __init__(self, controller: DesktopController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("设置 · AI静静")
        self.setMinimumWidth(590)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        general = QWidget()
        form = QFormLayout(general)
        self.data_root = QLineEdit(str(controller.paths.root))
        self.data_root.setReadOnly(True)
        form.addRow("知识数据目录", self.data_root)
        self.model = QComboBox()
        try:
            models = controller.model_choices()
        except ValueError:
            models = []
        for item in models:
            self.model.addItem(str(item["label"]), str(item["id"]))
            if item["id"] == controller.settings.default_model:
                self.model.setCurrentIndex(self.model.count() - 1)
        form.addRow("默认回答模型", self.model)
        self.archive = QCheckBox("将原始资料归档到应用目录")
        self.archive.setChecked(controller.settings.archive_originals)
        form.addRow("", self.archive)
        self.notes = QCheckBox("自动生成 Markdown Source Note")
        self.notes.setChecked(controller.settings.create_source_notes)
        form.addRow("", self.notes)
        self.synthesis = QCheckBox("入库时直连 DeepSeek 生成 AI 知识提炼（优先 Flash）")
        self.synthesis.setChecked(controller.settings.auto_synthesize_notes)
        form.addRow("", self.synthesis)
        self.vision = QCheckBox("使用 Kimi 进行高级视觉理解（可选；DeepSeek 暂不接收图片）")
        self.vision.setChecked(controller.settings.enable_cloud_vision)
        form.addRow("", self.vision)
        self.watched_enabled = QCheckBox("启用监听文件夹的后台增量同步")
        self.watched_enabled.setChecked(controller.settings.watched_folders_enabled)
        form.addRow("", self.watched_enabled)
        self.watched_minutes = QLineEdit(str(controller.settings.watched_scan_minutes))
        self.watched_minutes.setPlaceholderText("10")
        form.addRow("自动扫描间隔（分钟）", self.watched_minutes)
        self.update_url = QLineEdit(controller.settings.update_manifest_url or "")
        self.update_url.setPlaceholderText("由发布渠道提供的 HTTPS update.json，可留空")
        form.addRow("更新清单", self.update_url)
        tabs.addTab(general, "通用")

        providers = QWidget()
        provider_form = QFormLayout(providers)
        provider_status = {item["id"]: item for item in controller.providers.status()}
        self.deepseek_key = QLineEdit()
        self.deepseek_key.setEchoMode(QLineEdit.Password)
        self.deepseek_key.setPlaceholderText(
            "已配置，留空保留原密钥" if provider_status["deepseek"]["configured"] else "输入 DeepSeek API Key"
        )
        provider_form.addRow("DeepSeek API Key", self.deepseek_key)
        self.kimi_key = QLineEdit()
        self.kimi_key.setEchoMode(QLineEdit.Password)
        self.kimi_key.setPlaceholderText(
            "已配置，留空保留原密钥" if provider_status["kimi"]["configured"] else "输入 Kimi API Key"
        )
        provider_form.addRow("Kimi API Key", self.kimi_key)
        privacy = QLabel(
            "密钥优先保存在系统钥匙串；不支持钥匙串时才回退到仅当前用户可读的本地配置。\n"
            "AI 知识提炼直连 DeepSeek API，不会启动本地 Codex；全文检索和本地向量检索不会调用云模型。"
        )
        privacy.setWordWrap(True)
        privacy.setObjectName("muted")
        provider_form.addRow("", privacy)
        tabs.addTab(providers, "模型与隐私")

        obsidian = QWidget()
        obsidian_form = QFormLayout(obsidian)
        obsidian_row = QHBoxLayout()
        self.obsidian_path = QLineEdit(controller.settings.obsidian_vault or "")
        choose = QPushButton("选择…")
        choose.clicked.connect(self._choose_obsidian)
        obsidian_row.addWidget(self.obsidian_path, 1)
        obsidian_row.addWidget(choose)
        obsidian_form.addRow("Obsidian Vault（可选）", obsidian_row)
        hint = QLabel("Obsidian 不是必需依赖。配置后可导入 Vault 笔记，也可把 AI静静笔记导出到 Vault。")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        obsidian_form.addRow("", hint)
        tabs.addTab(obsidian, "Obsidian")

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_obsidian(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "选择 Obsidian Vault", self.obsidian_path.text())
        if value:
            self.obsidian_path.setText(value)

    def persist(self) -> None:
        if self.deepseek_key.text().strip():
            self.controller.providers.update("deepseek", api_key=self.deepseek_key.text())
        if self.kimi_key.text().strip():
            self.controller.providers.update("kimi", api_key=self.kimi_key.text())
        model = str(self.model.currentData() or self.controller.settings.default_model)
        settings = replace(
            self.controller.settings,
            default_model=model,
            archive_originals=self.archive.isChecked(),
            create_source_notes=self.notes.isChecked(),
            auto_synthesize_notes=self.synthesis.isChecked(),
            enable_cloud_vision=self.vision.isChecked(),
            watched_folders_enabled=self.watched_enabled.isChecked(),
            watched_scan_minutes=min(1440, max(1, int(self.watched_minutes.text() or "10"))),
            update_manifest_url=self.update_url.text().strip() or None,
            obsidian_vault=self.obsidian_path.text().strip() or None,
        )
        self.controller.save_settings(settings)


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self.import_token: CancellationToken | None = None
        self.import_items: list[str] = []
        self.conversation_id: str | None = None
        self.last_answer = None
        self.last_question = ""
        self.evidence_by_id: dict[str, object] = {}
        self._answer_busy = False
        self._answer_worker: Worker | None = None
        self._watch_scan_running = False
        self.setWindowTitle(PRODUCT_NAME)
        self.resize(1510, 920)
        self.setMinimumSize(1120, 700)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        self._load_models()
        self.refresh_library()
        self._refresh_status()
        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self._automatic_watch_scan)
        self._reset_sync_timer()
        QTimer.singleShot(2500, self._automatic_watch_scan)
        if controller.migrated_database:
            self.statusBar().showMessage(f"已迁移现有知识库：{controller.migrated_database}", 12000)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("文件")
        import_action = QAction("导入资料…", self)
        import_action.setShortcut(QKeySequence.Open)
        import_action.triggered.connect(self.choose_files)
        menu.addAction(import_action)
        url_action = QAction("导入网页或视频链接…", self)
        url_action.triggered.connect(self.add_url)
        menu.addAction(url_action)
        menu.addSeparator()
        open_data = QAction("打开知识数据目录", self)
        open_data.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.controller.paths.root))))
        menu.addAction(open_data)
        menu.addSeparator()
        quit_action = QAction("退出", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        menu.addAction(quit_action)

        knowledge_menu = self.menuBar().addMenu("知识库")
        refresh = QAction("刷新资料库", self)
        refresh.setShortcut(QKeySequence.Refresh)
        refresh.triggered.connect(self.refresh_library)
        knowledge_menu.addAction(refresh)
        sync = QAction("从 Obsidian 同步", self)
        sync.triggered.connect(self.sync_obsidian)
        knowledge_menu.addAction(sync)
        export = QAction("导出笔记到 Obsidian", self)
        export.triggered.connect(self.export_obsidian)
        knowledge_menu.addAction(export)
        knowledge_menu.addSeparator()
        quality = QAction("入库质检中心…", self)
        quality.triggered.connect(self.show_quality_center)
        knowledge_menu.addAction(quality)
        retrieval_lab = QAction("检索实验室…", self)
        retrieval_lab.triggered.connect(self.open_retrieval_lab)
        knowledge_menu.addAction(retrieval_lab)
        rebuild = QAction("重建中文语义索引…", self)
        rebuild.triggered.connect(self.rebuild_index)
        knowledge_menu.addAction(rebuild)
        duplicates = QAction("检查重复资料…", self)
        duplicates.triggered.connect(self.show_duplicates)
        knowledge_menu.addAction(duplicates)

        automation_menu = self.menuBar().addMenu("自动化")
        watch_add = QAction("添加监听文件夹…", self)
        watch_add.triggered.connect(self.add_watched_folder)
        automation_menu.addAction(watch_add)
        watch_scan = QAction("立即扫描监听文件夹", self)
        watch_scan.triggered.connect(self.scan_watched_folders)
        automation_menu.addAction(watch_scan)
        watch_manage = QAction("管理监听文件夹…", self)
        watch_manage.triggered.connect(self.manage_watched_folders)
        automation_menu.addAction(watch_manage)
        packages = QAction("Source Package 管理器…", self)
        packages.triggered.connect(self.show_source_packages)
        automation_menu.addAction(packages)

        workshop_menu = self.menuBar().addMenu("知识工坊")
        for label, kind in (
            ("综合报告", "report"), ("多资料比较", "compare"), ("时间线", "timeline"),
            ("测验题", "quiz"), ("复习闪卡", "flashcards"), ("思维导图", "mindmap"),
        ):
            action = QAction(label, self)
            action.triggered.connect(lambda _checked=False, value=kind: self.run_workshop(value))
            workshop_menu.addAction(action)

        safety_menu = self.menuBar().addMenu("数据安全")
        backup = QAction("立即创建完整备份", self)
        backup.triggered.connect(self.create_backup)
        safety_menu.addAction(backup)
        restore = QAction("从备份恢复…", self)
        restore.triggered.connect(self.restore_backup)
        safety_menu.addAction(restore)
        repair = QAction("数据库检查与修复…", self)
        repair.triggered.connect(self.repair_database)
        safety_menu.addAction(repair)

        settings = QAction("设置…", self)
        settings.setShortcut(QKeySequence.Preferences)
        settings.triggered.connect(self.open_settings)
        self.menuBar().addAction(settings)

        help_menu = self.menuBar().addMenu("帮助")
        updates = QAction("检查更新…", self)
        updates.triggered.connect(self.check_updates)
        help_menu.addAction(updates)
        diagnostics = QAction("系统诊断", self)
        diagnostics.triggered.connect(self.show_diagnostics)
        help_menu.addAction(diagnostics)
        about = QAction("关于 AI静静", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 8)
        root_layout.setSpacing(10)

        top = QFrame()
        top.setObjectName("brand")
        top_layout = QHBoxLayout(top)
        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(46, 46)
        try:
            logo_path = files("media_knowledge.desktop").joinpath("assets/ai_jingjing_mascot.png")
            logo.setPixmap(QIcon(str(logo_path)).pixmap(QSize(44, 44)))
        except (FileNotFoundError, TypeError):
            logo.setText("静")
            logo.setStyleSheet(
                "background:#d9f2fb;color:#164a70;border-radius:12px;font-size:21px;font-weight:800;"
            )
        titles = QVBoxLayout()
        name = QLabel(PRODUCT_NAME)
        name.setStyleSheet("font-size:20px;font-weight:700;")
        subtitle = QLabel("✦ 本地优先 · 多模态入库 · 可溯源问答")
        subtitle.setStyleSheet("color:#d7eff9;font-size:12px;")
        titles.addWidget(name)
        titles.addWidget(subtitle)
        top_layout.addWidget(logo)
        top_layout.addLayout(titles)
        top_layout.addStretch()
        self.global_search = QLineEdit()
        self.global_search.setPlaceholderText("搜索全部知识（不调用大模型）")
        self.global_search.setMinimumWidth(360)
        self.global_search.returnPressed.connect(self.run_search)
        top_layout.addWidget(self.global_search)
        search_button = QPushButton("搜索")
        search_button.clicked.connect(self.run_search)
        top_layout.addWidget(search_button)
        root_layout.addWidget(top)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._left_panel())
        splitter.addWidget(self._center_panel())
        splitter.addWidget(self._right_panel())
        splitter.setSizes([295, 840, 330])
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

    def _panel(self) -> tuple[QFrame, QVBoxLayout]:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        return panel, layout

    def _left_panel(self) -> QWidget:
        panel, layout = self._panel()
        actions = QHBoxLayout()
        import_button = QPushButton("＋ 导入资料")
        import_button.setObjectName("primary")
        import_button.clicked.connect(self.choose_files)
        url_button = QPushButton("链接")
        url_button.clicked.connect(self.add_url)
        actions.addWidget(import_button, 1)
        actions.addWidget(url_button)
        layout.addLayout(actions)
        hint = QLabel("可一次选择多个文件，也可拖入网页或音视频链接")
        hint.setObjectName("muted")
        layout.addWidget(hint)

        self.left_tabs = QTabWidget()
        library_tab = QWidget()
        library_layout = QVBoxLayout(library_tab)
        library_layout.setContentsMargins(0, 8, 0, 0)
        self.library_filter = QLineEdit()
        self.library_filter.setPlaceholderText("按标题筛选…")
        self.library_filter.textChanged.connect(self._filter_library)
        library_layout.addWidget(self.library_filter)
        self.collection_filter = QComboBox()
        self.collection_filter.addItem("全部知识空间", "")
        self.collection_filter.currentIndexChanged.connect(self._filter_library)
        library_layout.addWidget(self.collection_filter)
        self.scope_selected = QCheckBox("仅使用选中的资料进行搜索与问答")
        library_layout.addWidget(self.scope_selected)
        self.document_list = QListWidget()
        self.document_list.setWordWrap(True)
        self.document_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.document_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.document_list.currentItemChanged.connect(self._show_document)
        self.document_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.document_list.customContextMenuRequested.connect(self._document_menu)
        library_layout.addWidget(self.document_list, 1)
        self.left_tabs.addTab(library_tab, "资料库")

        task_tab = QWidget()
        task_layout = QVBoxLayout(task_tab)
        task_layout.setContentsMargins(0, 8, 0, 0)
        self.task_list = QListWidget()
        task_layout.addWidget(self.task_list, 1)
        controls = QHBoxLayout()
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_import)
        self.retry_button = QPushButton("重试失败项")
        self.retry_button.clicked.connect(self.retry_failed)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.cancel_button)
        task_layout.addLayout(controls)
        task_layout.addWidget(self.retry_button)
        self.left_tabs.addTab(task_tab, "任务")
        layout.addWidget(self.left_tabs, 1)
        return panel

    def _center_panel(self) -> QWidget:
        panel, layout = self._panel()
        toolbar = QHBoxLayout()
        new_chat = QPushButton("＋ 新对话")
        new_chat.clicked.connect(self.new_chat)
        toolbar.addWidget(new_chat)
        toolbar.addStretch()
        self.deep_analysis = QCheckBox("深度分析")
        toolbar.addWidget(self.deep_analysis)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(245)
        self.model_combo.currentIndexChanged.connect(self._save_default_model)
        toolbar.addWidget(self.model_combo)
        settings = QPushButton("设置")
        settings.clicked.connect(self.open_settings)
        toolbar.addWidget(settings)
        layout.addLayout(toolbar)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setHtml(self._welcome_html())
        self.chat.anchorClicked.connect(self._open_link)
        layout.addWidget(self.chat, 1)

        compose = QFrame()
        compose.setStyleSheet("QFrame{background:#fbfeff;border:1px solid #c5dce8;border-radius:12px;}")
        compose_layout = QVBoxLayout(compose)
        self.prompt = PromptEdit()
        self.prompt.setPlaceholderText("向你的知识库提问……（Ctrl+Enter 发送）")
        self.prompt.setFixedHeight(92)
        self.prompt.setStyleSheet("border:none;background:transparent;")
        self.prompt.submit.connect(self.ask)
        compose_layout.addWidget(self.prompt)
        row = QHBoxLayout()
        self.answer_status = QLabel("❄ 就绪")
        self.answer_status.setObjectName("muted")
        row.addWidget(self.answer_status)
        row.addStretch()
        save = QPushButton("保存笔记")
        save.clicked.connect(self.save_answer)
        row.addWidget(save)
        self.send_button = QPushButton("发送 ↗")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self.ask)
        row.addWidget(self.send_button)
        compose_layout.addLayout(row)
        layout.addWidget(compose)
        return panel

    def _right_panel(self) -> QWidget:
        panel, layout = self._panel()
        title = QLabel("证据与资料详情")
        title.setStyleSheet("font-size:15px;font-weight:700;")
        layout.addWidget(title)
        self.evidence_list = QListWidget()
        self.evidence_list.setMaximumHeight(275)
        self.evidence_list.currentItemChanged.connect(self._show_evidence)
        self.evidence_list.itemDoubleClicked.connect(lambda _item: self.open_source_reader())
        layout.addWidget(self.evidence_list)
        self.preview = QTextBrowser()
        self.preview.setOpenExternalLinks(True)
        self.preview.setPlaceholderText("点击资料或回答引用，在这里查看原始证据。")
        layout.addWidget(self.preview, 1)
        open_button = QPushButton("在 AI静静中查看原文与定位")
        open_button.clicked.connect(self.open_source_reader)
        layout.addWidget(open_button)
        return panel

    def _welcome_html(self) -> str:
        return """
        <div style='margin:44px auto;max-width:680px;color:#355569'>
          <h2 style='color:#174f72'>你好，我是 AI静静 ❄️</h2>
          <p>我只根据你导入的知识回答，并保留页码、幻灯片页、时间轴和原文路径。</p>
          <div style='background:#eaf6fb;border:1px solid #c9e3ef;border-radius:12px;padding:18px;margin-top:20px'>
            <b>可以这样开始</b><br><br>
            • 把 PDF、PPTX、Word、图片、音视频或 Markdown 拖入窗口<br>
            • 搜索某个概念，无需调用大模型<br>
            • 让我综合多份资料并给出可溯源回答
          </div>
        </div>"""

    def _load_models(self) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        try:
            choices = self.controller.model_choices()
        except ValueError as exc:
            self.statusBar().showMessage(f"模型配置错误：{exc}", 10000)
            choices = []
        for item in choices:
            self.model_combo.addItem(str(item["label"]), str(item["id"]))
            self.model_combo.setItemData(self.model_combo.count() - 1, str(item["description"]), Qt.ToolTipRole)
            if item["id"] == self.controller.settings.default_model:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.blockSignals(False)

    def _save_default_model(self) -> None:
        model = self.model_combo.currentData()
        if model and model != self.controller.settings.default_model:
            self.controller.save_settings(replace(self.controller.settings, default_model=str(model)))

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择要导入的资料",
            str(Path.home()),
            "所有支持的资料 (*.md *.markdown *.txt *.csv *.json *.yaml *.yml *.pdf *.docx *.pptx *.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif *.mp3 *.m4a *.wav *.aac *.flac *.ogg *.opus *.mp4 *.mov *.mkv *.avi *.webm *.m4v);;所有文件 (*)",
        )
        if paths:
            self.start_import(paths)

    def add_url(self) -> None:
        value, accepted = QInputDialog.getText(
            self,
            "导入网页或视频链接",
            "粘贴公开网页、微信视频号或音视频直链：",
            text="https://",
        )
        if accepted and value.strip():
            self.start_import([value.strip()])

    def start_import(self, items: list[str]) -> None:
        if self.import_token is not None:
            QMessageBox.information(self, "导入进行中", "请等待当前批次完成，或取消后再导入。")
            return
        self.import_items = list(items)
        self.import_token = CancellationToken()
        self.task_list.clear()
        for value in items:
            item = QListWidgetItem(f"等待处理  ·  {Path(value).name or value}")
            item.setData(Qt.UserRole, value)
            item.setData(Qt.UserRole + 1, "pending")
            self.task_list.addItem(item)
        self.left_tabs.setCurrentIndex(1)
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        token = self.import_token

        def execute(signals: WorkerSignals):
            return self.controller.ingest(
                items,
                progress=lambda event: signals.progress.emit(event),
                cancellation=token,
            )

        worker = Worker(execute)
        worker.signals.progress.connect(self._import_progress)
        worker.signals.result.connect(self._import_complete)
        worker.signals.error.connect(lambda message: self._operation_error("导入失败", message))
        worker.signals.finished.connect(self._import_finished)
        self.thread_pool.start(worker)
        self.statusBar().showMessage(f"正在后台导入 {len(items)} 份资料…")

    def _import_progress(self, event: ProgressEvent) -> None:
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            if item.data(Qt.UserRole) == event.item:
                item.setText(f"{event.percent:>3}%  ·  {event.message}\n{Path(event.item).name or event.item}")
                item.setData(Qt.UserRole + 1, event.stage)
                break
        self.statusBar().showMessage(event.message)

    def _import_complete(self, summary) -> None:
        failed = []
        for result in summary.results:
            if result.status == "failed":
                failed.append(result.item)
            for index in range(self.task_list.count()):
                item = self.task_list.item(index)
                if item.data(Qt.UserRole) == result.item:
                    icon = "✓" if result.status not in {"failed", "cancelled"} else "✕"
                    quality = result.quality_report or {}
                    quality_label = (
                        f" · 质检 {quality.get('score')} 分/{quality.get('grade')}"
                        if quality.get("score") is not None else ""
                    )
                    item.setText(
                        f"{icon} {result.title or Path(result.item).name}\n"
                        f"{result.status} · {result.chunks} 个知识块{quality_label}"
                    )
                    item.setData(Qt.UserRole + 1, result.status)
                    checks = [
                        f"{check.get('name')}：{check.get('detail')}"
                        for check in quality.get("checks", [])
                    ]
                    item.setToolTip(result.error or "\n".join([*result.warnings, *checks]))
        self.statusBar().showMessage(
            f"入库完成：{summary.succeeded}/{summary.total} 成功" + (f"，{summary.failed} 失败" if summary.failed else ""),
            12000,
        )
        self.refresh_library()
        self._refresh_status()

    def _import_finished(self) -> None:
        self.import_token = None
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.cancel_button.setEnabled(False)

    def toggle_pause(self) -> None:
        if not self.import_token:
            return
        if self.import_token.paused:
            self.import_token.resume()
            self.pause_button.setText("暂停")
            self.statusBar().showMessage("已继续导入")
        else:
            self.import_token.pause()
            self.pause_button.setText("继续")
            self.statusBar().showMessage("已暂停；当前解析步骤完成后生效")

    def cancel_import(self) -> None:
        if self.import_token:
            self.import_token.cancel()
            self.statusBar().showMessage("正在安全取消导入…")

    def retry_failed(self) -> None:
        failed = [
            str(self.task_list.item(i).data(Qt.UserRole))
            for i in range(self.task_list.count())
            if self.task_list.item(i).data(Qt.UserRole + 1) == "failed"
        ]
        if failed:
            self.start_import(failed)
        else:
            self.statusBar().showMessage("没有需要重试的项目", 4000)

    def refresh_library(self) -> None:
        selected = self.document_list.currentItem().data(Qt.UserRole) if self.document_list.currentItem() else None
        self.document_list.clear()
        try:
            documents = self.controller.documents()
        except Exception as exc:
            self._operation_error("无法读取资料库", str(exc))
            return
        current_collection = str(self.collection_filter.currentData() or "")
        collections = sorted({
            str(collection)
            for document in documents
            for collection in document.get("collections", [])
        })
        self.collection_filter.blockSignals(True)
        self.collection_filter.clear()
        self.collection_filter.addItem("全部知识空间", "")
        for collection in collections:
            self.collection_filter.addItem(collection, collection)
            if collection == current_collection:
                self.collection_filter.setCurrentIndex(self.collection_filter.count() - 1)
        self.collection_filter.blockSignals(False)
        for document in documents:
            enabled = "" if document.get("enabled", True) else "（已停用）"
            item = QListWidgetItem(
                f"{self._media_icon(str(document['media_type']))}  {document['title']}{enabled}\n"
                f"     {document['media_type']} · {document['chunks']} 个知识块"
            )
            item.setData(Qt.UserRole, document)
            item.setToolTip(str(document.get("local_path") or document.get("original_uri") or ""))
            self.document_list.addItem(item)
            if selected and isinstance(selected, dict) and selected.get("id") == document["id"]:
                self.document_list.setCurrentItem(item)
        self._filter_library(self.library_filter.text())

    @staticmethod
    def _media_icon(media_type: str) -> str:
        return {"pdf": "PDF", "presentation": "PPT", "image": "IMG", "audio": "AUD", "video": "VID", "web": "WEB", "markdown": "MD"}.get(media_type, "DOC")

    def _filter_library(self, _value: object = None) -> None:
        term = self.library_filter.text().strip().casefold()
        collection = str(self.collection_filter.currentData() or "")
        for index in range(self.document_list.count()):
            item = self.document_list.item(index)
            document = item.data(Qt.UserRole)
            wrong_collection = bool(
                collection and isinstance(document, dict) and collection not in document.get("collections", [])
            )
            item.setHidden(bool((term and term not in item.text().casefold()) or wrong_collection))

    def _show_document(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        document = item.data(Qt.UserRole)
        if not isinstance(document, dict):
            return
        self.preview.setMarkdown(
            f"# {document['title']}\n\n"
            f"- 类型：`{document['media_type']}`\n"
            f"- 知识块：{document['chunks']}\n"
            f"- 知识空间：{', '.join(document.get('collections', [])) or '未分组'}\n"
            f"- 标签：{', '.join(document.get('tags', [])) or '无'}\n"
            f"- 状态：{'参与检索' if document.get('enabled', True) else '已停用'}\n"
            f"- 更新时间：{document['updated_at']}\n"
            f"- 原始位置：{document.get('original_uri') or document.get('local_path') or '未记录'}\n\n"
            "可在中间输入问题，或用顶部搜索框查找原始证据。"
        )

    def _document_menu(self, position) -> None:
        item = self.document_list.itemAt(position)
        if not item:
            return
        menu = QMenu(self)
        read_action = menu.addAction("在 AI静静中阅读原文")
        open_action = menu.addAction("用原应用打开")
        chunks_action = menu.addAction("查看解析知识块")
        menu.addSeparator()
        rename_action = menu.addAction("重命名…")
        facets_action = menu.addAction("设置知识空间和标签…")
        document = item.data(Qt.UserRole)
        enabled = bool(document.get("enabled", True)) if isinstance(document, dict) else True
        enable_action = menu.addAction("停用（不参与搜索）" if enabled else "重新启用")
        reingest_action = menu.addAction("重新解析")
        menu.addSeparator()
        delete_action = menu.addAction("从知识库移除…")
        chosen = menu.exec(self.document_list.mapToGlobal(position))
        self.document_list.setCurrentItem(item)
        if chosen == read_action or chosen == chunks_action:
            self.open_source_reader()
        elif chosen == open_action:
            self.document_list.setCurrentItem(item)
            self.open_selected_source()
        elif chosen == rename_action:
            self.rename_selected_document()
        elif chosen == facets_action:
            self.edit_selected_facets()
        elif chosen == enable_action and isinstance(document, dict):
            self.controller.set_document_enabled(str(document["id"]), not enabled)
            self.refresh_library()
        elif chosen == reingest_action:
            self.reingest_selected_document()
        elif chosen == delete_action:
            self.delete_selected_document()

    def run_search(self) -> None:
        query = self.global_search.text().strip()
        if not query:
            return
        self.evidence_list.clear()
        self.preview.setMarkdown("正在执行本地混合检索……")
        document_ids, collections = self._active_scope()

        def execute(_signals: WorkerSignals):
            return self.controller.search(
                query, top_k=20, document_ids=document_ids, collections=collections
            )

        worker = Worker(execute)
        worker.signals.result.connect(self._search_complete)
        worker.signals.error.connect(lambda message: self._operation_error("搜索失败", message))
        self.thread_pool.start(worker)

    def _search_complete(self, results) -> None:
        self.evidence_by_id.clear()
        self.evidence_list.clear()
        for index, result in enumerate(results, 1):
            key = f"search-{index}"
            self.evidence_by_id[key] = result
            location = self._result_location(result)
            vector = result.debug.get("vector_score")
            keyword = result.debug.get("keyword_score")
            detail = f"融合 {result.score:.4f}"
            if vector is not None:
                detail += f" · 语义 {float(vector):.3f}"
            if keyword is not None:
                detail += " · 全文命中"
            item = QListWidgetItem(
                f"{index}. {result.title}\n{location or result.source.media_type} · {detail}"
            )
            item.setData(Qt.UserRole, key)
            self.evidence_list.addItem(item)
        if results:
            self.evidence_list.setCurrentRow(0)
        else:
            self.preview.setMarkdown("没有找到匹配的知识。")
        self.statusBar().showMessage(f"本地搜索完成：{len(results)} 条证据，未调用大模型", 8000)

    def ask(self) -> None:
        if self._answer_busy:
            self.statusBar().showMessage("上一条回答仍在生成，请稍候；当前输入不会丢失。", 5000)
            return
        question = self.prompt.toPlainText().strip()
        if not question:
            return
        self.last_question = question
        self.prompt.clear()
        self._append_user(question)
        self._set_answer_busy(True)
        self.answer_status.setText("正在检索选定知识…")
        model_id = str(self.model_combo.currentData() or "local-extractive")
        conversation = self.conversation_id
        deep = self.deep_analysis.isChecked()
        document_ids, collections = self._active_scope()

        def execute(signals: WorkerSignals):
            return self.controller.ask(
                question,
                conversation_id=conversation,
                model_id=model_id,
                deep_analysis=deep,
                document_ids=document_ids,
                collections=collections,
                progress=lambda stage, message: signals.progress.emit((stage, message)),
            )

        worker = Worker(execute)
        worker.signals.progress.connect(lambda value: self.answer_status.setText(value[1]))
        worker.signals.result.connect(self._answer_complete)
        worker.signals.error.connect(lambda message: self._answer_error(message))
        worker.signals.finished.connect(self._finish_answer_request)
        self._answer_worker = worker
        self.thread_pool.start(worker)

    def _set_answer_busy(self, busy: bool) -> None:
        self._answer_busy = busy
        self.send_button.setEnabled(not busy)
        self.send_button.setText("生成中…" if busy else "发送 ↗")

    def _finish_answer_request(self) -> None:
        """Restore the composer from deterministic result/error callbacks and as a fallback."""
        self._set_answer_busy(False)
        self._answer_worker = None
        self.prompt.setFocus(Qt.OtherFocusReason)

    def _selected_documents(self) -> list[dict[str, object]]:
        values = []
        for item in self.document_list.selectedItems():
            document = item.data(Qt.UserRole)
            if isinstance(document, dict):
                values.append(document)
        return values

    def _active_scope(self) -> tuple[list[str] | None, list[str] | None]:
        if self.scope_selected.isChecked():
            selected = self._selected_documents()
            if not selected and self.document_list.currentItem():
                value = self.document_list.currentItem().data(Qt.UserRole)
                selected = [value] if isinstance(value, dict) else []
            return [str(item["id"]) for item in selected] or None, None
        collection = str(self.collection_filter.currentData() or "")
        return None, [collection] if collection else None

    def _append_user(self, text: str) -> None:
        self.chat.append(
            f"<div style='margin:18px 8px 6px 22%;text-align:right;color:#6a8495;font-size:11px'>你</div>"
            f"<div style='margin:0 8px 18px 22%;background:#1b5578;color:white;padding:12px 15px;border-radius:14px'>"
            f"{html.escape(text).replace(chr(10), '<br>')}</div>"
        )
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    def _answer_complete(self, answer) -> None:
        self._finish_answer_request()
        self.last_answer = answer
        self.conversation_id = answer.conversation_id
        rendered = _markdown_html(answer.markdown)
        for evidence in answer.evidence:
            marker = f"[{evidence.evidence_id}]"
            rendered = rendered.replace(
                marker, f"<a href='aijj://citation/{evidence.evidence_id}'>{marker}</a>"
            )
        self.chat.append(
            "<div style='margin:8px 20% 6px 8px;color:#2d789e;font-size:11px;font-weight:600'>✦ AI静静</div>"
            f"<div style='margin:0 20% 20px 8px;background:#fbfeff;border:1px solid #c7e0ec;padding:14px;border-radius:14px'>{rendered}</div>"
        )
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())
        self.answer_status.setText(f"{answer.model} · {answer.confidence:.0%} 有据可查")
        self.evidence_by_id.clear()
        self.evidence_list.clear()
        for evidence in answer.evidence:
            self.evidence_by_id[evidence.evidence_id] = evidence
            item = QListWidgetItem(f"[{evidence.evidence_id}] {evidence.title}\n{evidence.locator() or evidence.source.media_type}")
            item.setData(Qt.UserRole, evidence.evidence_id)
            self.evidence_list.addItem(item)
        if self.evidence_list.count():
            self.evidence_list.setCurrentRow(0)
        self._refresh_status()

    def _answer_error(self, message: str) -> None:
        self._finish_answer_request()
        self.answer_status.setText("回答失败")
        self.chat.append(
            f"<div style='margin:8px 20% 20px 8px;background:#fff3f5;color:#934552;border:1px solid #efd3da;padding:14px;border-radius:12px'>"
            f"<b>未能生成回答</b><br>{html.escape(message)}</div>"
        )
        self._operation_error("回答失败", message)

    def _show_evidence(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        value = self.evidence_by_id.get(str(item.data(Qt.UserRole)))
        if value is None:
            return
        if hasattr(value, "content"):
            title = getattr(value, "title", "证据")
            content = getattr(value, "content", "")
            source = getattr(value, "source", None)
            location = value.locator() if hasattr(value, "locator") else self._result_location(value)
            target = getattr(source, "original_uri", None) or getattr(source, "local_path", None) or ""
            self.preview.setMarkdown(f"# {title}\n\n**{location}**\n\n{content}\n\n---\n\n来源：{target}")

    @staticmethod
    def _result_location(result) -> str:
        values = []
        if getattr(result, "page", None) is not None:
            values.append(f"P{result.page}")
        if getattr(result, "slide", None) is not None:
            values.append(f"S{result.slide}")
        if getattr(result, "timestamp_start", None) is not None:
            values.append(f"{result.timestamp_start:g}s")
        return " / ".join(values)

    def open_selected_source(self) -> None:
        target = None
        evidence_item = self.evidence_list.currentItem()
        if evidence_item:
            value = self.evidence_by_id.get(str(evidence_item.data(Qt.UserRole)))
            source = getattr(value, "source", None)
            target = getattr(source, "original_uri", None) or getattr(source, "local_path", None)
        if not target and self.document_list.currentItem():
            document = self.document_list.currentItem().data(Qt.UserRole)
            if isinstance(document, dict):
                target = document.get("original_uri") or document.get("local_path")
        if not target:
            self.statusBar().showMessage("当前项目没有可打开的原始位置", 5000)
            return
        if str(target).startswith(("http://", "https://")):
            QDesktopServices.openUrl(QUrl(str(target)))
        else:
            path = Path(str(target))
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                self.statusBar().showMessage("原始文件已移动或不存在", 6000)

    def save_answer(self) -> None:
        if not self.last_answer:
            self.statusBar().showMessage("当前没有可保存的回答", 5000)
            return
        try:
            path = self.controller.save_answer_note(self.last_answer, self.last_question)
            self.statusBar().showMessage(f"已保存到：{path}", 10000)
        except OSError as exc:
            self._operation_error("保存失败", str(exc))

    def new_chat(self) -> None:
        self.conversation_id = None
        self.last_answer = None
        self.last_question = ""
        self.chat.setHtml(self._welcome_html())
        self.evidence_list.clear()
        self.evidence_by_id.clear()
        self.preview.clear()
        self.answer_status.setText("❄ 就绪")

    def open_source_reader(self) -> None:
        document_id = None
        chunk_id = None
        evidence_item = self.evidence_list.currentItem()
        if evidence_item:
            value = self.evidence_by_id.get(str(evidence_item.data(Qt.UserRole)))
            document_id = getattr(value, "document_id", None)
            chunk_id = getattr(value, "chunk_id", None)
            source = getattr(value, "source", None)
            document_id = document_id or getattr(source, "document_id", None)
            chunk_id = chunk_id or getattr(source, "chunk_id", None)
        if not document_id and self.document_list.currentItem():
            current = self.document_list.currentItem().data(Qt.UserRole)
            if isinstance(current, dict):
                document_id = current.get("id")
        document = next(
            (value for value in self.controller.documents(limit=2000) if value["id"] == document_id),
            None,
        )
        if not document:
            self.statusBar().showMessage("请先选择一份资料或一条证据", 5000)
            return
        SourceReaderDialog(
            self.controller, document, chunk_id=str(chunk_id) if chunk_id else None, parent=self
        ).exec()

    def rename_selected_document(self) -> None:
        item = self.document_list.currentItem()
        document = item.data(Qt.UserRole) if item else None
        if not isinstance(document, dict):
            return
        title, accepted = QInputDialog.getText(
            self, "重命名资料", "新标题：", text=str(document["title"])
        )
        if accepted and title.strip():
            self.controller.rename_document(str(document["id"]), title)
            self.refresh_library()

    def edit_selected_facets(self) -> None:
        item = self.document_list.currentItem()
        document = item.data(Qt.UserRole) if item else None
        if not isinstance(document, dict):
            return
        spaces, accepted = QInputDialog.getText(
            self, "知识空间", "空间名称（多个用逗号分隔）：",
            text=", ".join(document.get("collections", [])),
        )
        if not accepted:
            return
        tags, accepted = QInputDialog.getText(
            self, "资料标签", "标签（多个用逗号分隔）：",
            text=", ".join(document.get("tags", [])),
        )
        if accepted:
            split = lambda value: [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
            self.controller.update_document_facets(
                str(document["id"]), collections=split(spaces), tags=split(tags)
            )
            self.refresh_library()

    def reingest_selected_document(self) -> None:
        item = self.document_list.currentItem()
        document = item.data(Qt.UserRole) if item else None
        if not isinstance(document, dict):
            return
        self._run_simple_background(
            "正在重新解析资料…",
            lambda: self.controller.reingest_document(str(document["id"])),
            self._import_complete,
        )

    def delete_selected_document(self) -> None:
        item = self.document_list.currentItem()
        document = item.data(Qt.UserRole) if item else None
        if not isinstance(document, dict):
            return
        choice = QMessageBox.question(
            self, "从知识库移除",
            f"移除“{document['title']}”的检索记录？\n\n原始归档仍会保留，可重新导入恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if choice == QMessageBox.Yes:
            self.controller.delete_document(str(document["id"]))
            self.refresh_library()
            self._refresh_status()

    def show_quality_center(self) -> None:
        QualityCenterDialog(self.controller, self).exec()

    def open_retrieval_lab(self) -> None:
        query, accepted = QInputDialog.getText(
            self, "检索实验室", "输入测试查询（将显示向量、全文和重排融合结果）："
        )
        if accepted and query.strip():
            self.global_search.setText(query.strip())
            self.run_search()

    def rebuild_index(self) -> None:
        self._run_simple_background(
            "正在下载/加载中文语义模型并重建索引，首次运行可能需要几分钟…",
            self.controller.rebuild_search_index,
            lambda report: self._sync_complete({"语义索引": report}),
        )

    def show_duplicates(self) -> None:
        groups = self.controller.duplicate_groups()
        if not groups:
            QMessageBox.information(self, "重复资料检查", "没有发现内容指纹相同的重复资料。")
            return
        lines = []
        for index, group in enumerate(groups, 1):
            lines.append(f"{index}. {group['document_count']} 份：{' / '.join(group['titles'])}")
        QMessageBox.information(
            self, "重复资料检查", "发现以下重复组。系统检索时会保留来源边界：\n\n" + "\n".join(lines)
        )

    def add_watched_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择要自动同步的文件夹", str(Path.home()))
        if not path:
            return
        collection, accepted = QInputDialog.getText(
            self, "同步到知识空间", "知识空间名称：", text=Path(path).name or "自动同步"
        )
        if accepted:
            self.controller.add_watched_folder(path, collection=collection or "自动同步")
            self.scan_watched_folders()

    def scan_watched_folders(self) -> None:
        if self._watch_scan_running:
            self.statusBar().showMessage("监听文件夹扫描已在进行", 4000)
            return
        self._watch_scan_running = True
        self.statusBar().showMessage("正在增量扫描监听文件夹…")
        worker = Worker(lambda _signals: self.controller.scan_watched_folders())
        worker.signals.result.connect(self._sync_complete)
        worker.signals.error.connect(lambda message: self._operation_error("自动同步失败", message))
        worker.signals.finished.connect(lambda: setattr(self, "_watch_scan_running", False))
        self.thread_pool.start(worker)

    def _automatic_watch_scan(self) -> None:
        if not self.controller.settings.watched_folders_enabled:
            return
        if not self.controller.watched_folders():
            return
        self.scan_watched_folders()

    def _reset_sync_timer(self) -> None:
        interval = max(1, int(self.controller.settings.watched_scan_minutes)) * 60 * 1000
        self.sync_timer.setInterval(interval)
        if self.controller.settings.watched_folders_enabled:
            self.sync_timer.start()
        else:
            self.sync_timer.stop()

    def manage_watched_folders(self) -> None:
        values = self.controller.watched_folders()
        if not values:
            QMessageBox.information(self, "监听文件夹", "还没有配置监听文件夹。")
            return
        labels = [
            f"{'✓' if item['enabled'] else '—'} {item['path']}  →  {item['collection']}"
            for item in values
        ]
        selected, accepted = QInputDialog.getItem(
            self, "管理监听文件夹", "选择要停止监听的目录（取消则不修改）：", labels, 0, False
        )
        if accepted:
            index = labels.index(selected)
            self.controller.remove_watched_folder(str(values[index]["id"]))
            self.statusBar().showMessage("已停止监听；已入库资料不会被删除", 7000)

    def show_source_packages(self) -> None:
        packages = self.controller.source_packages()
        if not packages:
            QMessageBox.information(self, "Source Package 管理器", "暂无成组归档的多模态资料。")
            return
        lines = []
        for package in packages[:80]:
            members = package.get("members") or []
            titles = [str(member.get("title") or member.get("item")) for member in members]
            lines.append(f"• {package.get('package_id')}\n  {' + '.join(titles)}")
        QMessageBox.information(self, "Source Package 管理器", "\n\n".join(lines))

    def run_workshop(self, artifact_type: str) -> None:
        labels = {
            "report": "综合报告", "compare": "多资料比较", "timeline": "时间线",
            "quiz": "测验题", "flashcards": "复习闪卡", "mindmap": "思维导图",
        }
        selected = self._selected_documents()
        title, accepted = QInputDialog.getText(
            self, f"生成{labels[artifact_type]}",
            "标题：", text=f"{labels[artifact_type]}-{datetime.now():%Y-%m-%d}",
        )
        if not accepted:
            return
        ids = [str(item["id"]) for item in selected] or None
        model_id = str(self.model_combo.currentData() or "local-extractive")

        def complete(artifact) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle(str(artifact["title"]))
            dialog.resize(900, 680)
            layout = QVBoxLayout(dialog)
            browser = QTextBrowser()
            browser.setMarkdown(str(artifact["markdown"]))
            layout.addWidget(browser, 1)
            note = QLabel(f"已保存：{artifact['path']}")
            note.setObjectName("muted")
            layout.addWidget(note)
            dialog.exec()

        self._run_simple_background(
            f"正在生成{labels[artifact_type]}…",
            lambda: self.controller.create_artifact(
                artifact_type, title, document_ids=ids, model_id=model_id
            ), complete,
        )

    def create_backup(self) -> None:
        self._run_simple_background(
            "正在创建可恢复备份…", self.controller.create_backup,
            lambda path: QMessageBox.information(self, "备份完成", f"已保存到：\n{path}\n\nAPI 密钥不会写入备份。"),
        )

    def restore_backup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 AI静静备份", str(self.controller.paths.backups),
            "AI静静备份 (*.aijjbackup);;所有文件 (*)",
        )
        if not path:
            return
        choice = QMessageBox.question(
            self, "恢复备份", "恢复前会自动创建当前数据的安全备份。确定继续吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if choice == QMessageBox.Yes:
            self._run_simple_background(
                "正在验证并恢复备份…", lambda: self.controller.restore_backup(path), self._sync_complete
            )

    def repair_database(self) -> None:
        health = self.controller.database_health()
        if health.get("ok"):
            message = "数据库完整性正常。仍可重建全文索引并执行优化。"
        else:
            message = f"检测到异常：{health}"
        choice = QMessageBox.question(
            self, "数据库检查与修复", message + "\n\n现在执行安全修复吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if choice == QMessageBox.Yes:
            self._run_simple_background(
                "正在重建全文索引并优化数据库…", self.controller.repair_database, self._sync_complete
            )

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.controller, self)
        if dialog.exec() == QDialog.Accepted:
            try:
                dialog.persist()
                self._load_models()
                self._reset_sync_timer()
                self._refresh_status()
                self.statusBar().showMessage("设置已保存", 5000)
            except (OSError, ValueError) as exc:
                self._operation_error("设置保存失败", str(exc))

    def sync_obsidian(self) -> None:
        self._run_simple_background("正在从 Obsidian 同步…", self.controller.sync_from_obsidian, self._sync_complete)

    def export_obsidian(self) -> None:
        self._run_simple_background("正在导出笔记到 Obsidian…", self.controller.export_notes_to_obsidian, self._sync_complete)

    def show_diagnostics(self) -> None:
        report = run_diagnostics(self.controller)
        lines = [
            f"Python {report['python']}",
            f"SQLite FTS5：{'可用' if report['sqlite_fts5'] else '不可用'}",
            f"FFmpeg：{report['ffmpeg'] or '不可用'}",
            "",
        ]
        lines.extend(
            f"{'✓' if item['available'] else '—'} {item['name']}：{item['purpose']}"
            for item in report["components"]
        )
        QMessageBox.information(self, "AI静静·系统诊断", "\n".join(lines))

    def check_updates(self) -> None:
        def complete(report) -> None:
            status = report.get("status")
            if status == "available":
                choice = QMessageBox.question(
                    self, "发现新版本",
                    f"当前 {report['current_version']}，最新 {report['latest_version']}。\n\n"
                    f"{report.get('notes') or ''}\n\n打开安全下载页面吗？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                )
                if choice == QMessageBox.Yes:
                    QDesktopServices.openUrl(QUrl(str(report["download_url"])))
            elif status == "current":
                QMessageBox.information(self, "检查更新", "当前已经是最新版本。")
            else:
                QMessageBox.information(self, "检查更新", str(report.get("notes") or "当前使用手动更新渠道。"))

        self._run_simple_background("正在通过 HTTPS 检查更新…", self.controller.check_for_updates, complete)

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 AI静静",
            f"<h2>{PRODUCT_NAME}</h2><p>版本 {__version__}</p>"
            "<p>本地优先的多模态知识摄取、检索与可溯源问答工具。</p>"
            "<p>Codex 和 Obsidian 均不是必需依赖。</p>",
        )

    def _run_simple_background(self, label: str, function: Callable[[], object], complete: Callable[[object], None]) -> None:
        self.statusBar().showMessage(label)
        worker = Worker(lambda _signals: function())
        worker.signals.result.connect(complete)
        worker.signals.error.connect(lambda message: self._operation_error("操作失败", message))
        self.thread_pool.start(worker)

    def _sync_complete(self, report) -> None:
        self.refresh_library()
        self._refresh_status()
        self.statusBar().showMessage(f"操作完成：{report}", 12000)

    def _refresh_status(self) -> None:
        try:
            status = self.controller.status()
            configured = "/".join(str(p["label"]) for p in status["providers"] if p["configured"]) or "本地模型"
            self.statusBar().showMessage(
                f"{status['documents']} 份资料 · {status['chunks']} 个知识块 · {configured} · 数据保存于 {self.controller.paths.root}"
            )
        except Exception as exc:
            self.statusBar().showMessage(f"状态读取失败：{exc}")

    def _open_link(self, url: QUrl) -> None:
        if url.scheme() == "aijj" and url.host() == "citation":
            evidence_id = url.path().strip("/")
            for index in range(self.evidence_list.count()):
                item = self.evidence_list.item(index)
                if str(item.data(Qt.UserRole)) == evidence_id:
                    self.evidence_list.setCurrentItem(item)
                    self.open_source_reader()
                    return
        QDesktopServices.openUrl(url)

    def _operation_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)
        self.statusBar().showMessage(f"{title}：{message}", 10000)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls and any(url.isLocalFile() or url.scheme() in {"http", "https"} for url in urls):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        values = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_file():
                    values.append(str(path))
            elif url.scheme() in {"http", "https"}:
                values.append(url.toString())
        if values:
            self.start_import(values)
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.import_token and not self.import_token.cancelled:
            choice = QMessageBox.question(
                self,
                "导入仍在进行",
                "关闭应用会取消当前导入。确定继续吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self.import_token.cancel()
        event.accept()


def create_application(data_root: str | Path | None = None) -> tuple[QApplication, MainWindow]:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName(PRODUCT_NAME)
    application.setOrganizationName("AI静静")
    try:
        icon_path = files("media_knowledge.desktop").joinpath("assets/ai_jingjing_mascot.png")
        application.setWindowIcon(QIcon(str(icon_path)))
    except (FileNotFoundError, TypeError):
        pass
    application.setStyle("Fusion")
    palette = application.palette()
    palette.setColor(QPalette.Window, QColor("#edf6fb"))
    palette.setColor(QPalette.WindowText, QColor("#294457"))
    palette.setColor(QPalette.Base, QColor("#fcfeff"))
    palette.setColor(QPalette.AlternateBase, QColor("#eaf4f9"))
    palette.setColor(QPalette.Text, QColor("#294457"))
    palette.setColor(QPalette.Button, QColor("#f8fcfe"))
    palette.setColor(QPalette.ButtonText, QColor("#294457"))
    palette.setColor(QPalette.Highlight, QColor("#9fd2e6"))
    palette.setColor(QPalette.HighlightedText, QColor("#153f5b"))
    application.setPalette(palette)
    application.setStyleSheet(APP_STYLE)
    window = MainWindow(DesktopController(data_root, migrate_legacy=data_root is None))
    return application, window


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ai-jingjing", description=PRODUCT_NAME)
    parser.add_argument("--data-dir", help="自定义知识数据目录")
    args = parser.parse_args(argv)
    try:
        application, window = create_application(args.data_dir)
        window.show()
        raise SystemExit(application.exec())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{PRODUCT_NAME}：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
