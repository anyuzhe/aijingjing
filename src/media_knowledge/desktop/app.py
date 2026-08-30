from __future__ import annotations

import argparse
import html
import os
import sys
import threading
import uuid
from dataclasses import replace
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import Callable

try:
    from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QImage, QImageReader, QKeySequence, QPalette, QPixmap, QTextDocument
    from PySide6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
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
        QSpinBox,
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
from ..product import DEFAULT_ANSWER_MODEL, DesktopSettings, PRODUCT_NAME
from ..qa.models import ImageAttachment
from ..storage import KnowledgeDatabase
from ..transcripts import TranscriptRepository
from .. import __version__
from .audio_player import MediaPlayerDialog
from .controller import DesktopController
from .diagnostics import run_diagnostics
from .glossary_manager_dialog import GlossaryManagerDialog
from .transcript_editor import TranscriptEditorDialog


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


KNOWLEDGE_TYPE_LABELS = {
    "source": "来源",
    "topic": "主题",
    "entity": "实体",
    "analysis": "分析",
    "decision": "决策",
    "output": "成果",
}
KNOWLEDGE_STATUS_LABELS = {
    "draft": "草稿",
    "current": "当前有效",
    "needs-review": "需要复核",
    "stale": "可能过期",
    "archived": "已归档",
}
KNOWLEDGE_MATURITY_LABELS = {
    "unreviewed": "未检查",
    "indexed": "已索引",
    "summarized": "已总结",
    "compiled": "已沉淀",
    "low-value": "低价值保留",
}
KNOWLEDGE_RELATION_LABELS = {
    "supports": "支持",
    "extends": "扩展",
    "contradicts": "冲突",
    "supersedes": "取代",
    "opens": "提出新问题",
}
KNOWLEDGE_HEALTH_LABELS = {
    "missing_metadata": "缺少元数据",
    "missing_summary": "缺少摘要",
    "missing_body": "正文为空",
    "source_without_evidence": "原始证据缺失",
    "orphan_item": "知识孤立",
    "isolated_source": "来源尚未沉淀",
    "stale_current": "当前知识长期未更新",
    "marked_stale": "知识已过期",
    "high_value_uncompiled": "高价值来源未编译",
    "compiled_without_source": "编译知识缺少来源",
    "noncanonical_tag": "标签不统一",
    "ambiguous_alias": "别名冲突",
}
PRIVACY_CATEGORY_LABELS = {
    "private_key": "私钥",
    "provider_api_key": "API 密钥",
    "github_token": "GitHub 令牌",
    "gitlab_token": "GitLab 令牌",
    "huggingface_token": "模型服务令牌",
    "slack_token": "协作服务令牌",
    "aws_access_key": "云平台密钥",
    "google_api_key": "Google API 密钥",
    "jwt_token": "JWT 令牌",
    "bearer_token": "Bearer 认证令牌",
    "credential_assignment": "疑似凭据赋值",
    "email_address": "电子邮箱",
    "phone_number": "手机号",
    "absolute_user_path": "本机用户路径",
    "sensitive_path": "敏感路径",
    "secret_like_file": "凭据类文件",
    "symbolic_link": "符号链接",
    "image_exif": "图片 EXIF 元数据",
    "image_gps_exif": "图片 GPS 位置",
    "image_text_unscanned": "图片文字未 OCR",
    "image_ocr_unavailable": "图片 OCR 不可用",
    "image_original_container_not_shareable": "原始图片容器不进入安全副本",
    "pdf_parser_unavailable": "PDF 解析组件不可用",
    "pdf_content_unscanned": "PDF 内容未完整扫描",
    "pdf_encrypted_content": "PDF 已加密",
    "pdf_visual_content_unscanned": "PDF 视觉内容未 OCR",
    "pdf_visual_content_unreadable": "PDF 视觉内容无法读取",
    "pdf_attachment_unscanned": "PDF 附件未完整扫描",
    "pdf_original_container_not_shareable": "原始 PDF 不进入安全副本",
    "media_content_unscanned": "音视频内容未完整扫描",
    "office_content_unscanned": "Office 内容未完整扫描",
    "office_binary_unscanned": "旧版 Office 二进制未扫描",
    "office_binary_content_unscanned": "Office 嵌入内容未扫描",
    "office_encrypted_content": "Office 内容已加密",
    "office_xml_unreadable": "Office XML 无法解析",
    "office_original_container_not_shareable": "原始 Office/ODF 不进入安全副本",
    "archive_content_unscanned": "压缩包内容未扫描",
    "binary_content_unscanned": "二进制内容未解析",
}
INGESTION_STAGE_LABELS = {
    "queued": "等待处理",
    "preparing": "识别资料",
    "extracting": "解析内容",
    "validating": "质量检查",
    "archiving": "归档原件",
    "indexing": "建立索引",
    "noting": "生成笔记",
    "complete": "完成",
    "failed": "处理失败",
    "cancelled": "已取消",
}
INGESTION_STATUS_LABELS = {
    "created": "已新建",
    "updated": "已更新",
    "unchanged": "无变化",
    "duplicate": "重复资料",
    "failed": "失败",
    "cancelled": "已取消",
}


def _markdown_html(markdown: str) -> str:
    document = QTextDocument()
    document.setMarkdown(markdown)
    return document.toHtml()


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(object)
    delta = Signal(str)
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


class KnowledgeCaptureDialog(QDialog):
    """Collect governance metadata before promoting an answer into durable knowledge."""

    def __init__(
        self,
        *,
        suggested_title: str,
        suggested_summary: str,
        evidence_count: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("沉淀为正式知识")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "把当前回答从一次性对话提升为可检索、可复核的正式知识。"
            f"保存后会关联当前回答使用的 {evidence_count} 条来源证据。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        form = QFormLayout()
        self.title_edit = QLineEdit(suggested_title.strip()[:180])
        self.title_edit.setAccessibleName("知识标题")
        form.addRow("标题（必填）", self.title_edit)
        self.type_combo = QComboBox()
        for item_type in ("analysis", "topic", "entity", "decision", "output"):
            self.type_combo.addItem(KNOWLEDGE_TYPE_LABELS[item_type], item_type)
        self.type_combo.setAccessibleName("知识类型")
        form.addRow("知识类型", self.type_combo)
        self.status_combo = QComboBox()
        for status in ("needs-review", "draft", "current"):
            self.status_combo.addItem(KNOWLEDGE_STATUS_LABELS[status], status)
        self.status_combo.setAccessibleName("知识状态")
        form.addRow("状态", self.status_combo)
        self.summary_edit = QPlainTextEdit(suggested_summary.strip()[:1000])
        self.summary_edit.setAccessibleName("知识摘要")
        self.summary_edit.setPlaceholderText("用一两句话概括这条知识，便于以后搜索和判断。")
        self.summary_edit.setFixedHeight(84)
        form.addRow("摘要", self.summary_edit)
        self.aliases_edit = QLineEdit()
        self.aliases_edit.setPlaceholderText("中文别名、English alias（用逗号分隔）")
        self.aliases_edit.setAccessibleName("知识别名")
        form.addRow("别名", self.aliases_edit)
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("例如：slam, lidar, 工程实践")
        self.tags_edit.setAccessibleName("知识标签")
        form.addRow("标签", self.tags_edit)
        layout.addLayout(form)

        note = QLabel("AI 生成内容默认标记为“需要复核”；只有确认无误后再改为“当前有效”。")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("沉淀为知识")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _split_values(value: str) -> list[str]:
        normalized = value.replace("，", ",").replace("\n", ",")
        return list(dict.fromkeys(item.strip() for item in normalized.split(",") if item.strip()))

    def accept(self) -> None:
        if not self.title_edit.text().strip():
            QMessageBox.warning(self, "还需要标题", "请填写一个便于以后检索的知识标题。")
            self.title_edit.setFocus(Qt.OtherFocusReason)
            return
        super().accept()

    def values(self) -> dict[str, object]:
        return {
            "title": self.title_edit.text().strip(),
            "item_type": str(self.type_combo.currentData()),
            "status": str(self.status_combo.currentData()),
            "summary": self.summary_edit.toPlainText().strip(),
            "aliases": self._split_values(self.aliases_edit.text()),
            "tags": self._split_values(self.tags_edit.text()),
        }


class KnowledgeHealthDialog(QDialog):
    def __init__(self, report: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("知识体检中心")
        self.resize(1040, 650)
        layout = QVBoxLayout(self)
        counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
        issue_count = int(report.get("issue_count") or len(report.get("issues") or []))
        summary = QLabel(
            f"正式知识 {int(counts.get('items', 0))} 条 · 待复核 {int(counts.get('needs_review', 0))} 条 · "
            f"可能过期 {int(counts.get('stale', 0))} 条 · 共发现 {issue_count} 个治理提醒"
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("font-size:15px;font-weight:650;color:#285d7a;")
        layout.addWidget(summary)
        intro = QLabel("问题会同时说明原因和恢复建议；状态不只依赖颜色，方便键盘和辅助功能用户判断。")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 6)
        self.table.setAccessibleName("知识健康问题列表")
        self.table.setHorizontalHeaderLabels(["级别", "问题", "知识", "类型", "说明", "建议操作"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        issues = report.get("issues") or []
        self.table.setRowCount(len(issues))
        severity_labels = {"error": "需处理", "warning": "需关注", "info": "建议"}
        for row_index, issue in enumerate(issues):
            values = [
                severity_labels.get(str(issue.get("severity")), str(issue.get("severity") or "建议")),
                KNOWLEDGE_HEALTH_LABELS.get(
                    str(issue.get("category") or issue.get("code") or ""),
                    str(issue.get("category") or issue.get("code") or "治理提醒"),
                ),
                str(issue.get("title") or issue.get("item_title") or "—"),
                KNOWLEDGE_TYPE_LABELS.get(str(issue.get("item_type") or ""), str(issue.get("item_type") or "—")),
                str(issue.get("message") or issue.get("detail") or ""),
                str(issue.get("suggestion") or issue.get("recovery") or "打开知识后补充信息"),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        layout.addWidget(self.table, 1)
        if not issues:
            empty = QLabel("✓ 当前没有发现需要处理的知识治理问题。")
            empty.setStyleSheet("color:#28705f;font-weight:600;")
            layout.addWidget(empty)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class KnowledgeTrashDialog(QDialog):
    """Compact recovery surface for governed knowledge tombstones."""

    def __init__(
        self, controller: DesktopController, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.restored_ids: list[str] = []
        self.setWindowTitle("知识回收站")
        self.resize(720, 480)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "删除的正式知识会保留条目、别名、标签、知识关系和 Markdown 笔记。"
            "选择一项即可恢复；原位置已有文件时会使用安全的新文件名。"
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.trash_list = QListWidget()
        self.trash_list.setAccessibleName("知识回收站列表")
        self.trash_list.setWordWrap(True)
        self.trash_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.trash_list.currentItemChanged.connect(self._selection_changed)
        self.trash_list.itemDoubleClicked.connect(lambda _item: self._restore_selected())
        layout.addWidget(self.trash_list, 1)
        self.status_label = QLabel("")
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.restore_button = QPushButton("恢复所选知识")
        self.restore_button.setAccessibleName("恢复所选知识")
        self.restore_button.setEnabled(False)
        self.restore_button.clicked.connect(self._restore_selected)
        actions.addWidget(self.restore_button)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        self.refresh_items()

    def refresh_items(self) -> None:
        self.trash_list.clear()
        try:
            rows = self.controller.knowledge_trash_items()
        except (OSError, ValueError) as exc:
            self.status_label.setText(f"无法读取知识回收站：{exc}")
            self.restore_button.setEnabled(False)
            return
        for record in rows:
            item_type = str(record.get("item_type") or "analysis")
            deleted_at = str(record.get("deleted_at") or "未知时间")
            note_label = "含 Markdown" if record.get("has_note") else "无独立笔记"
            relation_count = int(record.get("relation_count") or 0)
            item = QListWidgetItem(
                f"{KNOWLEDGE_TYPE_LABELS.get(item_type, item_type)} · "
                f"{record.get('title') or '未命名知识'}\n"
                f"删除于 {deleted_at} · {note_label} · {relation_count} 条关系"
            )
            item.setData(Qt.UserRole, record)
            self.trash_list.addItem(item)
        self.status_label.setText(
            "回收站为空。" if not rows else f"共有 {len(rows)} 条可恢复知识。"
        )
        self._selection_changed(self.trash_list.currentItem())

    def _selection_changed(self, item: QListWidgetItem | None, _previous=None) -> None:
        self.restore_button.setEnabled(item is not None)

    def _restore_selected(self) -> None:
        item = self.trash_list.currentItem()
        record = item.data(Qt.UserRole) if item is not None else None
        if not isinstance(record, dict):
            return
        try:
            restored = self.controller.restore_knowledge_item(
                str(record.get("tombstone_id") or "")
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.status_label.setText(f"恢复失败：{exc}")
            return
        restored_id = str(restored.get("id") or "")
        if restored_id:
            self.restored_ids.append(restored_id)
        relation_count = int(restored.get("restored_relation_count") or 0)
        self.refresh_items()
        self.status_label.setText(
            f"已恢复“{restored.get('title') or '未命名知识'}”及 {relation_count} 条仍有效关系。"
        )


class PromptEdit(QPlainTextEdit):
    submit = Signal()
    imagesPasted = Signal(object)

    def canInsertFromMimeData(self, source) -> bool:  # noqa: N802
        return bool(source.hasImage() or source.hasUrls() or super().canInsertFromMimeData(source))

    def insertFromMimeData(self, source) -> None:  # noqa: N802
        if source.hasImage():
            self.imagesPasted.emit(source.imageData())
            return
        local_files = [url.toLocalFile() for url in source.urls() if url.isLocalFile()]
        if local_files:
            readers = [QImageReader(path) for path in local_files]
            image_files = [path for path, reader in zip(local_files, readers) if reader.canRead()]
            if image_files and len(image_files) == len(local_files):
                self.imagesPasted.emit(image_files)
                return
        super().insertFromMimeData(source)

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
        self.resize(760, 780)
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
        self.embedding_provider = QComboBox()
        for label, value in (
            ("本地哈希检索（零下载、稳定可用）", "hash"),
            ("FastEmbed 多语义检索（仅使用已安装模型）", "fastembed"),
        ):
            self.embedding_provider.addItem(label, value)
            if value == controller.settings.embedding_provider:
                self.embedding_provider.setCurrentIndex(
                    self.embedding_provider.count() - 1
                )
        self.embedding_provider.setAccessibleName("本地向量检索方式")
        form.addRow("向量检索", self.embedding_provider)
        embedding_hint = QLabel(
            "导入和提问期间不会下载向量模型。新安装默认使用本地哈希；"
            "选择 FastEmbed 前请先显式准备模型。"
        )
        embedding_hint.setWordWrap(True)
        embedding_hint.setObjectName("muted")
        form.addRow("", embedding_hint)
        self.archive = QCheckBox("将原始资料归档到应用目录")
        self.archive.setChecked(controller.settings.archive_originals)
        form.addRow("", self.archive)
        self.notes = QCheckBox("自动生成 Markdown Source Note")
        self.notes.setChecked(controller.settings.create_source_notes)
        form.addRow("", self.notes)
        self.synthesis = QCheckBox("入库时直连 DeepSeek 生成 AI 知识提炼（优先 Flash）")
        self.synthesis.setChecked(controller.settings.auto_synthesize_notes)
        form.addRow("", self.synthesis)
        self.vision = QCheckBox("使用 DeepSeek Vision 或 Kimi 进行高级视觉理解")
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

        media = QWidget()
        media_form = QFormLayout(media)
        self.ocr_engine = QComboBox()
        for label, value in (
            ("自动（普通图片优先 RapidOCR）", "auto"),
            ("RapidOCR（轻量、速度优先）", "rapidocr"),
            ("PaddleOCR PP-StructureV3（复杂版面）", "paddleocr"),
        ):
            self.ocr_engine.addItem(label, value)
            if value == controller.settings.ocr_engine:
                self.ocr_engine.setCurrentIndex(self.ocr_engine.count() - 1)
        self.ocr_engine.setAccessibleName("OCR 解析引擎")
        media_form.addRow("OCR 引擎", self.ocr_engine)
        self.complex_ocr = QCheckBox("自动识别表格、公式、多栏等复杂版面")
        self.complex_ocr.setChecked(controller.settings.ocr_complex_layout_enabled)
        media_form.addRow("", self.complex_ocr)
        self.ocr_threshold = QDoubleSpinBox()
        self.ocr_threshold.setRange(0.0, 1.0)
        self.ocr_threshold.setSingleStep(0.05)
        self.ocr_threshold.setDecimals(2)
        self.ocr_threshold.setValue(controller.settings.ocr_low_confidence_threshold)
        self.ocr_threshold.setAccessibleName("OCR 低置信度阈值")
        media_form.addRow("低置信度阈值", self.ocr_threshold)

        self.transcription_profile = QComboBox()
        for label, value in (
            ("中文高精度（Qwen3-ASR 1.7B）", "chinese-accuracy"),
            ("快速预览（Qwen3-ASR 0.6B）", "fast-preview"),
            ("兼容模式（Whisper）", "compatibility"),
            ("自定义路线", "custom"),
        ):
            self.transcription_profile.addItem(label, value)
            if value == controller.settings.transcription_profile:
                self.transcription_profile.setCurrentIndex(self.transcription_profile.count() - 1)
        self.transcription_profile.setAccessibleName("转写处理方案")
        media_form.addRow("转写方案", self.transcription_profile)

        self.asr_provider = QComboBox()
        for label, value in (
            ("自动选择（按方案和本地可用性）", "auto"),
            ("Qwen3-ASR · Apple MLX", "qwen3-mlx"),
            ("Whisper · Apple MLX", "mlx-whisper"),
            ("faster-whisper · CUDA / CPU", "faster-whisper"),
        ):
            self.asr_provider.addItem(label, value)
            if value == controller.settings.asr_provider:
                self.asr_provider.setCurrentIndex(self.asr_provider.count() - 1)
        self.asr_provider.setAccessibleName("语音识别引擎")
        media_form.addRow("识别引擎", self.asr_provider)

        self.asr_model = QComboBox()
        for label, model in (
            ("Qwen3-ASR 1.7B（中文高精度）", "Qwen3-ASR-1.7B"),
            ("Qwen3-ASR 0.6B（快速预览）", "Qwen3-ASR-0.6B"),
            ("Whisper Large v3（高精度兼容）", "large-v3"),
            ("Whisper Medium", "medium"),
            ("Whisper Small", "small"),
            ("Whisper Base", "base"),
            ("Whisper Tiny（速度优先）", "tiny"),
        ):
            self.asr_model.addItem(label, model)
        current_asr_model = controller.settings.asr_model or controller.settings.whisper_model
        matched = self.asr_model.findData(current_asr_model)
        if matched >= 0:
            self.asr_model.setCurrentIndex(matched)
        self.asr_model.setAccessibleName("语音识别具体模型")
        self.whisper_model = self.asr_model  # compatibility for existing extensions
        media_form.addRow("具体模型", self.asr_model)

        model_row = QHBoxLayout()
        self.asr_model_status = QLabel()
        self.asr_model_status.setWordWrap(True)
        self.asr_model_status.setObjectName("muted")
        manage_models = QPushButton("管理本地模型…")
        manage_models.setAccessibleName("打开本地转写模型管理器")
        manage_models.clicked.connect(self._open_model_manager)
        model_row.addWidget(self.asr_model_status, 1)
        model_row.addWidget(manage_models)
        media_form.addRow("本地状态", model_row)

        self.transcription_engine = QComboBox()
        for label, value in (
            ("自动（Apple MLX → NVIDIA CUDA → CPU）", "auto"),
            ("Apple Silicon · MLX", "mlx"),
            ("NVIDIA · CUDA", "cuda"),
            ("CPU · int8", "cpu"),
        ):
            self.transcription_engine.addItem(label, value)
            if value == controller.settings.transcription_engine:
                self.transcription_engine.setCurrentIndex(self.transcription_engine.count() - 1)
        self.transcription_engine.setAccessibleName("音视频转写引擎")
        media_form.addRow("转写加速", self.transcription_engine)
        self.cpu_fallback = QCheckBox("加速引擎不可用时，允许明确降级到 CPU（较慢）")
        self.cpu_fallback.setChecked(controller.settings.transcription_allow_cpu_fallback)
        media_form.addRow("", self.cpu_fallback)

        self.transcription_language = QComboBox()
        for label, value in (
            ("自动识别", "auto"),
            ("中文", "zh"),
            ("英文", "en"),
        ):
            self.transcription_language.addItem(label, value)
            if value == controller.settings.transcription_language:
                self.transcription_language.setCurrentIndex(self.transcription_language.count() - 1)
        media_form.addRow("主要语言", self.transcription_language)
        self.word_timestamps = QCheckBox("保留词级时间戳（用于精确引用和说话人对齐）")
        self.word_timestamps.setChecked(controller.settings.word_timestamps)
        media_form.addRow("", self.word_timestamps)

        self.asr_context_terms = QPlainTextEdit()
        self.asr_context_terms.setPlainText("\n".join(controller.settings.asr_context_terms))
        self.asr_context_terms.setPlaceholderText("每行一个术语，例如：FLAC3D\nFAST-LIVO2\n结构面")
        self.asr_context_terms.setMaximumHeight(92)
        self.asr_context_terms.setAccessibleName("本次默认转写上下文术语")
        media_form.addRow("上下文术语", self.asr_context_terms)

        self.asr_knowledge_space_id = QLineEdit(
            controller.settings.asr_knowledge_space_id
        )
        self.asr_knowledge_space_id.setPlaceholderText("本地知识库")
        self.asr_knowledge_space_id.setAccessibleName("转写知识空间 ID")
        media_form.addRow("知识空间 ID", self.asr_knowledge_space_id)
        manage_glossaries = QPushButton("管理专业词库…")
        manage_glossaries.setAccessibleName("打开音视频专业词库管理器")
        manage_glossaries.clicked.connect(self._open_glossary_manager)
        media_form.addRow("专业词库", manage_glossaries)

        self.diarization = QCheckBox("区分多位说话人（需要已安装的本地模型）")
        self.diarization.setChecked(controller.settings.diarization_enabled)
        media_form.addRow("", self.diarization)
        self.diarization_provider = QComboBox()
        for label, value in (
            ("自动选择", "auto"),
            ("pyannote Community-1", "pyannote"),
            ("Sherpa-ONNX（预留）", "sherpa"),
            ("不执行说话人识别", "none"),
        ):
            self.diarization_provider.addItem(label, value)
            if value == controller.settings.diarization_provider:
                self.diarization_provider.setCurrentIndex(self.diarization_provider.count() - 1)
        media_form.addRow("说话人引擎", self.diarization_provider)
        speaker_row = QHBoxLayout()
        self.min_speakers = QSpinBox()
        self.min_speakers.setRange(1, 20)
        self.min_speakers.setValue(controller.settings.diarization_min_speakers)
        self.min_speakers.setPrefix("最少 ")
        self.min_speakers.setSuffix(" 人")
        self.max_speakers = QSpinBox()
        self.max_speakers.setRange(1, 20)
        self.max_speakers.setValue(controller.settings.diarization_max_speakers)
        self.max_speakers.setPrefix("最多 ")
        self.max_speakers.setSuffix(" 人")
        speaker_row.addWidget(self.min_speakers)
        speaker_row.addWidget(self.max_speakers)
        speaker_row.addStretch(1)
        media_form.addRow("人数范围", speaker_row)
        self.transcript_quality_gate = QCheckBox("质量为“需要复核/失败”时，不进入高可信问答索引")
        self.transcript_quality_gate.setChecked(controller.settings.transcript_quality_gate)
        media_form.addRow("", self.transcript_quality_gate)
        media_hint = QLabel(
            "每次 OCR 会保留文字框、置信度和降级原因；"
            "音视频会保留原始转写、校订稿、时间戳、质量报告及运行路线。"
            "缺少模型时会明确提示安装或切换，不会在导入过程中偷偷下载。"
            "任何技术回退都会写入任务记录；用户取消不会触发回退。"
        )
        media_hint.setWordWrap(True)
        media_hint.setObjectName("muted")
        media_form.addRow("", media_hint)
        self.transcription_profile.currentIndexChanged.connect(self._apply_transcription_profile)
        self.asr_provider.currentIndexChanged.connect(self._refresh_asr_model_status)
        self.asr_model.currentIndexChanged.connect(self._refresh_asr_model_status)
        self.diarization.toggled.connect(self._update_diarization_controls)
        self.min_speakers.valueChanged.connect(
            lambda value: self.max_speakers.setMinimum(value)
        )
        self._update_diarization_controls(self.diarization.isChecked())
        self._refresh_asr_model_status()
        media_scroll = QScrollArea()
        media_scroll.setWidgetResizable(True)
        media_scroll.setFrameShape(QFrame.Shape.NoFrame)
        media_scroll.setWidget(media)
        media_scroll.setAccessibleName("多媒体解析设置")
        tabs.addTab(media_scroll, "多媒体解析")

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

    @staticmethod
    def _managed_model_id(model: str, provider: str = "auto") -> str | None:
        if provider == "faster-whisper" and model in {
            "large-v3", "medium", "small", "base", "tiny",
        }:
            return f"faster-whisper-{model}"
        return {
            "Qwen3-ASR-1.7B": "qwen3-asr-1.7b-mlx",
            "Qwen3-ASR-0.6B": "qwen3-asr-0.6b-mlx",
            "large-v3": "whisper-large-v3-mlx",
            "medium": "whisper-medium-mlx",
            "small": "whisper-small-mlx",
            "base": "whisper-base-mlx",
            "tiny": "whisper-tiny-mlx",
        }.get(model)

    @Slot()
    def _apply_transcription_profile(self) -> None:
        profile = str(self.transcription_profile.currentData() or "compatibility")
        preferred: tuple[str, str] | None = {
            "chinese-accuracy": ("qwen3-mlx", "Qwen3-ASR-1.7B"),
            "fast-preview": ("qwen3-mlx", "Qwen3-ASR-0.6B"),
        }.get(profile)
        if profile == "compatibility":
            current = str(self.asr_model.currentData() or "")
            preferred = ("auto", current if current in {"large-v3", "medium", "small", "base", "tiny"} else self.controller.settings.whisper_model)
        if preferred:
            provider_index = self.asr_provider.findData(preferred[0])
            model_index = self.asr_model.findData(preferred[1])
            if provider_index >= 0:
                self.asr_provider.setCurrentIndex(provider_index)
            if model_index >= 0:
                self.asr_model.setCurrentIndex(model_index)
        self._refresh_asr_model_status()

    @Slot()
    def _refresh_asr_model_status(self) -> None:
        model = str(self.asr_model.currentData() or "")
        provider = str(self.asr_provider.currentData() or "auto")
        managed_id = self._managed_model_id(model, provider)
        if managed_id is None:
            self.asr_model_status.setText("尚未选择可识别的本地模型。")
            return
        try:
            status = self.controller.local_models.status(managed_id)
        except (OSError, ValueError) as error:
            self.asr_model_status.setText(f"无法检查模型状态：{error}")
            return
        if status.verified:
            size_gb = status.size_bytes / (1024 ** 3)
            self.asr_model_status.setText(
                f"已安装 · {size_gb:.2f} GB · {status.source} · {status.path}"
            )
        else:
            self.asr_model_status.setText(
                "尚未安装。导入时不会自动联网；请先打开模型管理器安装或登记已有目录。"
            )

    @Slot()
    def _open_model_manager(self) -> None:
        from .model_manager_dialog import ModelManagerDialog

        dialog = ModelManagerDialog(self.controller.local_models, self)
        dialog.exec()
        self._refresh_asr_model_status()

    @Slot()
    def _open_glossary_manager(self) -> None:
        knowledge_space_id = self.asr_knowledge_space_id.text().strip() or "本地知识库"
        dialog = GlossaryManagerDialog(
            self.controller,
            self,
            knowledge_space_id=knowledge_space_id,
        )
        dialog.exec()

    @Slot(bool)
    def _update_diarization_controls(self, enabled: bool) -> None:
        self.diarization_provider.setEnabled(enabled)
        self.min_speakers.setEnabled(enabled)
        self.max_speakers.setEnabled(enabled)

    def _choose_obsidian(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "选择 Obsidian Vault", self.obsidian_path.text())
        if value:
            self.obsidian_path.setText(value)

    def persist(self) -> None:
        configured_deepseek_now = bool(self.deepseek_key.text().strip())
        if configured_deepseek_now:
            self.controller.providers.update("deepseek", api_key=self.deepseek_key.text())
        if self.kimi_key.text().strip():
            self.controller.providers.update("kimi", api_key=self.kimi_key.text())
        model = str(self.model.currentData() or self.controller.settings.default_model)
        if configured_deepseek_now and model == "local-extractive":
            model = DEFAULT_ANSWER_MODEL
        asr_model = str(self.asr_model.currentData() or self.controller.settings.whisper_model)
        selected_asr_provider = str(self.asr_provider.currentData() or "auto")
        # Provider is part of model identity: Whisper MLX and CTranslate2
        # weights share model names but are not interchangeable.
        managed_model_id = self._managed_model_id(asr_model, selected_asr_provider)
        model_path = (
            self.controller.resolve_transcription_model(managed_model_id)
            if managed_model_id else None
        )
        managed_model_status = (
            self.controller.local_models.status(managed_model_id)
            if managed_model_id else None
        )
        model_sha256 = (
            managed_model_status.content_sha256
            if managed_model_status and managed_model_status.content_verified
            else None
        )
        legacy_whisper_model = (
            asr_model
            if asr_model in {"tiny", "base", "small", "medium", "large-v3"}
            else self.controller.settings.whisper_model
        )
        context_terms = list(dict.fromkeys(
            line.strip()
            for line in self.asr_context_terms.toPlainText().splitlines()
            if line.strip()
        ))[:200]
        diarization_path = self.controller.resolve_transcription_model("pyannote-community-1")
        diarization_status = self.controller.local_models.status("pyannote-community-1")
        whisper_fallback_path = self.controller.resolve_transcription_model(
            "whisper-small-mlx"
        )
        whisper_fallback_status = self.controller.local_models.status(
            "whisper-small-mlx"
        )
        settings = replace(
            self.controller.settings,
            default_model=model,
            embedding_provider=str(
                self.embedding_provider.currentData() or "hash"
            ),
            embedding_model=(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                if self.embedding_provider.currentData() == "fastembed"
                else "hash-384-v1"
            ),
            archive_originals=self.archive.isChecked(),
            create_source_notes=self.notes.isChecked(),
            auto_synthesize_notes=self.synthesis.isChecked(),
            enable_cloud_vision=self.vision.isChecked(),
            ocr_engine=str(self.ocr_engine.currentData() or "auto"),
            ocr_complex_layout_enabled=self.complex_ocr.isChecked(),
            ocr_low_confidence_threshold=float(self.ocr_threshold.value()),
            whisper_model=legacy_whisper_model,
            transcription_engine=str(self.transcription_engine.currentData() or "auto"),
            transcription_allow_cpu_fallback=self.cpu_fallback.isChecked(),
            transcription_profile=str(self.transcription_profile.currentData() or "compatibility"),
            asr_provider=selected_asr_provider,
            asr_model=asr_model,
            asr_model_path=model_path,
            asr_model_sha256=model_sha256,
            asr_whisper_fallback_model_path=whisper_fallback_path,
            asr_whisper_fallback_model_sha256=(
                whisper_fallback_status.content_sha256
                if whisper_fallback_status.content_verified else None
            ),
            transcription_language=str(self.transcription_language.currentData() or "auto"),
            asr_knowledge_space_id=(
                self.asr_knowledge_space_id.text().strip() or "本地知识库"
            ),
            asr_context_terms=context_terms,
            word_timestamps=self.word_timestamps.isChecked(),
            diarization_enabled=self.diarization.isChecked(),
            diarization_provider=str(self.diarization_provider.currentData() or "auto"),
            diarization_model_path=diarization_path,
            diarization_model_sha256=(
                diarization_status.content_sha256
                if diarization_status.content_verified else None
            ),
            diarization_min_speakers=int(self.min_speakers.value()),
            diarization_max_speakers=max(
                int(self.min_speakers.value()), int(self.max_speakers.value())
            ),
            transcript_quality_gate=self.transcript_quality_gate.isChecked(),
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
        self.import_job_id: str | None = None
        self.import_items: list[str] = []
        self.conversation_id: str | None = None
        self.last_answer = None
        self.last_question = ""
        self.evidence_by_id: dict[str, object] = {}
        self._answer_busy = False
        self._answer_worker: Worker | None = None
        self._pending_images: list[ImageAttachment] = []
        self._inflight_images: list[ImageAttachment] = []
        self._chat_entries: list[dict[str, object]] = []
        self._stream_text = ""
        self._stream_render_scheduled = False
        self._answer_cancelled = threading.Event()
        self.last_answer_id: str | None = None
        self.last_answer_markdown = ""
        self.last_retrieval_info: dict[str, object] = {}
        self._watch_scan_running = False
        self._watch_operation_token: object | None = None
        self._background_worker: Worker | None = None
        self._background_operation_label = ""
        self._active_db_operation_token: object | None = None
        self._active_db_operation_label = ""
        self._import_operation_token: object | None = None
        self._answer_operation_token: object | None = None
        self._search_operation_token: object | None = None
        self._media_players: list[MediaPlayerDialog] = []
        self.setWindowTitle(PRODUCT_NAME)
        self.resize(1510, 920)
        self.setMinimumSize(1120, 700)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        self._load_models()
        self.refresh_library()
        self.refresh_knowledge()
        self.refresh_conversations()
        self.refresh_ingestion_jobs()
        self._refresh_status()
        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self._automatic_watch_scan)
        self._reset_sync_timer()
        QTimer.singleShot(2500, self._automatic_watch_scan)
        if controller.migrated_database:
            self.statusBar().showMessage(f"已迁移现有知识库：{controller.migrated_database}", 12000)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("文件")
        new_conversation = QAction("新对话", self)
        new_conversation.setShortcut(QKeySequence.New)
        new_conversation.triggered.connect(self.new_chat)
        menu.addAction(new_conversation)
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
        focus_search = QAction("搜索全部知识", self)
        focus_search.setShortcut(QKeySequence("Ctrl+K"))
        focus_search.triggered.connect(self._focus_global_search)
        knowledge_menu.addAction(focus_search)
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
        transcript_editor = QAction("播放与校订音视频转写…", self)
        transcript_editor.triggered.connect(self.open_transcript_editor)
        knowledge_menu.addAction(transcript_editor)
        governance = QAction("知识体检中心…", self)
        governance.triggered.connect(self.show_knowledge_health)
        knowledge_menu.addAction(governance)
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
        safety_menu.addSeparator()
        privacy_scan = QAction("运行本地隐私扫描…", self)
        privacy_scan.triggered.connect(self.run_privacy_scan)
        safety_menu.addAction(privacy_scan)
        share_copy = QAction("生成安全分享副本…", self)
        share_copy.triggered.connect(self.create_safe_share_copy)
        safety_menu.addAction(share_copy)

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
        self.global_search.setAccessibleName("搜索全部知识")
        self.global_search.setAccessibleDescription("本地全文与语义搜索，不会调用云端大模型")
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
        stop_action = QAction("停止生成", self)
        stop_action.setShortcut(QKeySequence("Escape"))
        stop_action.triggered.connect(self.stop_answer)
        self.addAction(stop_action)

    def _focus_global_search(self) -> None:
        self.global_search.setFocus(Qt.ShortcutFocusReason)
        self.global_search.selectAll()
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
        self.document_list.setAccessibleName("知识资料列表")
        self.document_list.setWordWrap(True)
        self.document_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.document_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.document_list.currentItemChanged.connect(self._show_document)
        self.document_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.document_list.customContextMenuRequested.connect(self._document_menu)
        library_layout.addWidget(self.document_list, 1)
        self.left_tabs.addTab(library_tab, "资料库")

        knowledge_tab = QWidget()
        self.knowledge_tab = knowledge_tab
        knowledge_layout = QVBoxLayout(knowledge_tab)
        knowledge_layout.setContentsMargins(0, 8, 0, 0)
        self.knowledge_summary = QLabel("正在读取知识网络…")
        self.knowledge_summary.setObjectName("muted")
        self.knowledge_summary.setWordWrap(True)
        knowledge_layout.addWidget(self.knowledge_summary)
        self.knowledge_search = QLineEdit()
        self.knowledge_search.setPlaceholderText("搜索主题、实体、分析或决策…")
        self.knowledge_search.setAccessibleName("搜索正式知识")
        self.knowledge_search.textChanged.connect(self.refresh_knowledge)
        knowledge_layout.addWidget(self.knowledge_search)
        knowledge_filters = QHBoxLayout()
        self.knowledge_type_filter = QComboBox()
        self.knowledge_type_filter.addItem("全部类型", "")
        for item_type in ("source", "topic", "entity", "analysis", "decision", "output"):
            self.knowledge_type_filter.addItem(KNOWLEDGE_TYPE_LABELS[item_type], item_type)
        self.knowledge_type_filter.currentIndexChanged.connect(self.refresh_knowledge)
        knowledge_filters.addWidget(self.knowledge_type_filter)
        self.knowledge_status_filter = QComboBox()
        self.knowledge_status_filter.addItem("全部状态", "")
        for status in ("needs-review", "current", "draft", "stale", "archived"):
            self.knowledge_status_filter.addItem(KNOWLEDGE_STATUS_LABELS[status], status)
        self.knowledge_status_filter.currentIndexChanged.connect(self.refresh_knowledge)
        knowledge_filters.addWidget(self.knowledge_status_filter)
        knowledge_layout.addLayout(knowledge_filters)
        self.knowledge_list = QListWidget()
        self.knowledge_list.setAccessibleName("正式知识列表")
        self.knowledge_list.setWordWrap(True)
        self.knowledge_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.knowledge_list.currentItemChanged.connect(self._show_knowledge_item)
        self.knowledge_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.knowledge_list.customContextMenuRequested.connect(self._knowledge_menu)
        knowledge_layout.addWidget(self.knowledge_list, 1)
        knowledge_actions = QHBoxLayout()
        promote = QPushButton("沉淀当前回答")
        promote.setToolTip("把当前回答保存为主题、实体、分析、决策或成果")
        promote.clicked.connect(self.capture_last_answer_as_knowledge)
        knowledge_actions.addWidget(promote, 1)
        inspect = QPushButton("体检")
        inspect.setToolTip("检查待复核、过期、孤立和缺少来源的知识")
        inspect.clicked.connect(self.show_knowledge_health)
        knowledge_actions.addWidget(inspect)
        self.knowledge_trash_button = QPushButton("回收站")
        self.knowledge_trash_button.setAccessibleName("打开知识回收站")
        self.knowledge_trash_button.setToolTip("查看并恢复已删除的正式知识")
        self.knowledge_trash_button.clicked.connect(self.show_knowledge_trash)
        knowledge_actions.addWidget(self.knowledge_trash_button)
        knowledge_layout.addLayout(knowledge_actions)
        self.left_tabs.addTab(knowledge_tab, "知识")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(0, 8, 0, 0)
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("搜索对话标题或内容…")
        self.history_search_timer = QTimer(self)
        self.history_search_timer.setSingleShot(True)
        self.history_search_timer.setInterval(220)
        self.history_search_timer.timeout.connect(self.refresh_conversations)
        self.history_search.textChanged.connect(lambda _text: self.history_search_timer.start())
        history_layout.addWidget(self.history_search)
        self.history_list = QListWidget()
        self.history_list.setAccessibleName("历史对话列表")
        self.history_list.setWordWrap(True)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_list.itemActivated.connect(self.open_selected_conversation)
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._conversation_menu)
        history_layout.addWidget(self.history_list, 1)
        history_controls = QHBoxLayout()
        open_history = QPushButton("打开")
        open_history.clicked.connect(self.open_selected_conversation)
        history_controls.addWidget(open_history)
        rename_history = QPushButton("重命名")
        rename_history.clicked.connect(self.rename_selected_conversation)
        history_controls.addWidget(rename_history)
        export_history = QPushButton("导出")
        export_history.clicked.connect(self.export_selected_conversation)
        history_controls.addWidget(export_history)
        history_layout.addLayout(history_controls)
        self.left_tabs.addTab(history_tab, "对话")

        task_tab = QWidget()
        self.task_tab = task_tab
        task_layout = QVBoxLayout(task_tab)
        task_layout.setContentsMargins(0, 8, 0, 0)
        self.task_list = QListWidget()
        self.task_list.setAccessibleName("导入任务列表")
        self.task_list.currentItemChanged.connect(self._update_task_controls)
        self.task_list.itemActivated.connect(lambda _item: self.retry_failed())
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
        self.chat.setAccessibleName("对话内容")
        self.chat.setOpenExternalLinks(False)
        self.chat.setHtml(self._welcome_html())
        self.chat.anchorClicked.connect(self._open_link)
        layout.addWidget(self.chat, 1)

        self.answer_actions = QFrame()
        answer_actions_layout = QHBoxLayout(self.answer_actions)
        answer_actions_layout.setContentsMargins(4, 0, 4, 0)
        self.copy_answer_button = QPushButton("复制回答")
        self.copy_answer_button.setToolTip("复制当前回答的 Markdown 文本")
        self.copy_answer_button.clicked.connect(self.copy_last_answer)
        answer_actions_layout.addWidget(self.copy_answer_button)
        self.regenerate_button = QPushButton("重新生成")
        self.regenerate_button.setToolTip("使用同一问题和当前模型重新生成")
        self.regenerate_button.clicked.connect(self.regenerate_answer)
        answer_actions_layout.addWidget(self.regenerate_button)
        self.helpful_button = QPushButton("有帮助")
        self.helpful_button.clicked.connect(lambda: self.save_answer_feedback("up"))
        answer_actions_layout.addWidget(self.helpful_button)
        self.unhelpful_button = QPushButton("需改进")
        self.unhelpful_button.clicked.connect(lambda: self.save_answer_feedback("down"))
        answer_actions_layout.addWidget(self.unhelpful_button)
        self.capture_knowledge_button = QPushButton("沉淀为知识")
        self.capture_knowledge_button.setToolTip("将当前回答提升为可复核、可关联来源的正式知识")
        self.capture_knowledge_button.clicked.connect(self.capture_last_answer_as_knowledge)
        answer_actions_layout.addWidget(self.capture_knowledge_button)
        answer_actions_layout.addStretch()
        self.answer_actions.hide()
        layout.addWidget(self.answer_actions)

        compose = QFrame()
        compose.setStyleSheet("QFrame{background:#fbfeff;border:1px solid #c5dce8;border-radius:12px;}")
        compose_layout = QVBoxLayout(compose)
        self.prompt = PromptEdit()
        self.prompt.setAccessibleName("问题输入框")
        self.prompt.setPlaceholderText("输入问题，可直接粘贴截图或拖入图片……（Ctrl+Enter 发送）")
        self.prompt.setFixedHeight(92)
        self.prompt.setStyleSheet("border:none;background:transparent;")
        self.prompt.submit.connect(self._send_or_stop)
        self.prompt.imagesPasted.connect(self._receive_pasted_images)
        compose_layout.addWidget(self.prompt)
        self.attachment_list = QListWidget()
        self.attachment_list.setViewMode(QListWidget.IconMode)
        self.attachment_list.setFlow(QListWidget.LeftToRight)
        self.attachment_list.setWrapping(False)
        self.attachment_list.setIconSize(QSize(48, 48))
        self.attachment_list.setGridSize(QSize(150, 64))
        self.attachment_list.setFixedHeight(76)
        self.attachment_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.attachment_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.attachment_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.attachment_list.itemDoubleClicked.connect(lambda _item: self.remove_selected_image())
        self.attachment_list.setToolTip("已添加的图片；双击可移除")
        self.attachment_list.hide()
        compose_layout.addWidget(self.attachment_list)
        row = QHBoxLayout()
        self.attach_button = QPushButton("添加图片")
        self.attach_button.setToolTip("选择图片，也可以直接粘贴截图或把图片拖入编辑框")
        self.attach_button.clicked.connect(self.choose_chat_images)
        row.addWidget(self.attach_button)
        self.remove_image_button = QPushButton("移除图片")
        self.remove_image_button.clicked.connect(self.remove_selected_image)
        self.remove_image_button.hide()
        row.addWidget(self.remove_image_button)
        self.answer_status = QLabel("❄ 就绪")
        self.answer_status.setObjectName("muted")
        row.addWidget(self.answer_status)
        row.addStretch()
        save = QPushButton("保存笔记")
        save.clicked.connect(self.save_answer)
        row.addWidget(save)
        self.send_button = QPushButton("发送 ↗")
        self.send_button.setToolTip("Ctrl+Enter 发送；生成期间点击可停止；Esc 也可停止")
        self.send_button.setObjectName("primary")
        self.send_button.clicked.connect(self._send_or_stop)
        row.addWidget(self.send_button)
        compose_layout.addLayout(row)
        layout.addWidget(compose)
        return panel

    def _right_panel(self) -> QWidget:
        panel, layout = self._panel()
        title = QLabel("证据与资料详情")
        title.setStyleSheet("font-size:15px;font-weight:700;")
        layout.addWidget(title)
        self.evidence_quality = QLabel("尚未生成回答")
        self.evidence_quality.setObjectName("muted")
        self.evidence_quality.setWordWrap(True)
        layout.addWidget(self.evidence_quality)
        self.evidence_list = QListWidget()
        self.evidence_list.setAccessibleName("回答证据列表")
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
        explain_button = QPushButton("为什么使用这些证据")
        explain_button.clicked.connect(self.explain_retrieval)
        layout.addWidget(explain_button)
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

    def choose_chat_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择用于提问的图片",
            str(Path.home()),
            "图片 (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp *.gif);;所有文件 (*)",
        )
        if paths:
            self._add_chat_images(paths)

    def _receive_pasted_images(self, payload: object) -> None:
        if isinstance(payload, (list, tuple)):
            self._add_chat_images([str(value) for value in payload])
            return
        if isinstance(payload, QPixmap):
            payload = payload.toImage()
        if isinstance(payload, QImage) and not payload.isNull():
            try:
                attachment = self._normalize_chat_image(
                    payload, f"粘贴图片-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
                )
            except (OSError, ValueError) as exc:
                self._operation_error("无法粘贴图片", str(exc))
                return
            self._append_pending_image(attachment)
            self.statusBar().showMessage("已从剪贴板添加图片，可继续输入文字后发送", 5000)

    def _add_chat_images(self, paths: list[str]) -> None:
        failures: list[str] = []
        added = 0
        for value in paths:
            if len(self._pending_images) >= 4:
                failures.append("每次最多添加 4 张图片")
                break
            source = Path(value).expanduser().resolve()
            try:
                attachment = self._normalize_chat_image(source, source.name)
            except (OSError, ValueError) as exc:
                failures.append(f"{source.name}：{exc}")
                continue
            self._append_pending_image(attachment)
            added += 1
        if added:
            self.statusBar().showMessage(f"已添加 {added} 张图片，可继续输入文字后发送", 5000)
        if failures:
            self._operation_error("部分图片未添加", "\n".join(dict.fromkeys(failures)))

    def _normalize_chat_image(self, source: Path | QImage, filename: str) -> ImageAttachment:
        if isinstance(source, Path):
            if not source.is_file():
                raise ValueError("文件不存在")
            if source.stat().st_size > 25 * 1024 * 1024:
                raise ValueError("单张图片不能超过 25 MB")
            reader = QImageReader(str(source))
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                raise ValueError(reader.errorString() or "不是可识别的图片")
        else:
            image = QImage(source)
        if image.isNull() or image.width() < 2 or image.height() < 2:
            raise ValueError("图片为空或尺寸无效")
        if image.width() > 4096 or image.height() > 4096:
            image = image.scaled(
                4096, 4096, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        target_dir = self.controller.paths.assets / "chat-attachments"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{uuid.uuid4().hex}.png"
        if not image.save(str(target), "PNG"):
            raise OSError("无法保存规范化图片")
        if target.stat().st_size > 25 * 1024 * 1024:
            target.unlink(missing_ok=True)
            raise ValueError("处理后的图片仍超过 25 MB")
        return ImageAttachment(
            local_path=str(target),
            filename=filename,
            mime_type="image/png",
            width=image.width(),
            height=image.height(),
        )

    def _append_pending_image(self, attachment: ImageAttachment) -> None:
        if len(self._pending_images) >= 4:
            self.statusBar().showMessage("每次最多添加 4 张图片", 5000)
            return
        self._pending_images.append(attachment)
        self._refresh_attachment_list()

    def _refresh_attachment_list(self) -> None:
        self.attachment_list.clear()
        for attachment in self._pending_images:
            item = QListWidgetItem(QIcon(attachment.local_path), attachment.filename)
            item.setData(Qt.UserRole, attachment.local_path)
            item.setToolTip(
                f"{attachment.filename}\n{attachment.width or '?'} × {attachment.height or '?'}\n双击移除"
            )
            self.attachment_list.addItem(item)
        visible = bool(self._pending_images)
        self.attachment_list.setVisible(visible)
        self.remove_image_button.setVisible(visible)
        if visible:
            self.attachment_list.setCurrentRow(0)
            self.answer_status.setText(f"已添加 {len(self._pending_images)} 张图片 · 将发送给视觉模型")
        elif not self._answer_busy:
            self.answer_status.setText("❄ 就绪")

    def remove_selected_image(self) -> None:
        row = self.attachment_list.currentRow()
        if row < 0 or row >= len(self._pending_images):
            return
        removed = self._pending_images.pop(row)
        Path(removed.local_path).unlink(missing_ok=True)
        self._refresh_attachment_list()

    def _clear_pending_images(self, *, delete_files: bool = False) -> None:
        if delete_files:
            for attachment in self._pending_images:
                Path(attachment.local_path).unlink(missing_ok=True)
        self._pending_images = []
        self._refresh_attachment_list()

    def add_url(self) -> None:
        value, accepted = QInputDialog.getText(
            self,
            "导入网页或视频链接",
            "粘贴公开网页、微信、YouTube、B 站、抖音、小红书、X 或音视频直链：",
            text="https://",
        )
        if accepted and value.strip():
            self.start_import([value.strip()])

    def start_import(self, items: list[str]) -> None:
        if self.import_token is not None:
            QMessageBox.information(self, "导入进行中", "请等待当前批次完成，或取消后再导入。")
            return
        token = CancellationToken()
        operation_token = object()
        if not self._begin_db_operation("资料导入", operation_token, requested="导入资料"):
            return
        self.import_items = list(items)
        try:
            job = self.controller.create_ingestion_job(self.import_items)
            self.import_job_id = str(job["id"])
        except (OSError, ValueError) as exc:
            self._finish_db_operation(operation_token)
            self._operation_error("无法创建导入任务", str(exc))
            return
        self.import_token = token
        self._import_operation_token = operation_token
        self.task_list.clear()
        for value in items:
            item = QListWidgetItem(f"等待处理  ·  {Path(value).name or value}")
            item.setData(Qt.UserRole, value)
            item.setData(Qt.UserRole + 1, "pending")
            self.task_list.addItem(item)
        self.left_tabs.setCurrentWidget(self.task_tab)
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.retry_button.setEnabled(False)
        job_id = self.import_job_id

        def execute(signals: WorkerSignals):
            return self.controller.ingest(
                items,
                progress=lambda event: signals.progress.emit(event),
                cancellation=token,
                job_id=job_id,
            )

        worker = Worker(execute)
        worker.signals.progress.connect(self._import_progress)
        worker.signals.result.connect(self._import_complete)
        worker.signals.error.connect(self._import_error)
        worker.signals.finished.connect(
            lambda operation_token=operation_token: self._import_finished(operation_token)
        )
        self.thread_pool.start(worker)
        self.statusBar().showMessage(f"正在后台导入 {len(items)} 份资料…")

    def _import_progress(self, event: ProgressEvent) -> None:
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            if item.data(Qt.UserRole) == event.item:
                stage = INGESTION_STAGE_LABELS.get(event.stage, event.stage or "处理中")
                item.setText(
                    f"{event.percent:>3}%  ·  {stage}\n"
                    f"{event.message}\n{Path(event.item).name or event.item}"
                )
                item.setToolTip(f"当前阶段：{stage}\n{event.message}")
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
                        f"{INGESTION_STATUS_LABELS.get(result.status, result.status)} · "
                        f"{result.chunks} 个知识块{quality_label}"
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
        self.refresh_knowledge()
        self.refresh_conversations()
        self._refresh_status()

    def _import_finished(self, operation_token: object | None = None) -> None:
        if operation_token is not None and self._import_operation_token is not operation_token:
            return
        self.import_token = None
        self.import_job_id = None
        self._import_operation_token = None
        if operation_token is not None:
            self._finish_db_operation(operation_token)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.cancel_button.setEnabled(False)
        QTimer.singleShot(0, self.refresh_ingestion_jobs)

    def _import_error(self, message: str) -> None:
        self._operation_error("导入失败", message)
        QTimer.singleShot(0, self.refresh_ingestion_jobs)

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
            if self.import_job_id:
                try:
                    self.controller.cancel_ingestion_job(self.import_job_id)
                except (OSError, ValueError):
                    pass
            self.statusBar().showMessage("正在安全取消导入…")
            return
        item = self.task_list.currentItem()
        record = item.data(Qt.UserRole + 2) if item else None
        if isinstance(record, dict) and record.get("status") == "queued":
            self.controller.cancel_ingestion_job(str(record["id"]))
            self.refresh_ingestion_jobs()

    def retry_failed(self) -> None:
        if self._background_conflict("继续导入任务"):
            return
        if self.import_token is None:
            item = self.task_list.currentItem()
            record = item.data(Qt.UserRole + 2) if item else None
            if not isinstance(record, dict):
                self.statusBar().showMessage("请选择一条失败或待继续的导入任务", 4000)
                return
            status = str(record.get("status") or "")
            if status not in {"failed", "cancelled", "queued"}:
                self.statusBar().showMessage("这条任务已经完成，无需重试", 4000)
                return
            self._resume_ingestion_job(str(record["id"]), retry=status in {"failed", "cancelled"})
            return
        failed = [
            str(self.task_list.item(i).data(Qt.UserRole))
            for i in range(self.task_list.count())
            if self.task_list.item(i).data(Qt.UserRole + 1) == "failed"
        ]
        if failed:
            self.start_import(failed)
        else:
            self.statusBar().showMessage("没有需要重试的项目", 4000)

    def refresh_ingestion_jobs(self) -> None:
        if self.import_token is not None:
            return
        try:
            jobs = self.controller.ingestion_jobs(limit=100)
        except Exception as exc:
            self.statusBar().showMessage(f"无法读取导入任务：{exc}", 7000)
            return
        self.task_list.clear()
        icons = {
            "completed": "✓", "failed": "✕", "cancelled": "—",
            "queued": "↻", "running": "…",
        }
        labels = {
            "completed": "已完成", "failed": "有失败项", "cancelled": "已取消",
            "queued": "可继续", "running": "处理中",
        }
        for job in jobs:
            status = str(job.get("status") or "queued")
            total = int(job.get("total_items") or 0)
            succeeded = int(job.get("succeeded_items") or 0)
            failed = int(job.get("failed_items") or 0)
            progress = int(job.get("progress_percent") or 0)
            updated = str(job.get("updated_at") or "").replace("T", " ")[:16]
            item = QListWidgetItem(
                f"{icons.get(status, '•')} {labels.get(status, status)} · {progress}%\n"
                f"{succeeded}/{total} 成功{f' · {failed} 失败' if failed else ''} · {updated}"
            )
            item.setData(Qt.UserRole, str(job.get("id") or ""))
            item.setData(Qt.UserRole + 1, status)
            item.setData(Qt.UserRole + 2, job)
            item.setToolTip(str(job.get("message") or job.get("error") or "双击可继续或重试"))
            self.task_list.addItem(item)
        if self.task_list.count():
            self.task_list.setCurrentRow(0)
        self._update_task_controls()

    def _update_task_controls(self, _current: object = None, _previous: object = None) -> None:
        if self.import_token is not None:
            self.pause_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            self.retry_button.setEnabled(False)
            return
        item = self.task_list.currentItem()
        record = item.data(Qt.UserRole + 2) if item else None
        status = str(record.get("status") or "") if isinstance(record, dict) else ""
        self.pause_button.setEnabled(False)
        self.cancel_button.setEnabled(status == "queued")
        self.retry_button.setEnabled(status in {"failed", "cancelled", "queued"})
        self.retry_button.setText("继续任务" if status == "queued" else "重试失败项")

    def _resume_ingestion_job(self, job_id: str, *, retry: bool) -> None:
        if self.import_token is not None:
            return
        operation_token = object()
        if not self._begin_db_operation(
            "资料导入",
            operation_token,
            requested="继续导入任务",
        ):
            return
        try:
            record = self.controller.ingestion_job(job_id)
        except (OSError, ValueError) as exc:
            self._finish_db_operation(operation_token)
            self._operation_error("无法读取任务", str(exc))
            return
        items = [
            str(value.get("source")) for value in record.get("items", [])
            if isinstance(value, dict)
            and str(value.get("status")) in ({"failed", "cancelled"} if retry else {"queued"})
        ]
        if not items:
            self._finish_db_operation(operation_token)
            self.statusBar().showMessage("该任务没有可继续的资料", 5000)
            return
        self.import_items = items
        self.import_job_id = job_id
        self.import_token = CancellationToken()
        self._import_operation_token = operation_token
        self.task_list.clear()
        for value in items:
            item = QListWidgetItem(f"等待处理  ·  {Path(value).name or value}")
            item.setData(Qt.UserRole, value)
            item.setData(Qt.UserRole + 1, "pending")
            self.task_list.addItem(item)
        self.left_tabs.setCurrentWidget(self.task_tab)
        self._update_task_controls()
        token = self.import_token

        def execute(signals: WorkerSignals):
            function = self.controller.retry_ingestion_job if retry else self.controller.resume_ingestion_job
            return function(
                job_id,
                progress=lambda event: signals.progress.emit(event),
                cancellation=token,
            )

        worker = Worker(execute)
        worker.signals.progress.connect(self._import_progress)
        worker.signals.result.connect(self._import_complete)
        worker.signals.error.connect(self._import_error)
        worker.signals.finished.connect(
            lambda operation_token=operation_token: self._import_finished(operation_token)
        )
        self.thread_pool.start(worker)
        self.statusBar().showMessage(f"正在{'重试' if retry else '继续'} {len(items)} 份资料…")

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

    def refresh_knowledge(self, _value: object = None) -> None:
        if not hasattr(self, "knowledge_list"):
            return
        selected = self.knowledge_list.currentItem()
        selected_id = str(selected.data(Qt.UserRole).get("id")) if selected and isinstance(
            selected.data(Qt.UserRole), dict
        ) else ""
        query = self.knowledge_search.text().strip()
        item_type = str(self.knowledge_type_filter.currentData() or "")
        status = str(self.knowledge_status_filter.currentData() or "")
        try:
            rows = self.controller.knowledge_items(
                query=query,
                item_type=item_type or None,
                status=status or None,
                limit=500,
            )
            health = self.controller.knowledge_health()
        except Exception as exc:
            self.knowledge_summary.setText(f"无法读取知识网络：{exc}")
            return
        self.knowledge_list.clear()
        for record in rows:
            record_type = str(record.get("item_type") or record.get("type") or "analysis")
            record_status = str(record.get("status") or "draft")
            maturity = str(record.get("maturity") or "unreviewed")
            title = str(record.get("title") or "未命名知识")
            summary = str(record.get("summary") or "").strip()
            text = (
                f"{KNOWLEDGE_TYPE_LABELS.get(record_type, record_type)} · {title}\n"
                f"{KNOWLEDGE_STATUS_LABELS.get(record_status, record_status)} · "
                f"{KNOWLEDGE_MATURITY_LABELS.get(maturity, maturity)}"
            )
            if summary:
                text += f"\n{summary[:80]}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, record)
            aliases = record.get("aliases") or []
            item.setToolTip(summary or ("别名：" + "、".join(str(value) for value in aliases) if aliases else title))
            self.knowledge_list.addItem(item)
            if str(record.get("id")) == selected_id:
                self.knowledge_list.setCurrentItem(item)
        counts = health.get("counts") if isinstance(health.get("counts"), dict) else {}
        self.knowledge_summary.setText(
            f"知识 {int(counts.get('items', len(rows)))} 条 · "
            f"待复核 {int(counts.get('needs_review', 0))} · "
            f"过期 {int(counts.get('stale', 0))} · "
            f"提醒 {int(health.get('issue_count', len(health.get('issues') or [])))}"
        )

    def _show_knowledge_item(self, item: QListWidgetItem | None) -> None:
        if not item:
            return
        record = item.data(Qt.UserRole)
        if not isinstance(record, dict):
            return
        item_id = str(record.get("id") or "")
        try:
            detail = self.controller.knowledge_item(item_id) if item_id else record
        except (OSError, ValueError):
            detail = record
        item_type = str(detail.get("item_type") or detail.get("type") or "analysis")
        status = str(detail.get("status") or "draft")
        maturity = str(detail.get("maturity") or "unreviewed")
        aliases = "、".join(str(value) for value in detail.get("aliases") or []) or "无"
        tags = "、".join(str(value) for value in detail.get("tags") or []) or "无"
        relation_lines = []
        for relation in detail.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            relation_lines.append(
                f"- **{KNOWLEDGE_RELATION_LABELS.get(str(relation.get('relation_type') or relation.get('type') or ''), str(relation.get('relation_type') or relation.get('type') or '关联'))}** "
                f"{relation.get('related_title') or relation.get('target_title') or relation.get('source_title') or '相关知识'}"
            )
        body = str(detail.get("body") or "").strip()
        summary = str(detail.get("summary") or "").strip()
        markdown = (
            f"# {detail.get('title') or '未命名知识'}\n\n"
            f"- 类型：{KNOWLEDGE_TYPE_LABELS.get(item_type, item_type)}\n"
            f"- 状态：{KNOWLEDGE_STATUS_LABELS.get(status, status)}\n"
            f"- 成熟度：{KNOWLEDGE_MATURITY_LABELS.get(maturity, maturity)}\n"
            f"- 别名：{aliases}\n- 标签：{tags}\n\n"
            f"## 摘要\n\n{summary or '尚未填写摘要。'}\n\n"
            + (f"## 正文\n\n{body}\n\n" if body else "")
            + ("## 知识关系\n\n" + "\n".join(relation_lines) if relation_lines else "")
        )
        self.preview.setMarkdown(markdown)

    def _selected_knowledge_record(self) -> dict[str, object] | None:
        item = self.knowledge_list.currentItem() if hasattr(self, "knowledge_list") else None
        value = item.data(Qt.UserRole) if item else None
        return value if isinstance(value, dict) else None

    def _knowledge_menu(self, position) -> None:
        item = self.knowledge_list.itemAt(position)
        if not item:
            return
        self.knowledge_list.setCurrentItem(item)
        record = self._selected_knowledge_record()
        if not record:
            return
        menu = QMenu(self)
        review = menu.addAction("标记为需要复核")
        current = menu.addAction("确认当前有效")
        stale = menu.addAction("标记为可能过期")
        archived = menu.addAction("归档")
        menu.addSeparator()
        relation = menu.addAction("建立知识关系…")
        menu.addSeparator()
        delete = menu.addAction("删除正式知识…")
        chosen = menu.exec(self.knowledge_list.mapToGlobal(position))
        if chosen == review:
            self.set_selected_knowledge_status("needs-review")
        elif chosen == current:
            self.set_selected_knowledge_status("current")
        elif chosen == stale:
            self.set_selected_knowledge_status("stale")
        elif chosen == archived:
            self.set_selected_knowledge_status("archived")
        elif chosen == relation:
            self.relate_selected_knowledge()
        elif chosen == delete:
            self.delete_selected_knowledge()

    def set_selected_knowledge_status(self, status: str) -> None:
        record = self._selected_knowledge_record()
        if not record:
            return
        try:
            self.controller.update_knowledge_item(str(record["id"]), status=status)
        except (OSError, ValueError) as exc:
            self._operation_error("状态更新失败", str(exc))
            return
        self.refresh_knowledge()
        self.statusBar().showMessage(f"已标记为：{KNOWLEDGE_STATUS_LABELS.get(status, status)}", 5000)

    def relate_selected_knowledge(self) -> None:
        source = self._selected_knowledge_record()
        if not source:
            return
        candidates = [
            self.knowledge_list.item(index).data(Qt.UserRole)
            for index in range(self.knowledge_list.count())
            if self.knowledge_list.item(index) is not self.knowledge_list.currentItem()
            and isinstance(self.knowledge_list.item(index).data(Qt.UserRole), dict)
        ]
        if not candidates:
            QMessageBox.information(self, "没有可关联的知识", "至少需要两条正式知识才能建立关系。")
            return
        labels = [
            f"{KNOWLEDGE_TYPE_LABELS.get(str(value.get('item_type')), str(value.get('item_type')))} · {value.get('title')}"
            for value in candidates
        ]
        selected, accepted = QInputDialog.getItem(self, "关联其他知识", "目标知识：", labels, 0, False)
        if not accepted:
            return
        target = candidates[labels.index(selected)]
        relation_labels = {
            "支持（supports）": "supports",
            "扩展（extends）": "extends",
            "冲突（contradicts）": "contradicts",
            "取代（supersedes）": "supersedes",
            "提出新问题（opens）": "opens",
        }
        relation_label, accepted = QInputDialog.getItem(
            self, "关系类型", "当前知识与目标知识的关系：", list(relation_labels), 0, False
        )
        if not accepted:
            return
        try:
            self.controller.create_knowledge_relation(
                str(source["id"]), str(target["id"]), relation_labels[relation_label]
            )
        except (OSError, ValueError) as exc:
            self._operation_error("知识关联失败", str(exc))
            return
        self.refresh_knowledge()
        self.statusBar().showMessage("知识关系已建立", 5000)

    def delete_selected_knowledge(self) -> None:
        record = self._selected_knowledge_record()
        if not record:
            return
        choice = QMessageBox.question(
            self,
            "删除正式知识",
            f"确定将“{record.get('title') or '未命名知识'}”及其关系移到知识回收站吗？\n\n"
            "之后可以从回收站完整恢复；原始资料不会被删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        try:
            self.controller.delete_knowledge_item(str(record["id"]))
        except (OSError, RuntimeError, ValueError) as exc:
            self._operation_error("删除失败", str(exc))
            return
        self.refresh_knowledge()
        self.preview.clear()
        self.statusBar().showMessage("已移到知识回收站，可随时恢复", 6000)

    @staticmethod
    def _answer_summary(markdown: str) -> str:
        paragraphs = []
        for value in markdown.split("\n\n"):
            cleaned = value.strip().lstrip("#*- ").strip()
            if cleaned and not cleaned.startswith("["):
                paragraphs.append(cleaned.replace("\n", " "))
            if len(" ".join(paragraphs)) >= 240:
                break
        return " ".join(paragraphs)[:320]

    def capture_last_answer_as_knowledge(self) -> None:
        markdown = self.last_answer_markdown.strip()
        if not markdown:
            QMessageBox.information(self, "没有可沉淀的回答", "请先完成一次问答，再将有复用价值的内容沉淀为知识。")
            return
        question = self.last_question.strip()
        suggested_title = question[:100] if question else self._answer_summary(markdown)[:100]
        evidence_document_ids: list[str] = []
        for value in self.evidence_by_id.values():
            if isinstance(value, dict):
                source = value.get("source") if isinstance(value.get("source"), dict) else {}
                document_id = value.get("document_id") or source.get("document_id")
            else:
                source = getattr(value, "source", None)
                document_id = getattr(value, "document_id", None) or getattr(source, "document_id", None)
            if document_id and str(document_id) not in evidence_document_ids:
                evidence_document_ids.append(str(document_id))
        dialog = KnowledgeCaptureDialog(
            suggested_title=suggested_title or "AI静静知识",
            suggested_summary=self._answer_summary(markdown),
            evidence_count=len(evidence_document_ids),
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            record = self.controller.capture_answer_as_knowledge(
                markdown=markdown,
                question=question,
                conversation_id=self.conversation_id,
                answer_id=self.last_answer_id,
                evidence_document_ids=evidence_document_ids,
                **dialog.values(),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._operation_error("知识沉淀失败", str(exc))
            return
        self.refresh_knowledge()
        self.left_tabs.setCurrentWidget(self.knowledge_tab)
        for index in range(self.knowledge_list.count()):
            current = self.knowledge_list.item(index)
            payload = current.data(Qt.UserRole)
            if isinstance(payload, dict) and str(payload.get("id")) == str(record.get("id")):
                self.knowledge_list.setCurrentItem(current)
                break
        self.statusBar().showMessage("已沉淀为正式知识，并关联当前来源证据", 7000)

    def show_knowledge_health(self) -> None:
        try:
            report = self.controller.knowledge_health()
        except (OSError, ValueError) as exc:
            self._operation_error("知识体检失败", str(exc))
            return
        KnowledgeHealthDialog(report, self).exec()

    def show_knowledge_trash(self) -> None:
        dialog = KnowledgeTrashDialog(self.controller, self)
        dialog.exec()
        if dialog.restored_ids:
            self.refresh_knowledge()
            self.statusBar().showMessage(
                f"已从回收站恢复 {len(dialog.restored_ids)} 条正式知识", 6000
            )

    def refresh_conversations(self, _value: object = None) -> None:
        """Refresh the persisted chat list without disturbing the open chat."""
        selected_id = self._selected_conversation_id()
        self.history_list.clear()
        try:
            rows = self.controller.conversations(
                query=self.history_search.text().strip(), limit=200, offset=0
            )
        except Exception as exc:
            self.statusBar().showMessage(f"无法读取对话记录：{exc}", 8000)
            return
        for row in rows:
            conversation_id = str(row.get("conversation_id") or row.get("id") or "")
            if not conversation_id:
                continue
            title = str(row.get("title") or "未命名对话")
            message_count = int(row.get("message_count") or 0)
            updated = str(row.get("updated_at") or "").replace("T", " ")[:16]
            preview = str(row.get("preview") or row.get("last_message") or "").strip()
            suffix = f"{message_count} 条消息"
            if updated:
                suffix += f" · {updated}"
            text = f"{title}\n{suffix}"
            if preview and preview.casefold() not in title.casefold():
                text += f"\n{preview[:72]}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, conversation_id)
            item.setToolTip(preview or title)
            self.history_list.addItem(item)
            if conversation_id == selected_id or conversation_id == self.conversation_id:
                self.history_list.setCurrentItem(item)

    def _selected_conversation_id(self) -> str | None:
        item = self.history_list.currentItem() if hasattr(self, "history_list") else None
        return str(item.data(Qt.UserRole)) if item and item.data(Qt.UserRole) else None

    def open_selected_conversation(self, item: object = None) -> None:
        if isinstance(item, QListWidgetItem):
            self.history_list.setCurrentItem(item)
        conversation_id = self._selected_conversation_id()
        if not conversation_id:
            return
        if self._answer_busy:
            self.statusBar().showMessage("请先停止当前回答，再切换对话。", 5000)
            return
        try:
            record = self.controller.conversation_record(conversation_id)
        except (OSError, ValueError) as exc:
            self._operation_error("无法打开对话", str(exc))
            return
        answers = {
            str(value.get("answer_message_id")): value
            for value in record.get("answers", [])
            if isinstance(value, dict) and value.get("answer_message_id")
        }
        entries: list[dict[str, object]] = []
        last_answer: dict[str, object] | None = None
        last_partial = ""
        last_user_question = ""
        for message in record.get("messages", []):
            if not isinstance(message, dict) or message.get("role") == "system":
                continue
            role = str(message.get("role") or "assistant")
            entry: dict[str, object] = {
                "role": role,
                "content": str(message.get("content") or ""),
            }
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if role == "user":
                last_user_question = str(message.get("content") or "")
                raw_images = metadata.get("image_attachments", [])
                if isinstance(raw_images, list):
                    entry["images"] = [value for value in raw_images if isinstance(value, dict)]
            answer = answers.get(str(message.get("message_id") or ""))
            if role == "assistant" and answer:
                evidence = answer.get("evidence") if isinstance(answer.get("evidence"), list) else []
                entry["evidence_ids"] = [
                    str(value.get("evidence_id"))
                    for value in evidence if isinstance(value, dict) and value.get("evidence_id")
                ]
                entry["answer_id"] = str(answer.get("answer_id") or "")
                last_answer = answer
                last_partial = ""
            elif role == "assistant":
                last_answer = None
                last_partial = str(message.get("content") or "")
            entries.append(entry)
        self.conversation_id = conversation_id
        self._chat_entries = entries
        self._stream_text = ""
        self.last_answer = None
        self.last_answer_id = str(last_answer.get("answer_id")) if last_answer else None
        self.last_answer_markdown = (
            str(last_answer.get("markdown") or "") if last_answer else last_partial
        )
        self.last_retrieval_info = (
            dict(last_answer.get("retrieval_info") or {}) if last_answer else {}
        )
        self.last_question = (
            str(last_answer.get("question") or "") if last_answer else last_user_question
        )
        self._load_record_evidence(last_answer)
        feedback = last_answer.get("feedback") if last_answer and isinstance(last_answer.get("feedback"), dict) else {}
        self.helpful_button.setEnabled(feedback.get("rating") != "up")
        self.unhelpful_button.setEnabled(feedback.get("rating") != "down")
        self.answer_actions.setVisible(bool(self.last_answer_markdown))
        self.answer_status.setText(
            f"已打开：{record.get('title') or '未命名对话'} · {len(entries)} 条消息"
        )
        self._render_chat()

    def _load_record_evidence(self, answer: dict[str, object] | None) -> None:
        self.evidence_by_id.clear()
        self.evidence_list.clear()
        if not answer:
            self.evidence_quality.setText("此对话尚无已保存回答")
            self.preview.clear()
            return
        for value in answer.get("evidence", []):
            if not isinstance(value, dict) or not value.get("evidence_id"):
                continue
            evidence_id = str(value["evidence_id"])
            self.evidence_by_id[evidence_id] = value
            source = value.get("source") if isinstance(value.get("source"), dict) else {}
            location = self._dict_location(source)
            item = QListWidgetItem(
                f"[{evidence_id}] {value.get('title') or '证据'}\n"
                f"{location or source.get('media_type') or value.get('source_kind') or '知识库'}"
                f" · 综合分 {float(value.get('score') or 0):.3f}"
            )
            item.setData(Qt.UserRole, evidence_id)
            self.evidence_list.addItem(item)
        quality = self.last_retrieval_info.get("evidence_quality")
        label = self._quality_label(quality.get("level")) if isinstance(quality, dict) else None
        self.evidence_quality.setText(
            f"策略：{self._retrieval_label(self.last_retrieval_info.get('retrieval_strategy'))} · "
            f"证据：{label or ('已引用' if self.evidence_list.count() else '不足')}"
            + (f"\n{quality.get('explanation')}" if isinstance(quality, dict) and quality.get("explanation") else "")
        )
        if self.evidence_list.count():
            self.evidence_list.setCurrentRow(0)

    @staticmethod
    def _dict_location(source: dict[str, object]) -> str:
        values: list[str] = []
        if source.get("page_number") is not None:
            values.append(f"P{source['page_number']}")
        if source.get("slide_number") is not None:
            values.append(f"S{source['slide_number']}")
        if source.get("timestamp_start") is not None:
            values.append(f"{float(source['timestamp_start']):g}s")
        if source.get("section"):
            values.append(str(source["section"]))
        return " / ".join(values)

    @staticmethod
    def _quality_label(level: object) -> str:
        return {
            "well_supported": "证据充分",
            "partially_supported": "部分有据",
            "limited": "引用有限",
            "insufficient": "证据不足",
            "image_only": "仅基于图片",
        }.get(str(level or ""), str(level or ""))

    @staticmethod
    def _retrieval_label(strategy: object) -> str:
        return {
            "focused": "聚焦检索",
            "full_context": "小文档全文",
            "hierarchical": "长文档分层摘要",
        }.get(str(strategy or "focused"), str(strategy or "聚焦检索"))

    def _conversation_menu(self, position) -> None:
        item = self.history_list.itemAt(position)
        if not item:
            return
        self.history_list.setCurrentItem(item)
        menu = QMenu(self)
        open_action = menu.addAction("打开对话")
        rename_action = menu.addAction("重命名…")
        export_action = menu.addAction("导出 Markdown…")
        menu.addSeparator()
        delete_action = menu.addAction("删除对话…")
        chosen = menu.exec(self.history_list.mapToGlobal(position))
        if chosen == open_action:
            self.open_selected_conversation()
        elif chosen == rename_action:
            self.rename_selected_conversation()
        elif chosen == export_action:
            self.export_selected_conversation()
        elif chosen == delete_action:
            self.delete_selected_conversation()

    def rename_selected_conversation(self) -> None:
        conversation_id = self._selected_conversation_id()
        item = self.history_list.currentItem()
        if not conversation_id or not item:
            return
        current = item.text().splitlines()[0]
        title, accepted = QInputDialog.getText(self, "重命名对话", "新标题：", text=current)
        if accepted and title.strip():
            self.controller.rename_conversation(conversation_id, title.strip())
            self.refresh_conversations()

    def delete_selected_conversation(self) -> None:
        conversation_id = self._selected_conversation_id()
        if not conversation_id:
            return
        choice = QMessageBox.question(
            self, "删除对话", "确定删除这段对话及其回答、引用和反馈吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        self.controller.delete_conversation(conversation_id)
        if self.conversation_id == conversation_id:
            self.new_chat()
        self.refresh_conversations()
        self.statusBar().showMessage("对话已删除；知识资料不受影响", 6000)

    def export_selected_conversation(self) -> None:
        conversation_id = self._selected_conversation_id()
        if not conversation_id:
            return
        suggested = str(self.controller.paths.notes / "AI静静-对话.md")
        destination, _ = QFileDialog.getSaveFileName(
            self, "导出对话", suggested, "Markdown (*.md)"
        )
        if not destination:
            return
        try:
            path = self.controller.export_conversation(conversation_id, destination)
        except (OSError, ValueError) as exc:
            self._operation_error("导出失败", str(exc))
            return
        self.statusBar().showMessage(f"对话已导出：{path}", 10000)

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
        transcript_action = menu.addAction("播放与校订转写…")
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
        elif chosen == transcript_action:
            self.open_transcript_editor()
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
        operation_token = object()
        if not self._begin_db_operation(
            "知识库搜索",
            operation_token,
            requested="搜索知识库",
        ):
            return
        self._search_operation_token = operation_token
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
        worker.signals.finished.connect(
            lambda operation_token=operation_token: self._finish_search(operation_token)
        )
        self.thread_pool.start(worker)

    def _finish_search(self, operation_token: object) -> None:
        if self._search_operation_token is not operation_token:
            return
        self._search_operation_token = None
        self._finish_db_operation(operation_token)

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
            self.stop_answer()
            return
        question = self.prompt.toPlainText().strip()
        images = list(self._pending_images)
        if not question and not images:
            return
        operation_token = object()
        if not self._begin_db_operation(
            "回答生成",
            operation_token,
            requested="生成回答",
        ):
            return
        self._answer_operation_token = operation_token
        effective_question = question or "请仔细分析这张图片，并结合知识库说明其中的内容。"
        self.last_question = effective_question
        self.prompt.clear()
        self._inflight_images = images
        self._clear_pending_images()
        self._stream_text = ""
        self._answer_cancelled.clear()
        self.answer_actions.hide()
        self._append_user(question, images)
        self._set_answer_busy(True)
        self.answer_status.setText("正在检索选定知识…")
        model_id = str(self.model_combo.currentData() or "local-extractive")
        conversation = self.conversation_id or f"conv-{uuid.uuid4().hex}"
        self.conversation_id = conversation
        deep = self.deep_analysis.isChecked()
        document_ids, collections = self._active_scope()

        def execute(signals: WorkerSignals):
            def on_delta(value: str) -> None:
                if self._answer_cancelled.is_set():
                    raise RuntimeError("回答已停止")
                signals.delta.emit(value)

            return self.controller.ask(
                effective_question,
                conversation_id=conversation,
                model_id=model_id,
                deep_analysis=deep,
                document_ids=document_ids,
                collections=collections,
                progress=lambda stage, message: signals.progress.emit((stage, message)),
                delta_callback=on_delta,
                image_attachments=images,
            )

        worker = Worker(execute)
        worker.signals.progress.connect(lambda value: self.answer_status.setText(value[1]))
        worker.signals.delta.connect(self._answer_delta)
        worker.signals.result.connect(
            lambda answer, operation_token=operation_token: self._answer_complete(
                answer, operation_token
            )
        )
        worker.signals.error.connect(
            lambda message, operation_token=operation_token: self._answer_error(
                message,
                question=question,
                images=images,
                operation_token=operation_token,
            )
        )
        worker.signals.finished.connect(
            lambda operation_token=operation_token: self._finish_answer_request(
                operation_token
            )
        )
        self._answer_worker = worker
        self.thread_pool.start(worker)

    def _send_or_stop(self) -> None:
        if self._answer_busy:
            self.stop_answer()
        else:
            self.ask()

    def stop_answer(self) -> None:
        if not self._answer_busy:
            return
        self._answer_cancelled.set()
        self.send_button.setEnabled(False)
        self.send_button.setText("正在停止…")
        self.answer_status.setText("正在安全停止回答；已生成内容会保留")

    def _answer_delta(self, text: str) -> None:
        if self._answer_cancelled.is_set():
            return
        self._stream_text += text
        if self._stream_render_scheduled:
            return
        self._stream_render_scheduled = True
        QTimer.singleShot(35, self._flush_stream_render)

    def _flush_stream_render(self) -> None:
        self._stream_render_scheduled = False
        self._render_chat()

    def _set_answer_busy(self, busy: bool) -> None:
        self._answer_busy = busy
        self.send_button.setEnabled(True)
        self.send_button.setText("停止生成" if busy else "发送 ↗")

    def _finish_answer_request(self, operation_token: object | None = None) -> None:
        """Restore the composer from deterministic result/error callbacks and as a fallback."""
        operation_token = operation_token or self._answer_operation_token
        if (
            operation_token is not None
            and self._answer_operation_token is not operation_token
        ):
            return
        self._set_answer_busy(False)
        self._answer_worker = None
        self._answer_operation_token = None
        if operation_token is not None:
            self._finish_db_operation(operation_token)
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

    def _append_user(self, text: str, images: list[ImageAttachment] | None = None) -> None:
        self._chat_entries.append(
            {"role": "user", "content": text or "请分析这些图片", "images": list(images or [])}
        )
        self._render_chat()

    def _render_chat(self) -> None:
        if not self._chat_entries and not self._answer_busy and not self._stream_text:
            self.chat.setHtml(self._welcome_html())
            return
        parts: list[str] = ["<div style='max-width:920px;margin:0 auto'>"]
        for entry in self._chat_entries:
            role = str(entry.get("role") or "assistant")
            content = str(entry.get("content") or "")
            if role == "user":
                images = entry.get("images") if isinstance(entry.get("images"), list) else []
                attachments = "".join(self._chat_image_html(item) for item in images)
                parts.append(
                    "<div style='margin:18px 8px 6px 22%;text-align:right;color:#6a8495;font-size:11px'>你</div>"
                    "<div style='margin:0 8px 18px 22%;background:#1b5578;color:white;padding:12px 15px;border-radius:14px'>"
                    f"{attachments}{html.escape(content).replace(chr(10), '<br>')}</div>"
                )
            else:
                evidence_ids = entry.get("evidence_ids")
                rendered = self._answer_html(
                    content,
                    [str(value) for value in evidence_ids] if isinstance(evidence_ids, list) else [],
                )
                parts.append(
                    "<div style='margin:8px 20% 6px 8px;color:#2d789e;font-size:11px;font-weight:600'>"
                    "✦ AI静静 · AI 生成</div>"
                    "<div style='margin:0 20% 20px 8px;background:#fbfeff;border:1px solid #c7e0ec;"
                    f"padding:14px;border-radius:14px'>{rendered}</div>"
                )
        if self._answer_busy:
            draft = self._stream_text or "正在检索资料并组织回答……"
            rendered = _markdown_html(draft)
            parts.append(
                "<div style='margin:8px 20% 6px 8px;color:#2d789e;font-size:11px;font-weight:600'>"
                "✦ AI静静 · 正在生成</div>"
                "<div style='margin:0 20% 20px 8px;background:#fbfeff;border:1px solid #87bdd4;"
                f"padding:14px;border-radius:14px'>{rendered}</div>"
            )
        parts.append("</div>")
        self.chat.setHtml("".join(parts))
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    @staticmethod
    def _chat_image_html(value: object) -> str:
        if isinstance(value, ImageAttachment):
            path, filename = value.local_path, value.filename
        elif isinstance(value, dict):
            path = str(value.get("local_path") or "")
            filename = str(value.get("filename") or Path(path).name)
        else:
            return ""
        if not path or not Path(path).is_file():
            return f"<div style='font-size:11px;color:#d8eff9'>图片不可用：{html.escape(filename)}</div>"
        url = html.escape(QUrl.fromLocalFile(path).toString())
        return (
            "<div style='display:inline-block;margin:4px 4px 8px 0'>"
            f"<img src='{url}' width='150'/><br>"
            f"<span style='font-size:10px;color:#d8eff9'>{html.escape(filename)}</span></div>"
        )

    @staticmethod
    def _answer_html(markdown: str, evidence_ids: list[str]) -> str:
        rendered = _markdown_html(markdown)
        for evidence_id in evidence_ids:
            marker = f"[{evidence_id}]"
            rendered = rendered.replace(
                marker, f"<a href='aijj://citation/{evidence_id}'>{marker}</a>"
            )
        return rendered

    def _answer_complete(self, answer, operation_token: object | None = None) -> None:
        self._finish_answer_request(operation_token)
        self._inflight_images = []
        self._stream_text = ""
        self.last_answer = answer
        self.last_answer_id = answer.answer_id
        self.last_answer_markdown = answer.markdown
        self.last_retrieval_info = dict(answer.retrieval_info or {})
        self.conversation_id = answer.conversation_id
        self._chat_entries.append(
            {
                "role": "assistant",
                "content": answer.markdown,
                "evidence_ids": [item.evidence_id for item in answer.evidence],
                "answer_id": answer.answer_id,
            }
        )
        self._render_chat()
        citation_sources = {
            citation.document_id
            or citation.original_uri
            or citation.local_path
            or citation.title
            for citation in answer.citations
        }
        quality = answer.retrieval_info.get("evidence_quality") or {}
        level = self._quality_label(quality.get("label") or quality.get("level"))
        citation_coverage = quality.get("citation_coverage")
        if answer.citations:
            coverage_text = (
                f" · 引用覆盖 {float(citation_coverage):.0%}"
                if isinstance(citation_coverage, (int, float)) else ""
            )
            self.answer_status.setText(
                f"{answer.model} · {level or '已核验引用'} · 实际引用 {len(answer.citations)} 条证据"
                f" / {len(citation_sources)} 份资料{coverage_text}"
            )
        else:
            image_count = int(answer.retrieval_info.get("image_count") or 0)
            if image_count:
                self.answer_status.setText(
                    f"{answer.model} · 已理解 {image_count} 张图片 · 图片观察不计知识库引用"
                )
            else:
                self.answer_status.setText(
                    f"{answer.model} · 知识库证据不足或未采用引用"
                )
        self.evidence_quality.setText(
            f"策略：{self._retrieval_label(answer.retrieval_info.get('retrieval_strategy'))} · "
            f"证据：{level or ('有引用' if answer.citations else '不足')}"
            + (f"\n{quality.get('explanation')}" if quality.get("explanation") else "")
        )
        self.evidence_by_id.clear()
        self.evidence_list.clear()
        for evidence in answer.evidence:
            self.evidence_by_id[evidence.evidence_id] = evidence
            item = QListWidgetItem(
                f"[{evidence.evidence_id}] {evidence.title}\n"
                f"{evidence.locator() or evidence.source.media_type} · 综合分 {evidence.score:.3f}"
            )
            item.setData(Qt.UserRole, evidence.evidence_id)
            self.evidence_list.addItem(item)
        if self.evidence_list.count():
            self.evidence_list.setCurrentRow(0)
        self.answer_actions.show()
        self.helpful_button.setEnabled(True)
        self.unhelpful_button.setEnabled(True)
        self.refresh_conversations()
        self._refresh_status()

    def _answer_error(
        self,
        message: str,
        *,
        question: str = "",
        images: list[ImageAttachment] | None = None,
        operation_token: object | None = None,
    ) -> None:
        self._finish_answer_request(operation_token)
        stopped = self._answer_cancelled.is_set() or "已停止" in message or "cancel" in message.casefold()
        if stopped:
            if self._stream_text.strip():
                partial = self._stream_text + "\n\n*回答已由用户停止。*"
                self._chat_entries.append(
                    {"role": "assistant", "content": partial, "partial": True}
                )
                self.last_answer_markdown = self._stream_text
                self.last_answer_id = None
                if self.conversation_id:
                    try:
                        self.controller.save_partial_answer(self.conversation_id, partial)
                    except (OSError, ValueError):
                        pass
            self._stream_text = ""
            self._render_chat()
            self.answer_status.setText("回答已停止，可修改问题或重新生成")
            self.answer_actions.setVisible(bool(self.last_answer_markdown))
            self.helpful_button.setEnabled(False)
            self.unhelpful_button.setEnabled(False)
            self._answer_cancelled.clear()
            self.refresh_conversations()
            return
        if question and not self.prompt.toPlainText().strip():
            self.prompt.setPlainText(question)
        if images:
            for attachment in images:
                if all(item.local_path != attachment.local_path for item in self._pending_images):
                    self._pending_images.append(attachment)
            self._refresh_attachment_list()
        self._inflight_images = []
        self._stream_text = ""
        self.answer_status.setText("回答失败")
        self._chat_entries.append({"role": "assistant", "content": f"未能生成回答：{message}"})
        self._render_chat()
        self.refresh_conversations()
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
        elif isinstance(value, dict):
            source = value.get("source") if isinstance(value.get("source"), dict) else {}
            location = self._dict_location(source)
            target = source.get("original_uri") or source.get("local_path") or ""
            self.preview.setMarkdown(
                f"# {value.get('title') or '证据'}\n\n"
                f"**{location or source.get('media_type') or '知识库证据'}**\n\n"
                f"{value.get('content') or ''}\n\n---\n\n来源：{target}"
            )

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
            if isinstance(value, dict):
                source = value.get("source") if isinstance(value.get("source"), dict) else {}
                target = source.get("original_uri") or source.get("local_path")
            else:
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

    def copy_last_answer(self) -> None:
        if not self.last_answer_markdown:
            self.statusBar().showMessage("当前没有可复制的回答", 4000)
            return
        QApplication.clipboard().setText(self.last_answer_markdown)
        self.statusBar().showMessage("回答已复制到剪贴板", 4000)

    def regenerate_answer(self) -> None:
        if self._answer_busy:
            return
        if not self.last_question:
            self.statusBar().showMessage("当前没有可重新生成的问题", 4000)
            return
        self.prompt.setPlainText(self.last_question)
        self.ask()

    def save_answer_feedback(self, rating: str) -> None:
        if not self.last_answer_id:
            self.statusBar().showMessage("这条回答尚未保存，无法提交反馈", 4000)
            return
        comment = ""
        if rating == "down":
            comment, accepted = QInputDialog.getMultiLineText(
                self, "帮助 AI静静改进", "哪里不准确，或缺少了什么？（可留空）"
            )
            if not accepted:
                return
        try:
            self.controller.save_answer_feedback(self.last_answer_id, rating, comment.strip())
        except (OSError, ValueError) as exc:
            self._operation_error("反馈保存失败", str(exc))
            return
        self.helpful_button.setEnabled(rating != "up")
        self.unhelpful_button.setEnabled(rating != "down")
        self.statusBar().showMessage("感谢反馈；它已保存在本地，可用于后续质量评估", 6000)

    def explain_retrieval(self) -> None:
        info = self.last_retrieval_info
        if not info:
            QMessageBox.information(self, "证据说明", "当前还没有回答检索记录。")
            return
        strategy = self._retrieval_label(info.get("retrieval_strategy"))
        rewritten = str(info.get("retrieval_query") or info.get("rewritten_query") or "")
        quality = info.get("evidence_quality") if isinstance(info.get("evidence_quality"), dict) else {}
        lines = [
            f"检索策略：{strategy}",
            f"实际检索词：{rewritten or self.last_question or '原问题'}",
            f"候选证据：{info.get('candidate_count', info.get('retrieved_count', '—'))}",
            f"最终采用：{info.get('evidence_count', self.evidence_list.count())}",
        ]
        if quality:
            lines.extend(
                [
                    f"证据等级：{self._quality_label(quality.get('label') or quality.get('level')) or '—'}",
                    f"引用覆盖：{self._percentage_or_dash(quality.get('citation_coverage'))}",
                    f"来源多样性：{self._percentage_or_dash(quality.get('source_diversity'))}",
                ]
            )
            reasons = quality.get("reasons")
            if isinstance(reasons, list) and reasons:
                lines.append("\n判断依据：")
                lines.extend(f"• {reason}" for reason in reasons)
            elif quality.get("explanation"):
                lines.append(f"\n判断依据：{quality['explanation']}")
        lines.append(
            "\n说明：证据先按语义、全文命中和重排结果自动排序；这里展示的是可解释的证据质量，不是模型对答案正确率的主观百分比。"
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("为什么使用这些证据")
        dialog.resize(620, 460)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setPlainText("\n".join(lines))
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    @staticmethod
    def _percentage_or_dash(value: object) -> str:
        return f"{float(value):.0%}" if isinstance(value, (int, float)) else "—"

    def new_chat(self) -> None:
        if self._answer_busy:
            self.statusBar().showMessage("请先停止当前回答，再创建新对话。", 5000)
            return
        self.conversation_id = None
        self.history_list.clearSelection()
        self.last_answer = None
        self.last_question = ""
        self.last_answer_id = None
        self.last_answer_markdown = ""
        self.last_retrieval_info = {}
        self._chat_entries = []
        self._stream_text = ""
        self._render_chat()
        self.evidence_list.clear()
        self.evidence_by_id.clear()
        self.evidence_quality.setText("尚未生成回答")
        self.preview.clear()
        self.answer_actions.hide()
        self.helpful_button.setEnabled(True)
        self.unhelpful_button.setEnabled(True)
        self.answer_status.setText("❄ 就绪")
        self.prompt.clear()
        self._clear_pending_images(delete_files=True)

    def _selected_source_context(self) -> tuple[str | None, str | None, float | None]:
        document_id = None
        chunk_id = None
        timestamp_start: float | None = None
        evidence_item = self.evidence_list.currentItem()
        if evidence_item:
            value = self.evidence_by_id.get(str(evidence_item.data(Qt.UserRole)))
            if isinstance(value, dict):
                source = value.get("source") if isinstance(value.get("source"), dict) else {}
                document_id = value.get("document_id") or source.get("document_id")
                chunk_id = value.get("chunk_id") or source.get("chunk_id")
                raw_timestamp = value.get("timestamp_start")
                if raw_timestamp is None:
                    raw_timestamp = source.get("timestamp_start")
            else:
                document_id = getattr(value, "document_id", None)
                chunk_id = getattr(value, "chunk_id", None)
                source = getattr(value, "source", None)
                document_id = document_id or getattr(source, "document_id", None)
                chunk_id = chunk_id or getattr(source, "chunk_id", None)
                raw_timestamp = getattr(value, "timestamp_start", None)
                if raw_timestamp is None:
                    raw_timestamp = getattr(source, "timestamp_start", None)
            try:
                timestamp_start = float(raw_timestamp) if raw_timestamp is not None else None
            except (TypeError, ValueError):
                timestamp_start = None
        if not document_id and self.document_list.currentItem():
            current = self.document_list.currentItem().data(Qt.UserRole)
            if isinstance(current, dict):
                document_id = current.get("id")
        return (
            str(document_id) if document_id else None,
            str(chunk_id) if chunk_id else None,
            timestamp_start,
        )

    def _document_record(self, document_id: str | None) -> dict[str, object] | None:
        if not document_id:
            return None
        return next(
            (value for value in self.controller.documents(limit=2000) if value["id"] == document_id),
            None,
        )

    def _release_media_player(self, dialog: MediaPlayerDialog) -> None:
        if dialog in self._media_players:
            self._media_players.remove(dialog)

    def _open_media_player(
        self,
        document: dict[str, object],
        *,
        start_ms: int = 0,
    ) -> bool:
        latest = self.controller.latest_transcript(str(document["id"]))
        transcript = latest.get("transcript") if isinstance(latest, dict) else None
        transcript = transcript if isinstance(transcript, dict) else {}
        speaker_values = transcript.get("speakers")
        speaker_names = {
            str(item.get("id")): str(item.get("display_name") or item.get("id"))
            for item in speaker_values
            if isinstance(item, dict) and item.get("id")
        } if isinstance(speaker_values, list) else {}
        raw_segments = transcript.get("segments")
        segments: list[dict[str, object]] = []
        if isinstance(raw_segments, list):
            for value in raw_segments:
                if not isinstance(value, dict):
                    continue
                enriched = dict(value)
                speaker_id = str(enriched.get("speaker_id") or "")
                if speaker_id:
                    enriched["speaker_display_name"] = speaker_names.get(speaker_id, speaker_id)
                segments.append(enriched)
        target = (
            latest.get("media_path") if isinstance(latest, dict) else None
        ) or document.get("local_path") or document.get("original_uri")
        if not target:
            self.statusBar().showMessage("这份音视频没有可播放的本地原始资料", 6000)
            return False
        try:
            player = MediaPlayerDialog(
                str(target),
                title=f"原始证据 · {document['title']}",
                segments=segments,
                start_ms=max(0, int(start_ms)),
                autoplay=False,
                parent=self,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            self._operation_error("无法打开音视频播放器", str(exc))
            return False
        player.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._media_players.append(player)
        player.destroyed.connect(
            lambda _object=None, value=player: self._release_media_player(value)
        )
        player.show()
        player.raise_()
        player.activateWindow()
        return True

    def open_source_reader(self) -> None:
        document_id, chunk_id, timestamp_start = self._selected_source_context()
        document = self._document_record(document_id)
        if not document:
            self.statusBar().showMessage("请先选择一份资料或一条证据", 5000)
            return
        if str(document.get("media_type") or "").casefold() in {"audio", "video"}:
            self._open_media_player(
                document,
                start_ms=int(max(0.0, timestamp_start or 0.0) * 1000),
            )
            return
        SourceReaderDialog(
            self.controller, document, chunk_id=str(chunk_id) if chunk_id else None, parent=self
        ).exec()

    def open_transcript_editor(self) -> None:
        document_id, _chunk_id, timestamp_start = self._selected_source_context()
        document = self._document_record(document_id)
        if not document:
            self.statusBar().showMessage("请先选择一份音频或视频资料", 5000)
            return
        if str(document.get("media_type") or "").casefold() not in {"audio", "video"}:
            self.statusBar().showMessage("当前资料不是音频或视频，没有可校订的转写", 6000)
            return
        latest = self.controller.latest_transcript(str(document["id"]))
        if not isinstance(latest, dict):
            self.statusBar().showMessage("这份资料还没有 Transcript V2 转写，请重新解析", 7000)
            return
        run = latest.get("run") if isinstance(latest.get("run"), dict) else {}
        run_id = str(run.get("id") or "")
        if not run_id:
            self.statusBar().showMessage("转写任务记录不完整，请重新解析", 7000)
            return

        edited_segment_ids: set[str] = set()
        state = {"speaker_changed": False, "approved": False}
        with KnowledgeDatabase(self.controller.paths.database) as database:
            repository = TranscriptRepository(database)
            try:
                dialog = TranscriptEditorDialog(
                    repository,
                    run_id,
                    media_path=latest.get("media_path"),
                    play_callback=lambda _path, start, _end: self._open_media_player(
                        document, start_ms=start
                    ),
                    parent=self,
                )
            except (RuntimeError, KeyError, ValueError) as exc:
                self._operation_error("无法打开转写校订", str(exc))
                return
            dialog.transcriptSaved.connect(
                lambda _run, values: edited_segment_ids.update(
                    str(value) for value in values if str(value)
                )
            )
            dialog.speakerChanged.connect(
                lambda _run, _values: state.__setitem__("speaker_changed", True)
            )
            dialog.reviewApproved.connect(
                lambda _run: state.__setitem__("approved", True)
            )
            if timestamp_start is not None:
                closest = min(
                    range(len(dialog.transcript.segments)),
                    key=lambda index: abs(
                        dialog.transcript.segments[index].start_ms - timestamp_start * 1000
                    ),
                    default=None,
                )
                if closest is not None:
                    dialog.segment_list.setCurrentRow(closest)
            dialog.exec()

        needs_refresh = bool(edited_segment_ids or state["speaker_changed"])
        if not needs_refresh and not state["approved"]:
            return

        def persist_review() -> dict[str, object]:
            report: dict[str, object] = {"run_id": run_id}
            if needs_refresh:
                report["index"] = self.controller.refresh_transcript_index(
                    run_id,
                    affected_segment_ids=edited_segment_ids,
                )
            if state["approved"]:
                report["approval"] = self.controller.approve_transcript_for_retrieval(run_id)
            return report

        self._run_simple_background(
            "正在更新校订后的说话人、全文与语义索引…",
            persist_review,
            lambda report: self._sync_complete({"转写校订": report}),
        )

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
        operation_token = object()
        if not self._begin_db_operation(
            "监听文件夹同步",
            operation_token,
            requested="扫描监听文件夹",
        ):
            return
        self._watch_scan_running = True
        self._watch_operation_token = operation_token
        self.statusBar().showMessage("正在增量扫描监听文件夹…")
        worker = Worker(lambda _signals: self.controller.scan_watched_folders())
        worker.signals.result.connect(self._sync_complete)
        worker.signals.error.connect(lambda message: self._operation_error("自动同步失败", message))
        worker.signals.finished.connect(
            lambda operation_token=operation_token: self._finish_watch_scan(operation_token)
        )
        self.thread_pool.start(worker)

    def _finish_watch_scan(self, operation_token: object) -> None:
        if self._watch_operation_token is not operation_token:
            return
        self._watch_scan_running = False
        self._watch_operation_token = None
        self._finish_db_operation(operation_token)

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
            self.refresh_knowledge()
            dialog.exec()

        self._run_simple_background(
            f"正在生成{labels[artifact_type]}…",
            lambda: self.controller.create_artifact(
                artifact_type, title, document_ids=ids, model_id=model_id
            ), complete,
        )

    def run_privacy_scan(self) -> None:
        choice = QMessageBox.question(
            self,
            "本地隐私扫描",
            "是否同时使用本地 OCR 检查图片中的敏感文字？\n\n"
            "选择“否”会更快，但图片、PDF 和音视频中的隐藏内容会作为扫描局限明确列出。",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No,
        )
        if choice == QMessageBox.Cancel:
            return
        enable_ocr = choice == QMessageBox.Yes
        self._run_simple_background(
            "正在本地扫描隐私风险…",
            lambda: self.controller.privacy_scan(enable_image_ocr=enable_ocr),
            lambda report: self._show_privacy_report(report, "隐私扫描结果"),
        )

    def _show_privacy_report(self, report: dict[str, object], title: str) -> None:
        status_labels = {"clean": "未发现明显风险", "review": "需要人工复核", "blocked": "发现阻断风险"}
        findings = report.get("findings") or []
        lines = [
            f"# {status_labels.get(str(report.get('status')), str(report.get('status') or '扫描完成'))}",
            "",
            f"- 已扫描文件：{int(report.get('scanned_files') or 0)}",
            f"- 文本文件：{int(report.get('text_files_scanned') or 0)}",
            f"- 检查图片：{int(report.get('image_files_checked') or 0)}",
            f"- OCR 图片：{int(report.get('ocr_images_scanned') or 0)}",
            f"- 跳过文件：{int(report.get('skipped_files') or 0)}",
        ]
        if findings:
            lines.extend(["", "## 风险摘要", ""])
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                severity = {"block": "阻断", "blocked": "阻断", "review": "复核", "warning": "提醒"}.get(
                    str(finding.get("severity")), str(finding.get("severity") or "提醒")
                )
                category = str(finding.get("category") or "")
                category_label = PRIVACY_CATEGORY_LABELS.get(category)
                if category_label is None:
                    for prefix in ("image_ocr_", "image_metadata_"):
                        if category.startswith(prefix):
                            base = PRIVACY_CATEGORY_LABELS.get(category.removeprefix(prefix), "敏感信息")
                            category_label = f"图片{'OCR 文字' if prefix == 'image_ocr_' else '元数据'}中的{base}"
                            break
                lines.append(
                    f"- **{severity} · {category_label or '隐私风险'}**："
                    f"`{finding.get('redacted_path') or '已脱敏路径'}` · "
                    f"{finding.get('summary') or '发现需要复核的内容'}"
                )
        limitations = report.get("limitations") or []
        if limitations:
            lines.extend(["", "## 扫描局限", ""])
            lines.extend(f"- {value}" for value in limitations)
        lines.extend(
            [
                "",
                "---",
                "报告不会显示命中的密钥、Cookie、私人文本原文或完整绝对路径，避免扫描日志再次泄密。",
            ]
        )
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 590)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setAccessibleName("隐私扫描报告")
        browser.setMarkdown("\n".join(lines))
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def create_safe_share_copy(self) -> None:
        selected_documents = self._selected_documents()
        dialog = QDialog(self)
        dialog.setWindowTitle("生成安全分享副本")
        dialog.setMinimumWidth(650)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "系统钥匙串、模型配置、缓存、对话记录和数据库不会进入分享副本。"
            "检测到凭据、私人信息或无法检查的内容时，生成会停止。"
            "当前安全副本只发布通过全文件校验的 Markdown/纯文本；所有知识内容默认不勾选。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        include_notes = QCheckBox("包含正式知识与知识工坊 Markdown")
        include_notes.setAccessibleName("在分享副本中包含知识笔记")
        layout.addWidget(include_notes)
        include_sources = QCheckBox(
            f"包含当前选中的 {len(selected_documents)} 份原始资料（仅严格文本格式）"
        )
        include_sources.setEnabled(bool(selected_documents))
        include_sources.setAccessibleName("在分享副本中包含选中的原始资料")
        layout.addWidget(include_sources)
        scan_images = QCheckBox("使用本地 OCR 生成图片隐私风险报告（较慢）")
        layout.addWidget(scan_images)
        document_policy = QLabel(
            "PDF、Office/ODF 与图片仍会在本机深度检查并生成脱敏报告，但原始容器不会被原样复制。"
            "这是为了阻断不可达对象、压缩尾部和隐藏元数据；音视频和其他二进制同样保持阻断。"
        )
        document_policy.setWordWrap(True)
        document_policy.setObjectName("muted")
        layout.addWidget(document_policy)
        warning = QLabel("存在阻断级风险或无法确认安全时，系统会停止生成，而不是带风险继续导出。")
        warning.setText(
            "Source Notes 和已保存回答可能保留本机定位信息，默认不纳入分享。\n"
            "密钥、令牌、私钥、邮箱、手机号和敏感路径等阻断风险永远不允许绕过。"
        )
        warning.setWordWrap(True)
        warning.setObjectName("muted")
        layout.addWidget(warning)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("选择位置并生成")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        parent_directory = QFileDialog.getExistingDirectory(
            self, "选择分享副本保存位置", str(Path.home() / "Desktop")
        )
        if not parent_directory:
            return
        destination = Path(parent_directory) / (
            f"AI静静-安全分享-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        )
        document_ids = [str(value["id"]) for value in selected_documents] if include_sources.isChecked() else []

        def complete(report: dict[str, object]) -> None:
            QMessageBox.information(
                self,
                "安全分享副本已生成",
                f"保存位置：\n{report.get('destination') or destination}\n\n"
                f"文件数：{report.get('file_count', 0)}\n"
                f"清单 SHA-256：{report.get('manifest_sha256') or '已写入 manifest'}\n\n"
                "副本没有自动上传或发送。",
            )

        self._run_simple_background(
            "正在扫描并生成安全分享副本…",
            lambda: self.controller.create_safe_share_copy(
                destination,
                include_notes=include_notes.isChecked(),
                document_ids=document_ids,
                scan_images_with_ocr=scan_images.isChecked(),
                allow_review_findings=False,
            ),
            complete,
        )

    def create_backup(self) -> None:
        if self.import_token is not None or self._watch_scan_running:
            QMessageBox.information(self, "导入进行中", "请等待当前导入或同步结束后再创建完整备份。")
            return
        self._run_simple_background(
            "正在创建可恢复备份…", self.controller.create_backup,
            lambda path: QMessageBox.information(self, "备份完成", f"已保存到：\n{path}\n\nAPI 密钥不会写入备份。"),
        )

    def restore_backup(self) -> None:
        if self.import_token is not None or self._watch_scan_running or self._answer_busy:
            QMessageBox.information(self, "任务进行中", "请等待导入/同步结束并停止当前回答后再恢复备份。")
            return
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
                "正在验证并恢复备份…", lambda: self.controller.restore_backup(path), self._restore_complete
            )

    def _restore_complete(self, report: dict[str, object]) -> None:
        self.controller.reload()
        self.new_chat()
        self.refresh_library()
        self.refresh_knowledge()
        self.refresh_conversations()
        self.refresh_ingestion_jobs()
        self._load_models()
        self._reset_sync_timer()
        self._refresh_status()
        QMessageBox.information(
            self,
            "恢复完成",
            f"数据已通过完整性校验并恢复。\n\n恢复前安全备份：\n{report.get('safety_backup') or '已创建'}",
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
        ocr = report.get("ocr") if isinstance(report.get("ocr"), dict) else {}
        transcription = (
            report.get("transcription")
            if isinstance(report.get("transcription"), dict)
            else {}
        )
        route = (
            transcription.get("route")
            if isinstance(transcription.get("route"), dict)
            else {}
        )
        attempts = route.get("attempts") if isinstance(route.get("attempts"), list) else []
        attempt_labels = [
            f"{item.get('provider')} / {item.get('model')}"
            for item in attempts
            if isinstance(item, dict)
        ]
        route_label = (
            f"{route.get('profile') or '未指定方案'} · "
            + (" → ".join(attempt_labels) or "未形成可用路线")
            if route.get("available")
            else str(route.get("error") or "不可用")
        )
        machine = report.get("machine") if isinstance(report.get("machine"), dict) else {}
        offline = report.get("offline") if isinstance(report.get("offline"), dict) else {}
        embedding = report.get("embedding") if isinstance(report.get("embedding"), dict) else {}
        resource_scheduler = (
            report.get("resource_scheduler")
            if isinstance(report.get("resource_scheduler"), dict)
            else {}
        )
        diarization = (
            transcription.get("diarization")
            if isinstance(transcription.get("diarization"), dict)
            else {}
        )
        lines = [
            f"Python {report['python']}",
            f"设备：{machine.get('processor') or machine.get('machine') or '未知'} · "
            f"{machine.get('memory_gb') or '未知'} GB 统一/系统内存 · "
            f"Apple Silicon {'是' if machine.get('apple_silicon') else '否'}",
            f"SQLite FTS5：{'可用' if report['sqlite_fts5'] else '不可用'}",
            f"FFmpeg：{report['ffmpeg'] or '不可用'}",
            f"OCR：{ocr.get('requested_engine') or '自动'} · "
            f"RapidOCR {'可用' if ocr.get('rapidocr_available') else '不可用'} · "
            f"PaddleOCR {'可用' if ocr.get('paddleocr_available') else '可选未安装'}",
            f"转写路由：{route_label}",
            f"向量检索：{embedding.get('provider') or '未知'} · "
            f"{'本地就绪' if embedding.get('local_ready') else '本地模型缺失'}",
            f"说话人路线：{diarization.get('provider') or '未启用'} · "
            f"{diarization.get('reason') or '未检查'}",
            f"严格离线：{'通过' if offline.get('strict_ready') else '未通过'} · "
            f"隐藏模型下载 {'已阻止' if offline.get('hidden_model_downloads_blocked') else '需检查'}",
            f"本地高内存任务并发：{resource_scheduler.get('concurrency_limit') or 1} · "
            f"{resource_scheduler.get('release_policy') or '任务结束释放'}",
            "",
        ]
        risks = offline.get("risks") if isinstance(offline.get("risks"), list) else []
        if risks:
            lines.append("离线检查提示：")
            lines.extend(f"— {value}" for value in risks)
            lines.append("")
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
                    f"{report.get('notes') or ''}\n\n"
                    "是否由 AI静静下载更新包，并在打开前校验 SHA-256 完整性？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                )
                if choice == QMessageBox.Yes:
                    download_url = str(report.get("download_url") or "")
                    checksum = str(report.get("sha256") or "")

                    def downloaded(path: Path) -> None:
                        open_choice = QMessageBox.question(
                            self,
                            "更新包校验通过",
                            f"更新包已通过 SHA-256 校验：\n{path}\n\n现在打开安装包吗？",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if open_choice == QMessageBox.Yes:
                            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

                    self._run_simple_background(
                        "正在通过 HTTPS 下载并校验更新包…",
                        lambda: self.controller.download_update(download_url, checksum),
                        downloaded,
                    )
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

    def _background_conflict(self, requested: str) -> bool:
        """Report a conflict with the single database-wide operation guard."""

        if self._active_db_operation_token is None:
            return False
        active = self._active_db_operation_label.rstrip("…。 ") or "后台操作"
        message = f"“{active}”仍在进行，请完成后再{requested}。"
        self.statusBar().showMessage(message, 7000)
        QMessageBox.information(self, "后台操作进行中", message)
        return True

    def _begin_db_operation(
        self,
        label: str,
        token: object,
        *,
        requested: str,
    ) -> bool:
        """Acquire the app-wide operation slot with an identity-safe token."""

        if self._background_conflict(requested):
            return False
        self._active_db_operation_token = token
        self._active_db_operation_label = label
        return True

    def _finish_db_operation(self, token: object) -> None:
        """Release only the operation that acquired the current slot."""

        if self._active_db_operation_token is not token:
            return
        self._active_db_operation_token = None
        self._active_db_operation_label = ""

    def _run_simple_background(
        self,
        label: str,
        function: Callable[[], object],
        complete: Callable[[object], None],
    ) -> bool:
        worker = Worker(lambda _signals: function())
        if not self._begin_db_operation(
            label,
            worker,
            requested="启动另一项操作",
        ):
            return False

        self.statusBar().showMessage(label)
        self._background_worker = worker
        self._background_operation_label = label
        if self.centralWidget() is not None:
            self.centralWidget().setEnabled(False)
        self.menuBar().setEnabled(False)

        def release() -> None:
            # A result handler may immediately start a follow-up operation (for
            # example update check -> package download).  A late finished signal
            # from the previous worker must never unlock that newer operation.
            if self._background_worker is not worker:
                return
            self._background_worker = None
            self._background_operation_label = ""
            self._finish_db_operation(worker)
            if self.centralWidget() is not None:
                self.centralWidget().setEnabled(True)
            self.menuBar().setEnabled(True)

        def on_result(value: object) -> None:
            release()
            complete(value)

        def on_error(message: str) -> None:
            release()
            self._operation_error("操作失败", message)

        worker.signals.result.connect(on_result)
        worker.signals.error.connect(on_error)
        worker.signals.finished.connect(release)
        self.thread_pool.start(worker)
        return True

    def _sync_complete(self, report) -> None:
        self.refresh_library()
        self.refresh_knowledge()
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
        if self._background_worker is not None:
            active = self._background_operation_label.rstrip("…。 ") or "后台操作"
            QMessageBox.information(
                self,
                "后台操作仍在进行",
                f"“{active}”尚未完成。为避免数据库或备份损坏，请等待完成后再关闭应用。",
            )
            event.ignore()
            return
        importing = bool(self.import_token and not self.import_token.cancelled)
        answering = self._answer_busy
        if (
            self._active_db_operation_token is not None
            and not importing
            and not answering
        ):
            active = self._active_db_operation_label.rstrip("…。 ") or "后台操作"
            QMessageBox.information(
                self,
                "后台操作仍在进行",
                f"“{active}”尚未完成。为避免数据库损坏，请等待完成后再关闭应用。",
            )
            event.ignore()
            return
        if importing or answering:
            active = "、".join(
                value for value, enabled in (("资料导入", importing), ("回答生成", answering)) if enabled
            )
            choice = QMessageBox.question(
                self,
                "后台任务仍在进行",
                f"{active}尚未完成。关闭应用会安全停止任务；未完成导入可在下次启动时继续。确定关闭吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            if importing and self.import_token:
                self.import_token.cancel()
            if importing and self.import_job_id:
                try:
                    self.controller.cancel_ingestion_job(self.import_job_id)
                except (OSError, ValueError):
                    pass
            if answering:
                self._answer_cancelled.set()
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
