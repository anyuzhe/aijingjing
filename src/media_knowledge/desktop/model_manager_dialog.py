from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .model_manager import LocalModelManager, LocalModelStatus


def _human_size(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


class _ModelTask(QObject):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[..., LocalModelStatus]) -> None:
        super().__init__()
        self.operation = operation
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()

    def _check_cancelled(self) -> None:
        if self.cancelled.is_set():
            raise RuntimeError("模型安装已取消")

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation(
                progress=self.progress.emit,
                check_cancelled=self._check_cancelled,
            )
        except Exception as error:
            self.failed.emit(str(error) or type(error).__name__)
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


class ModelManagerDialog(QDialog):
    """Explicit model download/import manager; status refresh is local-only."""

    def __init__(self, manager: LocalModelManager, parent=None) -> None:
        super().__init__(parent)
        self.manager = manager
        self._thread: QThread | None = None
        self._task: _ModelTask | None = None
        self.setWindowTitle("本地模型管理 · AI静静")
        self.resize(1040, 610)
        self.setMinimumSize(860, 520)
        layout = QVBoxLayout(self)

        title = QLabel("转写与说话人模型")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        explanation = QLabel(
            "AI静静只使用已经安装在本机的权重。查看状态、导入资料和普通转写都不会自动联网；"
            "只有点击“下载安装”后才会访问 Hugging Face。模型权重不会被打进主程序。"
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("muted")
        layout.addWidget(explanation)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["用途", "模型", "引擎", "本地状态", "完整性", "大小", "许可证", "路径"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in range(7):
            header.setSectionResizeMode(column, header.ResizeToContents)
        header.setSectionResizeMode(1, header.Stretch)
        header.setSectionResizeMode(7, header.Stretch)
        self.table.itemSelectionChanged.connect(self._update_actions)
        layout.addWidget(self.table, 1)

        token_row = QHBoxLayout()
        token_label = QLabel("Hugging Face Token（仅受限模型下载时使用）")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("临时使用，不保存")
        self.token.setAccessibleName("Hugging Face 临时访问令牌")
        token_row.addWidget(token_label)
        token_row.addWidget(self.token, 1)
        layout.addLayout(token_row)

        action_row = QHBoxLayout()
        self.download_button = QPushButton("下载安装")
        self.download_button.clicked.connect(self._download)
        self.import_button = QPushButton("复制导入…")
        self.import_button.clicked.connect(self._import)
        self.register_button = QPushButton("使用已有目录…")
        self.register_button.clicked.connect(self._register)
        self.remove_button = QPushButton("移除托管模型")
        self.remove_button.setObjectName("danger")
        self.remove_button.clicked.connect(self._remove)
        self.verify_button = QPushButton("校验 SHA-256")
        self.verify_button.setAccessibleName("重新校验选中模型的全部文件")
        self.verify_button.clicked.connect(self._verify)
        self.refresh_button = QPushButton("刷新状态")
        self.refresh_button.clicked.connect(self.refresh)
        self.close_button = QPushButton("完成")
        self.close_button.clicked.connect(self.accept)
        for button in (
            self.download_button,
            self.import_button,
            self.register_button,
            self.remove_button,
            self.verify_button,
            self.refresh_button,
        ):
            action_row.addWidget(button)
        action_row.addStretch(1)
        action_row.addWidget(self.close_button)
        layout.addLayout(action_row)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.status_label = QLabel("请选择模型。")
        self.status_label.setObjectName("muted")
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.status_label, 2)
        progress_row.addWidget(self.cancel_button)
        layout.addLayout(progress_row)
        self.refresh()

    def selected_model_id(self) -> str | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        item = self.table.item(rows[0].row(), 0)
        return str(item.data(Qt.UserRole)) if item else None

    @Slot()
    def refresh(self) -> None:
        selected = self.selected_model_id()
        statuses = self.manager.statuses()
        self.table.setRowCount(len(statuses))
        for row, status in enumerate(statuses):
            spec = status.spec
            values = (
                "语音识别" if spec.kind == "asr" else "说话人识别",
                spec.label,
                spec.provider,
                (
                    "已安装并通过内容校验"
                    if status.content_verified
                    else ("已安装（待内容校验）" if status.verified else ("本地文件异常" if status.installed else "未安装"))
                ),
                (
                    f"SHA {status.content_sha256[:10]}…"
                    if status.content_verified and status.content_sha256
                    else ("结构通过" if status.verified else "未通过")
                ),
                _human_size(status.size_bytes) if status.installed else f"约 {spec.approximate_size_gb:g} GB",
                spec.license_name,
                status.path or "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(
                    (status.content_sha256 or spec.description)
                    if column == 4 else (value if column == 7 else spec.description)
                )
                if column == 0:
                    item.setData(Qt.UserRole, spec.model_id)
                self.table.setItem(row, column, item)
            if spec.model_id == selected:
                self.table.selectRow(row)
        if self.table.rowCount() and not self.table.selectionModel().hasSelection():
            self.table.selectRow(0)
        self._update_actions()

    @Slot()
    def _update_actions(self) -> None:
        model_id = self.selected_model_id()
        busy = self._task is not None
        status = self.manager.status(model_id) if model_id else None
        self.download_button.setEnabled(bool(model_id and not busy and status and not status.installed))
        self.import_button.setEnabled(bool(model_id and not busy and status and not status.installed))
        self.register_button.setEnabled(bool(model_id and not busy))
        self.remove_button.setEnabled(bool(model_id and not busy and status and status.installed))
        self.verify_button.setEnabled(bool(model_id and not busy and status and status.installed))
        self.refresh_button.setEnabled(not busy)
        self.close_button.setEnabled(not busy)
        if status and not busy:
            self.status_label.setText(
                f"{status.spec.label}："
                + (
                    f"已就绪且内容校验通过（{status.source}）"
                    if status.content_verified
                    else (f"本地结构可用，建议执行 SHA-256 校验（{status.source}）" if status.verified else "尚未安装")
                )
            )

    def _start(self, operation: Callable[..., LocalModelStatus], initial: str) -> None:
        if self._task is not None:
            return
        self.status_label.setText(initial)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.cancel_button.show()
        thread = QThread(self)
        task = _ModelTask(operation)
        task.moveToThread(thread)
        task.progress.connect(self.status_label.setText)
        task.succeeded.connect(self._succeeded)
        task.failed.connect(self._failed)
        task.finished.connect(thread.quit)
        task.finished.connect(task.deleteLater)
        thread.finished.connect(self._finished)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(task.run)
        self._thread = thread
        self._task = task
        self._update_actions()
        thread.start()

    @Slot()
    def _download(self) -> None:
        model_id = self.selected_model_id()
        if not model_id:
            return
        spec = self.manager.spec(model_id)
        if spec.gated and not self.token.text().strip():
            QMessageBox.warning(
                self,
                "需要访问令牌",
                "该模型需先在 Hugging Face 接受使用条件，再临时输入访问令牌。AI静静不会保存令牌。",
            )
            return
        answer = QMessageBox.question(
            self,
            "确认下载本地模型",
            f"将下载“{spec.label}”（约 {spec.approximate_size_gb:g} GB）到：\n"
            f"{self.manager.root}\n\n只有这次明确操作会联网。是否继续？",
        )
        if answer != QMessageBox.Yes:
            return
        token = self.token.text().strip() or None
        self.token.clear()
        self._start(
            lambda **kwargs: self.manager.download(model_id, token=token, **kwargs),
            f"准备下载 {spec.label}…",
        )

    @Slot()
    def _import(self) -> None:
        model_id = self.selected_model_id()
        if not model_id:
            return
        source = QFileDialog.getExistingDirectory(self, "选择已经下载的模型目录")
        if not source:
            return
        self._start(
            lambda **kwargs: self.manager.import_model(model_id, Path(source), **kwargs),
            "准备复制本地模型…",
        )

    @Slot()
    def _register(self) -> None:
        model_id = self.selected_model_id()
        if not model_id:
            return
        source = QFileDialog.getExistingDirectory(self, "选择已经下载的模型目录")
        if not source:
            return
        self._start(
            lambda **kwargs: self.manager.register_path(
                model_id, Path(source), **kwargs
            ),
            "正在登记并校验已有模型目录；模型文件不会被复制…",
        )

    @Slot()
    def _remove(self) -> None:
        model_id = self.selected_model_id()
        if not model_id:
            return
        status = self.manager.status(model_id)
        detail = (
            "将删除 AI静静模型目录中的托管副本。"
            if status.source in {"managed", "downloaded", "imported"}
            else "该模型来自外部目录或公共缓存，只会取消登记，不会删除原文件。"
        )
        if QMessageBox.warning(
            self,
            "确认移除模型",
            f"{status.spec.label}\n{detail}\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        try:
            removed = self.manager.remove(model_id)
        except OSError as error:
            QMessageBox.critical(self, "移除失败", str(error))
            return
        self.status_label.setText("托管模型已删除。" if removed else "已取消外部模型登记，原文件保持不变。")
        self.refresh()

    @Slot()
    def _verify(self) -> None:
        model_id = self.selected_model_id()
        if not model_id:
            return
        self._start(
            lambda **kwargs: self.manager.verify(model_id, **kwargs),
            "准备校验模型内容…",
        )

    @Slot(object)
    def _succeeded(self, status: LocalModelStatus) -> None:
        self.status_label.setText(f"{status.spec.label} 已安装并通过本地校验。")

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.status_label.setText("模型任务失败。")
        QMessageBox.critical(self, "模型任务失败", message)

    @Slot()
    def _finished(self) -> None:
        self._task = None
        self._thread = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.hide()
        self.cancel_button.hide()
        self.refresh()

    @Slot()
    def _cancel(self) -> None:
        if self._task:
            self._task.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在安全取消；已完成的网络请求结束后会清理临时文件…")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._task is not None:
            QMessageBox.information(self, "模型任务仍在运行", "请先取消任务并等待临时文件清理完成。")
            event.ignore()
            return
        super().closeEvent(event)


__all__ = ["ModelManagerDialog"]
