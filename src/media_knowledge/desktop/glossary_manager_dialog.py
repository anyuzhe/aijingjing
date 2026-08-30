from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..storage import KnowledgeDatabase
from ..transcripts import TranscriptRepository
from .controller import DesktopController


_SCOPE_LABELS = {
    "global": "全局",
    "knowledge_space": "知识空间",
    "source": "单一来源",
}


class _NewGlossaryDialog(QDialog):
    def __init__(
        self,
        knowledge_space_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建专业词库")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("例如：岩体力学")
        self.name.setAccessibleName("术语库名称")
        form.addRow("词库名称", self.name)

        self.scope = QComboBox()
        self.scope.addItem("全局（所有音视频）", "global")
        self.scope.addItem("知识空间", "knowledge_space")
        self.scope.addItem("单一来源", "source")
        self.scope.setAccessibleName("术语库生效范围")
        form.addRow("生效范围", self.scope)

        self.scope_id = QLineEdit()
        self.scope_id.setText(knowledge_space_id)
        self.scope_id.setAccessibleName("术语库范围标识")
        form.addRow("范围 ID", self.scope_id)
        layout.addLayout(form)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setObjectName("muted")
        layout.addWidget(self.hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("创建")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.scope.currentIndexChanged.connect(self._update_scope)
        self._update_scope()

    @Slot()
    def _update_scope(self) -> None:
        scope = str(self.scope.currentData() or "global")
        self.scope_id.setEnabled(scope != "global")
        if scope == "global":
            self.hint.setText("全局词库对所有音视频转写生效，不需要范围 ID。")
        elif scope == "knowledge_space":
            self.hint.setText("仅在知识空间 ID 完全匹配时生效。")
        else:
            self.hint.setText("仅对来源 ID 完全匹配的文件或网页生效。")

    @Slot()
    def _validate_and_accept(self) -> None:
        if not self.name.text().strip():
            QMessageBox.warning(self, "无法创建", "请输入词库名称。")
            self.name.setFocus()
            return
        if self.scope.currentData() != "global" and not self.scope_id.text().strip():
            QMessageBox.warning(self, "无法创建", "请输入范围 ID。")
            self.scope_id.setFocus()
            return
        self.accept()

    def values(self) -> tuple[str, str, str | None]:
        scope = str(self.scope.currentData() or "global")
        return (
            self.name.text().strip(),
            scope,
            self.scope_id.text().strip() if scope != "global" else None,
        )


class GlossaryManagerDialog(QDialog):
    """Local manager for global, knowledge-space, and source ASR glossaries."""

    def __init__(
        self,
        controller: DesktopController,
        parent: QWidget | None = None,
        *,
        knowledge_space_id: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.knowledge_space_id = (
            str(knowledge_space_id or controller.settings.asr_knowledge_space_id).strip()
            or "本地知识库"
        )
        self.setWindowTitle("专业词库管理")
        self.setMinimumSize(760, 500)
        self.resize(900, 600)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "专业词库会在音视频转写前合并为上下文术语。"
            "可按全局、知识空间或单一来源控制生效范围。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        glossary_panel = QWidget()
        glossary_layout = QVBoxLayout(glossary_panel)
        glossary_layout.setContentsMargins(0, 0, 6, 0)
        glossary_layout.addWidget(QLabel("术语库"))
        self.glossary_list = QListWidget()
        self.glossary_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.glossary_list.setAccessibleName("专业词库列表")
        self.glossary_list.currentItemChanged.connect(self._refresh_terms)
        glossary_layout.addWidget(self.glossary_list, 1)
        glossary_buttons = QHBoxLayout()
        add_glossary = QPushButton("新建词库…")
        add_glossary.clicked.connect(self._add_glossary)
        self.toggle_glossary_button = QPushButton("停用")
        self.toggle_glossary_button.clicked.connect(self._toggle_glossary)
        self.delete_glossary_button = QPushButton("删除")
        self.delete_glossary_button.setObjectName("danger")
        self.delete_glossary_button.clicked.connect(self._delete_glossary)
        glossary_buttons.addWidget(add_glossary)
        glossary_buttons.addWidget(self.toggle_glossary_button)
        glossary_buttons.addWidget(self.delete_glossary_button)
        glossary_layout.addLayout(glossary_buttons)
        splitter.addWidget(glossary_panel)

        term_panel = QWidget()
        term_layout = QVBoxLayout(term_panel)
        term_layout.setContentsMargins(6, 0, 0, 0)
        self.term_heading = QLabel("术语")
        term_layout.addWidget(self.term_heading)
        self.term_list = QListWidget()
        self.term_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.term_list.setAccessibleName("当前词库术语列表")
        self.term_list.currentItemChanged.connect(self._update_actions)
        term_layout.addWidget(self.term_list, 1)
        term_buttons = QHBoxLayout()
        self.add_term_button = QPushButton("添加术语…")
        self.add_term_button.clicked.connect(self._add_term)
        self.delete_term_button = QPushButton("删除术语")
        self.delete_term_button.setObjectName("danger")
        self.delete_term_button.clicked.connect(self._delete_term)
        term_buttons.addWidget(self.add_term_button)
        term_buttons.addWidget(self.delete_term_button)
        term_buttons.addStretch(1)
        term_layout.addLayout(term_buttons)
        splitter.addWidget(term_panel)
        splitter.setSizes([360, 520])

        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.button(QDialogButtonBox.Close).setText("关闭")
        close_buttons.rejected.connect(self.reject)
        layout.addWidget(close_buttons)
        self.refresh()

    def _repository(self, database: KnowledgeDatabase) -> TranscriptRepository:
        return TranscriptRepository(database)

    def refresh(self, *, select_glossary_id: str | None = None) -> None:
        previous = select_glossary_id or self.selected_glossary_id()
        self.glossary_list.blockSignals(True)
        self.glossary_list.clear()
        with KnowledgeDatabase(self.controller.paths.database) as database:
            glossaries = self._repository(database).list_glossaries()
        selected_row = -1
        for row, glossary in enumerate(glossaries):
            scope = _SCOPE_LABELS.get(glossary.scope, glossary.scope)
            range_text = f" · {glossary.scope_id}" if glossary.scope_id else ""
            state = "已启用" if glossary.enabled else "已停用"
            item = QListWidgetItem(f"{glossary.name}\n{scope}{range_text} · {state}")
            item.setData(Qt.UserRole, glossary.id)
            item.setData(Qt.UserRole + 1, glossary.enabled)
            item.setToolTip(f"{scope}{range_text}，{state}")
            self.glossary_list.addItem(item)
            if glossary.id == previous:
                selected_row = row
        self.glossary_list.blockSignals(False)
        if self.glossary_list.count():
            self.glossary_list.setCurrentRow(selected_row if selected_row >= 0 else 0)
        else:
            self._refresh_terms()
        self._update_actions()

    def selected_glossary_id(self) -> str | None:
        item = self.glossary_list.currentItem()
        return str(item.data(Qt.UserRole)) if item is not None else None

    def selected_term_id(self) -> str | None:
        item = self.term_list.currentItem()
        return str(item.data(Qt.UserRole)) if item is not None else None

    @Slot()
    def _refresh_terms(self, *_: object) -> None:
        glossary_id = self.selected_glossary_id()
        self.term_list.clear()
        if glossary_id is None:
            self.term_heading.setText("术语")
            self._update_actions()
            return
        current = self.glossary_list.currentItem()
        title = current.text().splitlines()[0] if current is not None else "当前词库"
        self.term_heading.setText(f"术语 · {title}")
        with KnowledgeDatabase(self.controller.paths.database) as database:
            terms = self._repository(database).list_glossary_terms(glossary_id)
        for term in terms:
            variants = f"\n别名：{'、'.join(term.variants)}" if term.variants else ""
            item = QListWidgetItem(f"{term.canonical_term}{variants}")
            item.setData(Qt.UserRole, term.id)
            item.setToolTip(term.notes or term.canonical_term)
            self.term_list.addItem(item)
        if self.term_list.count():
            self.term_list.setCurrentRow(0)
        self._update_actions()

    @Slot()
    def _update_actions(self, *_: object) -> None:
        glossary_item = self.glossary_list.currentItem()
        has_glossary = glossary_item is not None
        enabled = bool(glossary_item.data(Qt.UserRole + 1)) if has_glossary else False
        self.toggle_glossary_button.setEnabled(has_glossary)
        self.toggle_glossary_button.setText("停用" if enabled else "启用")
        self.delete_glossary_button.setEnabled(has_glossary)
        self.add_term_button.setEnabled(has_glossary)
        self.delete_term_button.setEnabled(self.term_list.currentItem() is not None)

    def create_glossary(self, name: str, scope: str, scope_id: str | None = None) -> str:
        with KnowledgeDatabase(self.controller.paths.database) as database:
            glossary = self._repository(database).create_glossary(
                name,
                scope=scope,
                scope_id=scope_id,
            )
        self.refresh(select_glossary_id=glossary.id)
        return glossary.id

    def add_term(self, glossary_id: str, term: str) -> str:
        with KnowledgeDatabase(self.controller.paths.database) as database:
            created = self._repository(database).add_glossary_term(glossary_id, term)
        self.refresh(select_glossary_id=glossary_id)
        return created.id

    @Slot()
    def _add_glossary(self) -> None:
        dialog = _NewGlossaryDialog(self.knowledge_space_id, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            self.create_glossary(*dialog.values())
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "无法创建词库", str(error))

    @Slot()
    def _toggle_glossary(self) -> None:
        glossary_id = self.selected_glossary_id()
        item = self.glossary_list.currentItem()
        if glossary_id is None or item is None:
            return
        try:
            with KnowledgeDatabase(self.controller.paths.database) as database:
                self._repository(database).update_glossary(
                    glossary_id,
                    enabled=not bool(item.data(Qt.UserRole + 1)),
                )
            self.refresh(select_glossary_id=glossary_id)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "无法更新词库", str(error))

    @Slot()
    def _delete_glossary(self) -> None:
        glossary_id = self.selected_glossary_id()
        item = self.glossary_list.currentItem()
        if glossary_id is None or item is None:
            return
        name = item.text().splitlines()[0]
        choice = QMessageBox.question(
            self,
            "删除专业词库",
            f"确定删除“{name}”及其中全部术语吗？此操作无法撤销。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        try:
            with KnowledgeDatabase(self.controller.paths.database) as database:
                self._repository(database).delete_glossary(glossary_id)
            self.refresh()
        except OSError as error:
            QMessageBox.warning(self, "无法删除词库", str(error))

    @Slot()
    def _add_term(self) -> None:
        glossary_id = self.selected_glossary_id()
        if glossary_id is None:
            return
        value, accepted = QInputDialog.getText(
            self,
            "添加专业术语",
            "标准术语",
            text="",
        )
        if not accepted or not value.strip():
            return
        try:
            self.add_term(glossary_id, value.strip())
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "无法添加术语", str(error))

    @Slot()
    def _delete_term(self) -> None:
        glossary_id = self.selected_glossary_id()
        term_id = self.selected_term_id()
        item = self.term_list.currentItem()
        if glossary_id is None or term_id is None or item is None:
            return
        term = item.text().splitlines()[0]
        choice = QMessageBox.question(
            self,
            "删除专业术语",
            f"确定从当前词库删除“{term}”吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        try:
            with KnowledgeDatabase(self.controller.paths.database) as database:
                self._repository(database).delete_glossary_term(term_id)
            self.refresh(select_glossary_id=glossary_id)
        except OSError as error:
            QMessageBox.warning(self, "无法删除术语", str(error))
