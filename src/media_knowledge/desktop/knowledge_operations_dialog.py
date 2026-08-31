from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from .controller import DesktopController


class _Signals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class _Task(QRunnable):
    def __init__(self, function: Callable[[], object]):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    def run(self) -> None:
        try:
            self.signals.result.emit(self.function())
        except Exception as exc:  # pragma: no cover - Qt thread delivery
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class KnowledgeOperationsDialog(QDialog):
    """One discoverable desktop surface for review, trust, eval, SOP and export."""

    def __init__(self, controller: DesktopController, parent: QWidget | None = None):
        super().__init__(parent)
        self.controller = controller
        self.thread_pool = QThreadPool.globalInstance()
        self.setWindowTitle("知识运营中心 · AI静静")
        self.setMinimumSize(980, 720)
        self.resize(1120, 800)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "在这里审核 AI 提炼的候选知识、评估来源可信度、检查冲突与新鲜度、"
            "运行黄金问题集、维护 SOP，并生成可脱离数据库阅读的 LLM Wiki。"
        )
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("知识运营功能标签")
        layout.addWidget(self.tabs, 1)
        self._build_proposals_tab()
        self._build_quality_tab()
        self._build_evaluation_tab()
        self._build_workflows_tab()
        self._build_wiki_tab()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh_all()

    def _build_proposals_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.proposal_summary = QLabel("正在读取候选知识…")
        self.proposal_summary.setObjectName("muted")
        row.addWidget(self.proposal_summary)
        row.addStretch(1)
        refresh = QPushButton("刷新候选")
        refresh.clicked.connect(self.refresh_proposals)
        row.addWidget(refresh)
        layout.addLayout(row)
        splitter = QSplitter(Qt.Horizontal)
        self.proposal_list = QListWidget()
        self.proposal_list.setAccessibleName("待审核知识候选列表")
        self.proposal_list.currentItemChanged.connect(self._show_proposal)
        splitter.addWidget(self.proposal_list)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        self.proposal_detail = QPlainTextEdit()
        self.proposal_detail.setReadOnly(True)
        self.proposal_detail.setAccessibleName("候选知识内容和来源证据")
        detail_layout.addWidget(self.proposal_detail, 1)
        actions = QHBoxLayout()
        self.accept_proposal = QPushButton("接受为待复核知识")
        self.accept_proposal.setObjectName("primary")
        self.accept_proposal.clicked.connect(lambda: self._review_proposal("accept", False))
        actions.addWidget(self.accept_proposal)
        self.merge_proposal = QPushButton("合并到重复知识")
        self.merge_proposal.clicked.connect(lambda: self._review_proposal("accept", True))
        actions.addWidget(self.merge_proposal)
        self.reject_proposal = QPushButton("拒绝候选")
        self.reject_proposal.setObjectName("danger")
        self.reject_proposal.clicked.connect(lambda: self._review_proposal("reject", False))
        actions.addWidget(self.reject_proposal)
        detail_layout.addLayout(actions)
        splitter.addWidget(detail)
        splitter.setSizes([360, 650])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(tab, "候选审核")

    def _build_quality_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.quality_summary = QLabel("正在检查来源、冲突和重复资料…")
        self.quality_summary.setObjectName("muted")
        self.quality_summary.setWordWrap(True)
        layout.addWidget(self.quality_summary)
        self.quality_table = QTableWidget(0, 4)
        self.quality_table.setAccessibleName("来源质量和冲突问题列表")
        self.quality_table.setHorizontalHeaderLabels(["级别", "资料", "问题", "资料 ID"])
        self.quality_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.quality_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.quality_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.quality_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.quality_table.itemSelectionChanged.connect(self._quality_selection_changed)
        layout.addWidget(self.quality_table, 1)
        form = QFormLayout()
        self.source_class = QComboBox()
        for label, value in (
            ("未评估", "unassessed"), ("官方资料", "official"),
            ("一手资料", "primary"), ("研究论文", "research"),
            ("行业资料", "industry"), ("媒体报道", "media"),
            ("社区内容", "community"), ("个人资料", "personal"),
        ):
            self.source_class.addItem(label, value)
        form.addRow("来源类型", self.source_class)
        self.source_reliability = QComboBox()
        for label, value in (("未评估", "unassessed"), ("高", "high"), ("中", "medium"), ("低", "low")):
            self.source_reliability.addItem(label, value)
        form.addRow("可靠性", self.source_reliability)
        self.source_valid_until = QLineEdit()
        self.source_valid_until.setPlaceholderText("YYYY-MM-DD，可留空")
        form.addRow("有效期至", self.source_valid_until)
        self.source_notes = QLineEdit()
        self.source_notes.setPlaceholderText("说明判断依据、适用范围或需要交叉核验的内容")
        form.addRow("评估说明", self.source_notes)
        layout.addLayout(form)
        quality_actions = QHBoxLayout()
        quality_actions.addStretch(1)
        self.save_assessment = QPushButton("保存来源评估")
        self.save_assessment.setObjectName("primary")
        self.save_assessment.clicked.connect(self._save_source_assessment)
        quality_actions.addWidget(self.save_assessment)
        layout.addLayout(quality_actions)
        self.tabs.addTab(tab, "来源与冲突")

    def _build_evaluation_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        hint = QLabel(
            "黄金问题集用于持续测量检索命中率与引用质量。默认只做本地检索评测；"
            "勾选引用评测后才会调用当前云模型。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        path_row = QHBoxLayout()
        self.eval_path = QLineEdit()
        self.eval_path.setPlaceholderText("选择符合 docs/golden-evaluation.example.json 格式的数据集")
        self.eval_path.setAccessibleName("黄金问题集 JSON 路径")
        path_row.addWidget(self.eval_path, 1)
        choose = QPushButton("选择 JSON…")
        choose.clicked.connect(self._choose_evaluation_file)
        path_row.addWidget(choose)
        layout.addLayout(path_row)
        options = QHBoxLayout()
        options.addWidget(QLabel("Top K"))
        self.eval_top_k = QSpinBox()
        self.eval_top_k.setRange(1, 12)
        self.eval_top_k.setValue(10)
        options.addWidget(self.eval_top_k)
        self.eval_citations = QCheckBox("同时评测回答引用（会调用当前模型）")
        options.addWidget(self.eval_citations)
        options.addStretch(1)
        self.run_eval = QPushButton("开始评测")
        self.run_eval.setObjectName("primary")
        self.run_eval.clicked.connect(self._run_evaluation)
        options.addWidget(self.run_eval)
        layout.addLayout(options)
        self.eval_status = QLabel("尚未运行评测")
        self.eval_status.setObjectName("muted")
        self.eval_status.setWordWrap(True)
        layout.addWidget(self.eval_status)
        self.eval_table = QTableWidget(0, 5)
        self.eval_table.setAccessibleName("黄金问题集评测结果")
        self.eval_table.setHorizontalHeaderLabels(["用例", "问题", "命中", "首个相关排名", "倒数排名"])
        self.eval_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.eval_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.eval_table, 1)
        self.tabs.addTab(tab, "黄金评测")

    def _build_workflows_tab(self) -> None:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.workflow_list = QListWidget()
        self.workflow_list.setAccessibleName("SOP 流程列表")
        self.workflow_list.currentItemChanged.connect(self._show_workflow)
        layout.addWidget(self.workflow_list, 1)
        editor = QWidget()
        form = QFormLayout(editor)
        self.workflow_name = QLineEdit()
        self.workflow_name.setAccessibleName("SOP 名称")
        form.addRow("名称", self.workflow_name)
        self.workflow_description = QLineEdit()
        form.addRow("用途", self.workflow_description)
        self.workflow_trigger = QLineEdit()
        self.workflow_trigger.setPlaceholderText("例如：导入课程录音后")
        form.addRow("触发条件", self.workflow_trigger)
        self.workflow_steps = QPlainTextEdit()
        self.workflow_steps.setPlaceholderText("每行一个步骤；这是可审查流程说明，不会执行任意代码")
        self.workflow_steps.setAccessibleName("SOP 步骤")
        form.addRow("步骤", self.workflow_steps)
        self.workflow_privacy = QLineEdit()
        self.workflow_privacy.setPlaceholderText("例如：原始音频仅本机；精校片段可发送 DeepSeek")
        form.addRow("隐私边界", self.workflow_privacy)
        workflow_actions = QHBoxLayout()
        new_workflow = QPushButton("新建")
        new_workflow.clicked.connect(self._new_workflow)
        workflow_actions.addWidget(new_workflow)
        save_workflow = QPushButton("保存 SOP")
        save_workflow.setObjectName("primary")
        save_workflow.clicked.connect(self._save_workflow)
        workflow_actions.addWidget(save_workflow)
        form.addRow("", workflow_actions)
        layout.addWidget(editor, 2)
        self.tabs.addTab(tab, "SOP 流程库")

    def _build_wiki_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        title = QLabel("便携 LLM Wiki")
        title.setStyleSheet("font-size:18px;font-weight:700;")
        layout.addWidget(title)
        explanation = QLabel(
            "SQLite 仍是唯一事实源；编译会生成带 frontmatter、双向链接、目录、标签索引、"
            "原始资料/成果状态页和操作日志的 Markdown 镜像。它可以直接复制、备份或用 Obsidian 打开。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.wiki_path = QLineEdit(str(self.controller.paths.notes / "LLM-Wiki"))
        self.wiki_path.setReadOnly(True)
        self.wiki_path.setAccessibleName("便携 Wiki 输出目录")
        layout.addWidget(self.wiki_path)
        self.wiki_status = QLabel("尚未在本次会话中编译")
        self.wiki_status.setObjectName("muted")
        self.wiki_status.setWordWrap(True)
        layout.addWidget(self.wiki_status)
        wiki_actions = QHBoxLayout()
        self.compile_wiki_button = QPushButton("重新编译 Wiki")
        self.compile_wiki_button.setObjectName("primary")
        self.compile_wiki_button.clicked.connect(self._compile_wiki)
        wiki_actions.addWidget(self.compile_wiki_button)
        open_button = QPushButton("打开 Wiki 目录")
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.wiki_path.text()))
        )
        wiki_actions.addWidget(open_button)
        wiki_actions.addStretch(1)
        layout.addLayout(wiki_actions)
        layout.addStretch(1)
        self.tabs.addTab(tab, "便携 Wiki")

    def refresh_all(self) -> None:
        self.refresh_proposals()
        self.refresh_quality()
        self.refresh_workflows()

    def refresh_proposals(self) -> None:
        self.proposal_list.clear()
        try:
            values = self.controller.knowledge_proposals()
        except (OSError, ValueError) as exc:
            self.proposal_summary.setText(f"候选读取失败：{exc}")
            return
        for value in values:
            duplicate = " · 发现同名/别名知识" if value.get("duplicate_item_id") else ""
            confidence = value.get("confidence")
            confidence_text = f" · 置信度 {float(confidence):.0%}" if confidence is not None else ""
            item = QListWidgetItem(
                f"{value.get('title') or '未命名候选'}\n"
                f"{value.get('proposed_type') or 'topic'}{confidence_text}{duplicate}"
            )
            item.setData(Qt.UserRole, value)
            self.proposal_list.addItem(item)
        self.proposal_summary.setText(f"待审核 {len(values)} 条 · 接受后仍标记为“需要复核”")
        if values:
            self.proposal_list.setCurrentRow(0)
        else:
            self.proposal_detail.setPlainText("暂无候选。完成一次启用知识卡片的深度精校后，会在这里出现。")
        self._update_proposal_actions()

    def _show_proposal(self, item: QListWidgetItem | None) -> None:
        value = item.data(Qt.UserRole) if item else None
        if not isinstance(value, dict):
            self.proposal_detail.clear()
            self._update_proposal_actions()
            return
        lines = [
            str(value.get("title") or "未命名候选"), "",
            str(value.get("body") or ""), "", "来源与审计", 
            f"- 来源资料：{value.get('source_document_id') or '未关联'}",
            f"- 精校任务：{value.get('correction_run_id') or '无'}",
            f"- 证据片段：{', '.join(value.get('source_segment_ids') or []) or '无'}",
            f"- 重复知识：{value.get('duplicate_item_id') or '未发现'}",
        ]
        self.proposal_detail.setPlainText("\n".join(lines))
        self._update_proposal_actions()

    def _update_proposal_actions(self) -> None:
        item = self.proposal_list.currentItem()
        value = item.data(Qt.UserRole) if item else None
        enabled = isinstance(value, dict)
        self.accept_proposal.setEnabled(enabled)
        self.reject_proposal.setEnabled(enabled)
        self.merge_proposal.setEnabled(bool(enabled and value.get("duplicate_item_id")))

    def _review_proposal(self, decision: str, merge: bool) -> None:
        item = self.proposal_list.currentItem()
        value = item.data(Qt.UserRole) if item else None
        if not isinstance(value, dict):
            return
        if decision == "reject":
            choice = QMessageBox.question(
                self, "拒绝候选", "确定把这条候选标记为已拒绝吗？原始资料和精校稿不会删除。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return
        self.accept_proposal.setEnabled(False)
        self.merge_proposal.setEnabled(False)
        self.reject_proposal.setEnabled(False)
        self.proposal_summary.setText("正在保存审核决定并重建便携 Wiki…")
        task = _Task(lambda: self.controller.review_knowledge_proposal(
            str(value["id"]), decision=decision, merge_duplicate=merge
        ))
        task.signals.result.connect(
            lambda _result: self._proposal_review_complete(decision, merge)
        )
        task.signals.error.connect(
            self._proposal_review_failed
        )
        task.signals.finished.connect(self._update_proposal_actions)
        self.thread_pool.start(task)

    def _proposal_review_complete(self, decision: str, merge: bool) -> None:
        self.refresh_proposals()
        self.refresh_quality()
        message = "候选已合并，并标记为需要复核" if merge else (
            "候选已保存为待复核正式知识" if decision == "accept" else "候选已拒绝"
        )
        QMessageBox.information(self, "处理完成", message)

    def _proposal_review_failed(self, message: str) -> None:
        self.proposal_summary.setText(f"候选处理失败：{message}")
        QMessageBox.warning(self, "候选处理失败", message)

    def refresh_quality(self) -> None:
        try:
            report = self.controller.source_quality_center()
        except (OSError, ValueError) as exc:
            self.quality_summary.setText(f"检查失败：{exc}")
            return
        issues = report.get("issues") if isinstance(report.get("issues"), list) else []
        contradictions = (
            report.get("contradictions") if isinstance(report.get("contradictions"), list) else []
        )
        duplicates = report.get("duplicates") if isinstance(report.get("duplicates"), list) else []
        rows = list(issues)
        rows.extend(
            {
                "severity": "warning",
                "title": f"{relation.get('source_title') or relation.get('source_item_id')} ↔ "
                         f"{relation.get('target_title') or relation.get('target_item_id')}",
                "message": str(relation.get("summary") or "存在明确冲突关系，请人工比较证据与适用范围"),
                "document_id": "",
            }
            for relation in contradictions
        )
        rows.extend(
            {
                "severity": "warning",
                "title": " / ".join(str(title) for title in duplicate.get("titles") or []),
                "message": f"检测到 {int(duplicate.get('document_count') or 0)} 份内容指纹相同的资料",
                "document_id": "",
            }
            for duplicate in duplicates
        )
        self.quality_table.setRowCount(len(rows))
        for row, issue in enumerate(rows):
            values = (
                "错误" if issue.get("severity") == "error" else "提醒",
                str(issue.get("title") or "未命名资料"),
                str(issue.get("message") or ""),
                str(issue.get("document_id") or ""),
            )
            for column, value in enumerate(values):
                self.quality_table.setItem(row, column, QTableWidgetItem(value))
        counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
        self.quality_summary.setText(
            f"来源问题 {counts.get('issues', 0)} 项 · 明确冲突关系 {counts.get('contradictions', 0)} 条 · "
            f"重复资料组 {counts.get('duplicate_groups', 0)} 组。所有判断都保留为可人工复核状态。"
        )
        if rows:
            self.quality_table.selectRow(0)

    def _selected_quality_document(self) -> str | None:
        row = self.quality_table.currentRow()
        item = self.quality_table.item(row, 3) if row >= 0 else None
        return item.text().strip() if item and item.text().strip() else None

    def _quality_selection_changed(self) -> None:
        document_id = self._selected_quality_document()
        self.save_assessment.setEnabled(bool(document_id))
        if not document_id:
            return
        try:
            value = self.controller.source_assessment(document_id)
        except (OSError, ValueError):
            return
        for combo, key in ((self.source_class, "source_class"), (self.source_reliability, "reliability")):
            index = combo.findData(value.get(key))
            if index >= 0:
                combo.setCurrentIndex(index)
        self.source_valid_until.setText(str(value.get("valid_until") or ""))
        self.source_notes.setText(str(value.get("notes") or ""))

    def _save_source_assessment(self) -> None:
        document_id = self._selected_quality_document()
        if not document_id:
            return
        valid_until = self.source_valid_until.text().strip()
        if valid_until and (len(valid_until) != 10 or valid_until[4:5] != "-" or valid_until[7:8] != "-"):
            QMessageBox.warning(self, "日期格式不正确", "有效期请使用 YYYY-MM-DD，或留空。")
            self.source_valid_until.setFocus()
            return
        try:
            current = self.controller.source_assessment(document_id)
            self.controller.save_source_assessment(
                document_id,
                source_class=str(self.source_class.currentData()),
                reliability=str(self.source_reliability.currentData()),
                extraction_completeness=current.get("extraction_completeness"),
                published_at=current.get("published_at"),
                valid_until=valid_until or None,
                notes=self.source_notes.text().strip(),
                checked=True,
                metadata=current.get("metadata") if isinstance(current.get("metadata"), dict) else {},
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法保存来源评估", str(exc))
            return
        self.refresh_quality()

    def _choose_evaluation_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择黄金问题集", str(Path.home()), "JSON (*.json)")
        if path:
            self.eval_path.setText(path)

    def _run_evaluation(self) -> None:
        path = self.eval_path.text().strip()
        if not path:
            QMessageBox.warning(self, "还需要数据集", "请先选择黄金问题集 JSON。")
            return
        self.run_eval.setEnabled(False)
        self.eval_status.setText("正在运行评测；可继续查看其他标签页…")
        task = _Task(lambda: self.controller.run_golden_evaluation(
            path,
            top_k=int(self.eval_top_k.value()),
            evaluate_citations=self.eval_citations.isChecked(),
        ))
        task.signals.result.connect(self._evaluation_complete)
        task.signals.error.connect(lambda message: QMessageBox.warning(self, "评测失败", message))
        task.signals.finished.connect(lambda: self.run_eval.setEnabled(True))
        self.thread_pool.start(task)

    def _evaluation_complete(self, report: object) -> None:
        if not isinstance(report, dict):
            return
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        metric_text = " · ".join(f"{key} {float(value):.1%}" for key, value in metrics.items())
        self.eval_status.setText(
            f"完成 {report.get('case_count', 0)} 个用例 · {metric_text or '无可用指标'}"
        )
        cases = report.get("cases") if isinstance(report.get("cases"), list) else []
        self.eval_table.setRowCount(len(cases))
        for row, case in enumerate(cases):
            values = (
                str(case.get("case_id") or ""), str(case.get("query") or ""),
                "命中" if case.get("hit") else "未命中",
                str(case.get("first_relevant_rank") or "—"),
                f"{float(case.get('reciprocal_rank') or 0):.3f}",
            )
            for column, value in enumerate(values):
                self.eval_table.setItem(row, column, QTableWidgetItem(value))

    def refresh_workflows(self) -> None:
        self.workflow_list.clear()
        try:
            values = self.controller.workflows(include_archived=True)
        except (OSError, ValueError):
            values = []
        for value in values:
            item = QListWidgetItem(f"{value.get('name') or '未命名 SOP'}\n{value.get('status') or 'current'}")
            item.setData(Qt.UserRole, value)
            self.workflow_list.addItem(item)
        if values:
            self.workflow_list.setCurrentRow(0)

    def _show_workflow(self, item: QListWidgetItem | None) -> None:
        value = item.data(Qt.UserRole) if item else None
        if not isinstance(value, dict):
            return
        self.workflow_name.setText(str(value.get("name") or ""))
        self.workflow_description.setText(str(value.get("description") or ""))
        trigger = value.get("trigger") if isinstance(value.get("trigger"), dict) else {}
        self.workflow_trigger.setText(str(trigger.get("description") or ""))
        self.workflow_steps.setPlainText("\n".join(str(step) for step in value.get("steps") or []))
        privacy = value.get("privacy") if isinstance(value.get("privacy"), dict) else {}
        self.workflow_privacy.setText(str(privacy.get("boundary") or ""))

    def _new_workflow(self) -> None:
        self.workflow_list.clearSelection()
        self.workflow_name.clear()
        self.workflow_description.clear()
        self.workflow_trigger.clear()
        self.workflow_steps.clear()
        self.workflow_privacy.clear()
        self.workflow_name.setFocus()

    def _save_workflow(self) -> None:
        steps = [line.strip() for line in self.workflow_steps.toPlainText().splitlines() if line.strip()]
        if not self.workflow_name.text().strip() or not steps:
            QMessageBox.warning(self, "流程不完整", "请填写名称，并至少添加一个步骤。")
            return
        current = self.workflow_list.currentItem()
        value = current.data(Qt.UserRole) if current else None
        try:
            self.controller.save_workflow(
                workflow_id=str(value.get("id")) if isinstance(value, dict) else None,
                name=self.workflow_name.text().strip(),
                description=self.workflow_description.text().strip(),
                trigger={"description": self.workflow_trigger.text().strip()},
                steps=steps,
                model_policy={"mode": "documented-only"},
                privacy={"boundary": self.workflow_privacy.text().strip()},
                status=str(value.get("status") or "current") if isinstance(value, dict) else "current",
            )
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "无法保存 SOP", str(exc))
            return
        self.refresh_workflows()

    def _compile_wiki(self) -> None:
        self.compile_wiki_button.setEnabled(False)
        self.wiki_status.setText("正在后台编译 Markdown、关系索引和链接检查…")
        task = _Task(self.controller.compile_portable_wiki)
        task.signals.result.connect(self._wiki_compile_complete)
        task.signals.error.connect(
            self._wiki_compile_failed
        )
        task.signals.finished.connect(
            lambda: self.compile_wiki_button.setEnabled(True)
        )
        self.thread_pool.start(task)

    def _wiki_compile_complete(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
        self.wiki_status.setText(
            f"已编译 {result.get('item_count', 0)} 条知识、{result.get('relation_count', 0)} 条关系、"
            f"{len(result.get('written_files') or [])} 个 Markdown 文件。"
            + (f" 另有 {len(warnings)} 项治理提醒（主要是孤立知识）。" if warnings else " 链接检查通过。")
        )

    def _wiki_compile_failed(self, message: str) -> None:
        self.wiki_status.setText(f"Wiki 编译失败：{message}")
        QMessageBox.warning(self, "Wiki 编译失败", message)


__all__ = ["KnowledgeOperationsDialog"]
