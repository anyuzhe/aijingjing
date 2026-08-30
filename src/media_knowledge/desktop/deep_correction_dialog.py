from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
import html
import time
from urllib.parse import urlparse


DEEP_CORRECTION_STEPS = (
    "原始转写与时间轴校验",
    "音频质量与失败区段检测",
    "碎片合并与章节切分",
    "说话人轮次复核",
    "专业术语候选提取",
    "上下文语义精校",
    "外部证据交叉核验",
    "全文术语与指代一致性检查",
    "不确定内容与可信度标注",
    "方法论、知识卡与架构图整理",
    "完整性、引用与幻觉门禁",
)

HIGH_CONFIDENCE_THRESHOLD = 0.85
CHANGE_STATUSES = frozenset({"pending", "accepted", "rejected"})


def _bounded_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return min(1.0, max(0.0, confidence))


def format_elapsed(seconds: object) -> str:
    """Format elapsed seconds for a stable, screen-reader-friendly label."""

    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        total = 0
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_correction_timestamp(milliseconds: object) -> str:
    try:
        total_seconds = max(0, int(float(milliseconds)) // 1000)
    except (TypeError, ValueError, OverflowError):
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_time_range(start_ms: object, end_ms: object) -> str:
    return f"{format_correction_timestamp(start_ms)}–{format_correction_timestamp(end_ms)}"


def confidence_text(value: object) -> str:
    confidence = _bounded_confidence(value)
    level = "高置信度" if confidence >= HIGH_CONFIDENCE_THRESHOLD else (
        "中置信度" if confidence >= 0.65 else "低置信度"
    )
    return f"{level} · {confidence:.0%}"


def is_navigable_evidence_url(url: object) -> bool:
    candidate = str(url or "").strip()
    if not candidate:
        return False
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    return parsed.scheme == "file" and bool(parsed.path)


@dataclass(frozen=True, slots=True)
class CorrectionEvidence:
    label: str
    url: str

    def __post_init__(self) -> None:
        label = str(self.label or "").strip()
        url = str(self.url or "").strip()
        object.__setattr__(self, "label", label or url or "未命名证据")
        object.__setattr__(self, "url", url)

    @property
    def navigable(self) -> bool:
        return is_navigable_evidence_url(self.url)


@dataclass(frozen=True, slots=True)
class CorrectionChange:
    id: str
    start_ms: int
    end_ms: int
    speaker: str
    raw_text: str
    corrected_text: str
    confidence: float
    uncertain: bool = False
    uncertainty_reason: str = ""
    evidence: tuple[CorrectionEvidence, ...] = ()
    rationale: str = ""
    status: str = "pending"

    def __post_init__(self) -> None:
        change_id = str(self.id or "").strip()
        if not change_id:
            raise ValueError("精校变更 ID 不能为空")
        start_ms = max(0, int(self.start_ms))
        end_ms = max(start_ms, int(self.end_ms))
        status = str(self.status or "pending").strip().lower()
        if status not in CHANGE_STATUSES:
            raise ValueError(f"未知精校决策状态：{status}")
        evidence = tuple(
            item if isinstance(item, CorrectionEvidence) else CorrectionEvidence(*item)
            for item in self.evidence
        )
        object.__setattr__(self, "id", change_id)
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "end_ms", end_ms)
        object.__setattr__(self, "speaker", str(self.speaker or "").strip())
        object.__setattr__(self, "raw_text", str(self.raw_text or ""))
        object.__setattr__(self, "corrected_text", str(self.corrected_text or ""))
        object.__setattr__(self, "confidence", _bounded_confidence(self.confidence))
        object.__setattr__(self, "uncertainty_reason", str(self.uncertainty_reason or "").strip())
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "rationale", str(self.rationale or "").strip())
        object.__setattr__(self, "status", status)

    @property
    def time_range(self) -> str:
        return format_time_range(self.start_ms, self.end_ms)

    @property
    def navigable_evidence(self) -> tuple[CorrectionEvidence, ...]:
        return tuple(item for item in self.evidence if item.navigable)


def change_is_bulk_eligible(
    change: CorrectionChange,
    *,
    threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> bool:
    """Only untouched, evidenced, certain, high-confidence changes are safe in bulk."""

    return bool(
        change.status == "pending"
        and not change.uncertain
        and change.confidence >= _bounded_confidence(threshold)
        and change.navigable_evidence
    )


try:
    from PySide6.QtCore import QSize, Qt, QTimer, Signal, Slot
    from PySide6.QtGui import QColor, QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QTextBrowser,
        QVBoxLayout,
        QWidget,
    )

    QT_WIDGETS_AVAILABLE = True
    QT_WIDGETS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - desktop dependency is optional
    QT_WIDGETS_AVAILABLE = False
    QT_WIDGETS_IMPORT_ERROR = exc


if QT_WIDGETS_AVAILABLE:
    _STYLE = """
    QDialog#deepCorrectionDialog { background: #eaf5fb; }
    QFrame#summaryCard, QFrame#progressCard, QFrame#changeCard,
    QFrame#detailCard, QFrame#textCard, QFrame#errorCard {
      background: #fbfdff;
      border: 1px solid #bfd9e7;
      border-radius: 12px;
    }
    QFrame#errorCard { background: #fff4f5; border-color: #ddb8c1; }
    QLabel#dialogTitle { color: #153f58; font-size: 19px; font-weight: 700; }
    QLabel#sectionTitle { color: #285b78; font-size: 13px; font-weight: 700; }
    QLabel#mutedText { color: #4b6c7e; font-size: 12px; }
    QLabel#stateIdle, QLabel#stateCancelled {
      color: #365e73; background: #e6f1f6; border: 1px solid #bfd5df;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#stateRunning {
      color: #155b78; background: #dff3fb; border: 1px solid #a9d7e8;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#stateFailed {
      color: #8a3248; background: #fbe7eb; border: 1px solid #ddb2bd;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#stateCompleted {
      color: #1f6657; background: #e2f3ee; border: 1px solid #afd7cc;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#stepWaiting, QLabel#stepDone, QLabel#stepCurrent, QLabel#stepFailed {
      border-radius: 8px; padding: 6px 8px; font-size: 11px;
    }
    QLabel#stepWaiting { color: #557484; background: #f0f6f8; border: 1px solid #d6e4ea; }
    QLabel#stepDone { color: #266455; background: #e6f4f0; border: 1px solid #c5e0d8; }
    QLabel#stepCurrent { color: #155d7e; background: #dff3fb; border: 2px solid #4b9fc2; font-weight: 650; }
    QLabel#stepFailed { color: #8a3348; background: #fbe8ec; border: 2px solid #c66e82; font-weight: 650; }
    QListWidget, QPlainTextEdit, QTextBrowser {
      background: #fcfeff; color: #24475b; border: 1px solid #bfd8e5;
      border-radius: 9px; selection-background-color: #d5edf7;
      selection-color: #123f59;
    }
    QListWidget { outline: none; padding: 5px; }
    QListWidget::item { min-height: 62px; padding: 7px 8px; border-radius: 8px; }
    QListWidget::item:hover { background: #ecf7fb; }
    QListWidget::item:selected { background: #d5edf7; color: #17475f; }
    QPlainTextEdit#rawText { background: #f1f6f8; color: #476574; }
    QTextBrowser { padding: 5px 7px; }
    QProgressBar {
      min-height: 15px; border: 1px solid #bfd6e1; border-radius: 7px;
      background: #e3eef3; color: #244b61; text-align: center;
    }
    QProgressBar::chunk { background: #4a9dc0; border-radius: 6px; }
    QPushButton {
      min-height: 32px; background: #fbfdff; color: #24475b;
      border: 1px solid #b9d4e1; border-radius: 8px; padding: 5px 11px;
    }
    QPushButton:hover { background: #e4f3f9; border-color: #75b3ce; }
    QPushButton:pressed { background: #d2eaf4; }
    QPushButton:focus, QListWidget:focus, QPlainTextEdit:focus, QTextBrowser:focus {
      border: 2px solid #287fa7;
    }
    QPushButton:disabled { color: #839ba7; background: #edf3f6; border-color: #d5e2e8; }
    QPushButton#primaryButton { color: white; font-weight: 650; background: #216f96; border-color: #216f96; }
    QPushButton#primaryButton:hover { background: #2c82aa; border-color: #2c82aa; }
    QPushButton#primaryButton:disabled {
      color: #8199a5; background: #e9f0f3; border-color: #d1dfe5; font-weight: 500;
    }
    QPushButton#dangerButton { color: #84364b; background: #fff8f9; border-color: #d8aeba; }
    QPushButton#dangerButton:disabled { color: #8b9da6; background: #edf3f6; border-color: #d5e2e8; }
    QLabel#decisionPending { color: #5f6170; }
    QLabel#decisionAccepted { color: #236454; }
    QLabel#decisionRejected { color: #8b3c50; }
    """

    _STATE_TEXT = {
        "idle": "尚未开始",
        "running": "正在精校",
        "cancelling": "正在安全取消",
        "failed": "精校失败",
        "completed": "精校完成",
        "cancelled": "已取消",
    }
    _STATE_OBJECT = {
        "idle": "stateIdle",
        "running": "stateRunning",
        "cancelling": "stateRunning",
        "failed": "stateFailed",
        "completed": "stateCompleted",
        "cancelled": "stateCancelled",
    }
    _DECISION_TEXT = {
        "pending": "待确认",
        "accepted": "已接受",
        "rejected": "已拒绝",
    }


    class DeepCorrectionDialog(QDialog):
        """Backend-neutral review surface for a multi-stage deep correction run."""

        startRequested = Signal()
        cancelRequested = Signal()
        retryRequested = Signal()
        exportRequested = Signal()
        acceptRequested = Signal(str)
        rejectRequested = Signal(str)

        def __init__(
            self,
            *,
            source_name: str = "",
            raw_text: str = "",
            corrected_text: str = "",
            changes: Iterable[CorrectionChange] = (),
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._state = "idle"
            self._completed_steps = 0
            self._current_step: int | None = None
            self._progress_detail = ""
            self._elapsed_base_seconds = 0
            self._elapsed_started_at: float | None = None
            self._expanded_text = False
            self._document_raw_text = ""
            self._document_corrected_text = ""
            self._changes: dict[str, CorrectionChange] = {}
            self._change_order: list[str] = []
            self._shortcuts: list[QShortcut] = []

            self.setObjectName("deepCorrectionDialog")
            self.setWindowTitle("深度精校 · AI静静")
            self.setMinimumSize(980, 720)
            self.resize(1240, 860)
            self.setModal(False)
            self.setStyleSheet(_STYLE)

            self._elapsed_timer = QTimer(self)
            self._elapsed_timer.setInterval(1000)
            self._elapsed_timer.timeout.connect(self._refresh_elapsed)

            self._build_ui()
            self._connect_signals()
            self._install_shortcuts()
            self.set_document(source_name, raw_text, corrected_text)
            self.set_changes(changes)
            self.mark_idle()

        @property
        def state(self) -> str:
            return self._state

        @property
        def elapsed_seconds(self) -> int:
            elapsed = self._elapsed_base_seconds
            if self._elapsed_started_at is not None:
                elapsed += max(0, int(time.monotonic() - self._elapsed_started_at))
            return elapsed

        @property
        def selected_change_id(self) -> str | None:
            item = self.change_list.currentItem()
            return str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

        @property
        def changes(self) -> tuple[CorrectionChange, ...]:
            return tuple(self._changes[change_id] for change_id in self._change_order)

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(11)

            summary = QFrame()
            summary.setObjectName("summaryCard")
            summary_layout = QVBoxLayout(summary)
            summary_layout.setContentsMargins(14, 11, 14, 11)
            summary_layout.setSpacing(5)
            title_row = QHBoxLayout()
            self.title_label = QLabel("深度精校工作台")
            self.title_label.setObjectName("dialogTitle")
            self.title_label.setAccessibleName("深度精校工作台标题")
            title_row.addWidget(self.title_label, 1)
            self.state_label = QLabel()
            self.state_label.setAccessibleName("深度精校运行状态")
            title_row.addWidget(self.state_label)
            summary_layout.addLayout(title_row)
            self.source_label = QLabel()
            self.source_label.setObjectName("mutedText")
            self.source_label.setWordWrap(True)
            self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.source_label.setAccessibleName("当前精校资料")
            summary_layout.addWidget(self.source_label)
            note = QLabel("原始识别始终只读；每项改动都显示时间、说话人、置信度、不确定标记和证据。")
            note.setObjectName("mutedText")
            note.setWordWrap(True)
            note.setAccessibleName("深度精校安全说明")
            summary_layout.addWidget(note)
            root.addWidget(summary)

            progress_frame = QFrame()
            progress_frame.setObjectName("progressCard")
            progress_layout = QVBoxLayout(progress_frame)
            progress_layout.setContentsMargins(13, 10, 13, 11)
            progress_layout.setSpacing(7)
            progress_header = QHBoxLayout()
            self.current_step_label = QLabel("等待开始 · 共 11 步")
            self.current_step_label.setObjectName("sectionTitle")
            self.current_step_label.setWordWrap(True)
            self.current_step_label.setAccessibleName("当前深度精校步骤")
            progress_header.addWidget(self.current_step_label, 1)
            self.elapsed_label = QLabel("已用时 00:00")
            self.elapsed_label.setObjectName("mutedText")
            self.elapsed_label.setAccessibleName("深度精校已用时间")
            progress_header.addWidget(self.elapsed_label)
            progress_layout.addLayout(progress_header)
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, len(DEEP_CORRECTION_STEPS))
            self.progress_bar.setFormat("%v / %m 步完成")
            self.progress_bar.setAccessibleName("深度精校总进度")
            progress_layout.addWidget(self.progress_bar)
            self.step_grid = QGridLayout()
            self.step_grid.setHorizontalSpacing(7)
            self.step_grid.setVerticalSpacing(6)
            self.step_labels: list[QLabel] = []
            for index, step in enumerate(DEEP_CORRECTION_STEPS):
                label = QLabel()
                label.setWordWrap(True)
                label.setMinimumHeight(32)
                self.step_grid.addWidget(label, index // 3, index % 3)
                self.step_labels.append(label)
            progress_layout.addLayout(self.step_grid)
            root.addWidget(progress_frame)

            main_splitter = QSplitter(Qt.Orientation.Horizontal)
            main_splitter.setChildrenCollapsible(False)
            main_splitter.setAccessibleName("精校变更与证据详情分栏")

            change_frame = QFrame()
            change_frame.setObjectName("changeCard")
            change_layout = QVBoxLayout(change_frame)
            change_layout.setContentsMargins(11, 10, 11, 10)
            change_layout.setSpacing(7)
            change_header = QHBoxLayout()
            self.change_count_label = QLabel("变更：0 项")
            self.change_count_label.setObjectName("sectionTitle")
            self.change_count_label.setAccessibleName("精校变更统计")
            change_header.addWidget(self.change_count_label, 1)
            self.bulk_accept_button = QPushButton("批量接受：0 项")
            self.bulk_accept_button.setAccessibleName("批量接受高置信且有证据的变更")
            self.bulk_accept_button.setAccessibleDescription(
                "只接受未标记不确定、置信度不低于百分之八十五且带有效证据链接的待确认项"
            )
            self.bulk_accept_button.setToolTip("仅处理高置信、有证据且无不确定标记的待确认项（Ctrl+Alt+A）")
            change_header.addWidget(self.bulk_accept_button)
            change_layout.addLayout(change_header)
            self.change_list = QListWidget()
            self.change_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.change_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.change_list.setAccessibleName("深度精校变更列表")
            self.change_list.setAccessibleDescription("使用上下方向键选择变更，右侧查看原始文字、精校文字和证据")
            change_layout.addWidget(self.change_list, 1)
            decision_row = QHBoxLayout()
            self.accept_button = QPushButton("接受所选")
            self.accept_button.setObjectName("primaryButton")
            self.accept_button.setAccessibleName("接受当前精校变更")
            self.accept_button.setToolTip("接受当前变更（Ctrl+Shift+A）")
            decision_row.addWidget(self.accept_button)
            self.reject_button = QPushButton("拒绝所选")
            self.reject_button.setObjectName("dangerButton")
            self.reject_button.setAccessibleName("拒绝当前精校变更")
            self.reject_button.setToolTip("拒绝当前变更（Ctrl+Shift+R）")
            decision_row.addWidget(self.reject_button)
            change_layout.addLayout(decision_row)
            main_splitter.addWidget(change_frame)

            detail_frame = QFrame()
            detail_frame.setObjectName("detailCard")
            detail_layout = QVBoxLayout(detail_frame)
            detail_layout.setContentsMargins(11, 10, 11, 10)
            detail_layout.setSpacing(7)
            detail_header = QHBoxLayout()
            detail_title = QLabel("变更证据详情")
            detail_title.setObjectName("sectionTitle")
            detail_header.addWidget(detail_title, 1)
            self.toggle_text_button = QPushButton("展开长文本")
            self.toggle_text_button.setAccessibleName("展开原始与精校长文本")
            self.toggle_text_button.setToolTip("展开或收起完整文本区域（Ctrl+L）")
            detail_header.addWidget(self.toggle_text_button)
            detail_layout.addLayout(detail_header)

            metadata_grid = QGridLayout()
            metadata_grid.setHorizontalSpacing(12)
            metadata_grid.setVerticalSpacing(4)
            self.time_range_label = QLabel("时间：—")
            self.time_range_label.setAccessibleName("当前变更时间范围")
            self.speaker_label = QLabel("说话人：—")
            self.speaker_label.setAccessibleName("当前变更说话人")
            self.confidence_label = QLabel("置信度：—")
            self.confidence_label.setAccessibleName("当前变更置信度")
            self.uncertainty_label = QLabel("不确定标记：—")
            self.uncertainty_label.setWordWrap(True)
            self.uncertainty_label.setAccessibleName("当前变更不确定标记")
            metadata_grid.addWidget(self.time_range_label, 0, 0)
            metadata_grid.addWidget(self.speaker_label, 0, 1)
            metadata_grid.addWidget(self.confidence_label, 1, 0)
            metadata_grid.addWidget(self.uncertainty_label, 1, 1)
            detail_layout.addLayout(metadata_grid)

            self.rationale_label = QLabel("校正依据：请选择一项变更。")
            self.rationale_label.setObjectName("mutedText")
            self.rationale_label.setWordWrap(True)
            self.rationale_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.rationale_label.setAccessibleName("当前变更校正依据")
            detail_layout.addWidget(self.rationale_label)

            self.evidence_browser = QTextBrowser()
            self.evidence_browser.setOpenExternalLinks(True)
            self.evidence_browser.setMaximumHeight(82)
            self.evidence_browser.setAccessibleName("当前变更证据链接")
            self.evidence_browser.setAccessibleDescription("链接可用键盘聚焦并在系统浏览器中打开")
            detail_layout.addWidget(self.evidence_browser)

            text_splitter = QSplitter(Qt.Orientation.Horizontal)
            text_splitter.setChildrenCollapsible(False)
            text_splitter.setAccessibleName("原始文字与精校文字并排对照")
            raw_card = QFrame()
            raw_card.setObjectName("textCard")
            raw_layout = QVBoxLayout(raw_card)
            raw_layout.setContentsMargins(8, 8, 8, 8)
            raw_heading = QLabel("原始识别 · 只读证据")
            raw_heading.setObjectName("sectionTitle")
            raw_layout.addWidget(raw_heading)
            self.raw_editor = QPlainTextEdit()
            self.raw_editor.setObjectName("rawText")
            self.raw_editor.setReadOnly(True)
            self.raw_editor.setAccessibleName("当前变更原始识别文字，只读")
            self.raw_editor.setAccessibleDescription("精校不会覆盖这份原始识别证据")
            raw_layout.addWidget(self.raw_editor, 1)
            text_splitter.addWidget(raw_card)

            corrected_card = QFrame()
            corrected_card.setObjectName("textCard")
            corrected_layout = QVBoxLayout(corrected_card)
            corrected_layout.setContentsMargins(8, 8, 8, 8)
            corrected_heading = QLabel("深度精校结果 · 待确认")
            corrected_heading.setObjectName("sectionTitle")
            corrected_layout.addWidget(corrected_heading)
            self.corrected_editor = QPlainTextEdit()
            self.corrected_editor.setReadOnly(True)
            self.corrected_editor.setAccessibleName("当前变更深度精校文字，只读")
            self.corrected_editor.setAccessibleDescription("通过接受或拒绝按钮决定是否采纳此项精校")
            corrected_layout.addWidget(self.corrected_editor, 1)
            text_splitter.addWidget(corrected_card)
            text_splitter.setSizes([500, 500])
            detail_layout.addWidget(text_splitter, 1)
            self.text_splitter = text_splitter
            main_splitter.addWidget(detail_frame)
            main_splitter.setSizes([410, 790])
            main_splitter.setStretchFactor(0, 1)
            main_splitter.setStretchFactor(1, 2)
            root.addWidget(main_splitter, 1)

            self.error_frame = QFrame()
            self.error_frame.setObjectName("errorCard")
            error_layout = QVBoxLayout(self.error_frame)
            error_layout.setContentsMargins(12, 9, 12, 9)
            error_heading = QLabel("处理失败，可恢复")
            error_heading.setObjectName("sectionTitle")
            error_layout.addWidget(error_heading)
            self.error_label = QLabel()
            self.error_label.setWordWrap(True)
            self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.error_label.setAccessibleName("深度精校错误信息")
            error_layout.addWidget(self.error_label)
            self.recovery_label = QLabel()
            self.recovery_label.setWordWrap(True)
            self.recovery_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.recovery_label.setAccessibleName("深度精校失败恢复方式")
            error_layout.addWidget(self.recovery_label)
            root.addWidget(self.error_frame)

            footer = QHBoxLayout()
            self.feedback_label = QLabel("点击“开始精校”运行 11 步深度校正。")
            self.feedback_label.setObjectName("mutedText")
            self.feedback_label.setWordWrap(True)
            self.feedback_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.feedback_label.setAccessibleName("深度精校操作反馈")
            footer.addWidget(self.feedback_label, 1)
            self.start_button = QPushButton("开始精校")
            self.start_button.setObjectName("primaryButton")
            self.start_button.setAccessibleName("开始深度精校")
            self.start_button.setToolTip("开始 11 步深度精校（Ctrl+Enter）")
            footer.addWidget(self.start_button)
            self.cancel_button = QPushButton("取消任务")
            self.cancel_button.setObjectName("dangerButton")
            self.cancel_button.setAccessibleName("取消正在运行的深度精校")
            self.cancel_button.setToolTip("请求安全取消，已完成结果不会丢失（Esc）")
            footer.addWidget(self.cancel_button)
            self.retry_button = QPushButton("重新尝试")
            self.retry_button.setAccessibleName("按照恢复方式重新尝试深度精校")
            self.retry_button.setToolTip("保留已用时间并重新尝试（Ctrl+R）")
            footer.addWidget(self.retry_button)
            self.export_button = QPushButton("导出精校稿")
            self.export_button.setAccessibleName("导出完成的深度精校稿")
            self.export_button.setToolTip("导出确认后的完整精校稿（Ctrl+Shift+E）")
            footer.addWidget(self.export_button)
            self.close_button = QPushButton("关闭")
            self.close_button.setAccessibleName("关闭深度精校工作台")
            footer.addWidget(self.close_button)
            root.addLayout(footer)

            self._set_text_expanded(False)
            self.error_frame.hide()
            self._set_tab_order()

        def _connect_signals(self) -> None:
            self.start_button.clicked.connect(self._start_clicked)
            self.cancel_button.clicked.connect(self._cancel_clicked)
            self.retry_button.clicked.connect(self._retry_clicked)
            self.export_button.clicked.connect(self._export_clicked)
            self.close_button.clicked.connect(self.reject)
            self.bulk_accept_button.clicked.connect(self.accept_eligible_bulk)
            self.accept_button.clicked.connect(self.accept_selected)
            self.reject_button.clicked.connect(self.reject_selected)
            self.toggle_text_button.clicked.connect(self.toggle_text_expansion)
            self.change_list.currentItemChanged.connect(self._on_change_selected)

        def _install_shortcuts(self) -> None:
            for sequence, callback in (
                ("Ctrl+Return", self._start_clicked),
                ("Ctrl+R", self._retry_clicked),
                ("Ctrl+Shift+E", self._export_clicked),
                ("Ctrl+Shift+A", self.accept_selected),
                ("Ctrl+Shift+R", self.reject_selected),
                ("Ctrl+Alt+A", self.accept_eligible_bulk),
                ("Ctrl+L", self.toggle_text_expansion),
            ):
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)

        def _set_tab_order(self) -> None:
            ordered = (
                self.start_button,
                self.cancel_button,
                self.retry_button,
                self.export_button,
                self.change_list,
                self.bulk_accept_button,
                self.accept_button,
                self.reject_button,
                self.evidence_browser,
                self.raw_editor,
                self.corrected_editor,
                self.toggle_text_button,
                self.close_button,
            )
            for first, second in zip(ordered, ordered[1:]):
                self.setTabOrder(first, second)

        def set_document(self, source_name: str, raw_text: str, corrected_text: str) -> None:
            source = str(source_name or "").strip() or "未命名资料"
            self._document_raw_text = str(raw_text or "")
            self._document_corrected_text = str(corrected_text or "")
            self.source_label.setText(f"当前资料：{source}")
            self.source_label.setAccessibleDescription(f"正在处理 {source}")
            if not self._changes:
                self.raw_editor.setPlainText(self._document_raw_text)
                self.corrected_editor.setPlainText(self._document_corrected_text)

        def set_changes(self, changes: Iterable[CorrectionChange]) -> None:
            selected = self.selected_change_id if hasattr(self, "change_list") else None
            self._changes = {}
            self._change_order = []
            for change in changes:
                if not isinstance(change, CorrectionChange):
                    raise TypeError("changes 必须由 CorrectionChange 组成")
                if change.id in self._changes:
                    raise ValueError(f"精校变更 ID 重复：{change.id}")
                self._changes[change.id] = change
                self._change_order.append(change.id)
            self._refresh_change_list(selected)

        def set_change_status(self, change_id: str, status: str) -> CorrectionChange:
            normalized = str(status or "").strip().lower()
            if normalized not in CHANGE_STATUSES:
                raise ValueError(f"未知精校决策状态：{normalized}")
            change = self._changes[str(change_id)]
            updated = replace(change, status=normalized)
            self._changes[updated.id] = updated
            self._refresh_change_list(updated.id)
            return updated

        def eligible_bulk_change_ids(self) -> tuple[str, ...]:
            return tuple(
                change_id
                for change_id in self._change_order
                if change_is_bulk_eligible(self._changes[change_id])
            )

        def set_progress(
            self,
            completed_steps: int,
            *,
            current_step: int | None = None,
            detail: str = "",
        ) -> None:
            completed = min(len(DEEP_CORRECTION_STEPS), max(0, int(completed_steps)))
            if current_step is None and completed < len(DEEP_CORRECTION_STEPS):
                current = completed + 1
            elif current_step is None:
                current = None
            else:
                current = min(len(DEEP_CORRECTION_STEPS), max(1, int(current_step)))
            self._completed_steps = completed
            self._current_step = current
            self._progress_detail = str(detail or "").strip()
            self._refresh_progress()

        def set_elapsed_seconds(self, seconds: object) -> None:
            self._elapsed_base_seconds = max(0, int(_bounded_nonnegative(seconds)))
            if self._elapsed_started_at is not None:
                self._elapsed_started_at = time.monotonic()
            self._refresh_elapsed()

        def mark_idle(self) -> None:
            self._stop_elapsed()
            self._elapsed_base_seconds = 0
            self._state = "idle"
            self._completed_steps = 0
            self._current_step = None
            self._progress_detail = ""
            self.error_frame.hide()
            self.feedback_label.setText("点击“开始精校”运行 11 步深度校正。")
            self._apply_state()

        def mark_running(self, *, reset_elapsed: bool = True) -> None:
            if reset_elapsed:
                self._elapsed_base_seconds = 0
                self._completed_steps = 0
                self._current_step = 1
                self._progress_detail = ""
            self._state = "running"
            self.error_frame.hide()
            self._start_elapsed()
            self.feedback_label.setText("正在处理；可以安全取消，原始转写和已完成步骤不会被覆盖。")
            self._apply_state()

        def mark_cancelling(self) -> None:
            self._state = "cancelling"
            self.feedback_label.setText("已请求安全取消，正在等待当前原子步骤结束。")
            self._apply_state()

        def mark_cancelled(self, message: str = "任务已取消；可重新开始。") -> None:
            self._stop_elapsed()
            self._state = "cancelled"
            self.feedback_label.setText(str(message or "任务已取消；可重新开始。"))
            self._apply_state()

        def mark_failed(self, error: str, recovery_path: str = "") -> None:
            self._stop_elapsed()
            self._state = "failed"
            message = str(error or "发生未知错误。").strip()
            recovery = str(recovery_path or "").strip() or (
                "检查模型、网络或密钥配置后点击“重新尝试”；原始转写与已完成步骤仍会保留。"
            )
            self.error_label.setText(f"错误：{message}")
            self.recovery_label.setText(f"恢复方式：{recovery}")
            self.error_frame.show()
            self.feedback_label.setText("精校未完成。请按上方恢复方式处理后重新尝试。")
            self._apply_state()
            self.retry_button.setFocus(Qt.FocusReason.OtherFocusReason)

        def mark_completed(self) -> None:
            self._stop_elapsed()
            self._state = "completed"
            self._completed_steps = len(DEEP_CORRECTION_STEPS)
            self._current_step = None
            self._progress_detail = "全部门禁已通过"
            self.error_frame.hide()
            self.feedback_label.setText("深度精校已完成。请逐项确认变更，确认后即可导出。")
            self._apply_state()

        def _start_elapsed(self) -> None:
            if self._elapsed_started_at is None:
                self._elapsed_started_at = time.monotonic()
            self._elapsed_timer.start()
            self._refresh_elapsed()

        def _stop_elapsed(self) -> None:
            if self._elapsed_started_at is not None:
                self._elapsed_base_seconds += max(0, int(time.monotonic() - self._elapsed_started_at))
                self._elapsed_started_at = None
            self._elapsed_timer.stop()
            if hasattr(self, "elapsed_label"):
                self._refresh_elapsed()

        @Slot()
        def _refresh_elapsed(self) -> None:
            display = format_elapsed(self.elapsed_seconds)
            self.elapsed_label.setText(f"已用时 {display}")
            self.elapsed_label.setAccessibleDescription(f"深度精校累计已用时间 {display}")

        def _apply_state(self) -> None:
            state = self._state
            self.state_label.setText(_STATE_TEXT[state])
            self.state_label.setObjectName(_STATE_OBJECT[state])
            self._repolish(self.state_label)
            self.start_button.setEnabled(state in {"idle", "cancelled"})
            self.cancel_button.setEnabled(state == "running")
            self.retry_button.setEnabled(state == "failed")
            self.export_button.setEnabled(state == "completed")
            self.close_button.setEnabled(state not in {"running", "cancelling"})
            self._refresh_progress()
            self._refresh_decision_actions()

        def _refresh_progress(self) -> None:
            self.progress_bar.setValue(self._completed_steps)
            if self._state == "completed":
                heading = "全部 11 步已完成"
            elif self._current_step is not None:
                step_name = DEEP_CORRECTION_STEPS[self._current_step - 1]
                heading = f"第 {self._current_step}/11 步 · {step_name}"
            else:
                heading = "等待开始 · 共 11 步"
            if self._progress_detail:
                heading += f" · {self._progress_detail}"
            self.current_step_label.setText(heading)
            self.progress_bar.setAccessibleDescription(heading)
            for index, (label, step_name) in enumerate(zip(self.step_labels, DEEP_CORRECTION_STEPS), start=1):
                if index <= self._completed_steps:
                    status, object_name = "已完成", "stepDone"
                elif index == self._current_step and self._state == "failed":
                    status, object_name = "失败", "stepFailed"
                elif index == self._current_step and self._state in {"running", "cancelling"}:
                    status = "正在取消" if self._state == "cancelling" else "进行中"
                    object_name = "stepCurrent"
                else:
                    status, object_name = "等待", "stepWaiting"
                label.setText(f"{status} · {index}. {step_name}")
                label.setAccessibleName(f"步骤 {index}，{step_name}，状态 {status}")
                label.setObjectName(object_name)
                self._repolish(label)

        def _refresh_change_list(self, selected_change_id: str | None = None) -> None:
            self.change_list.blockSignals(True)
            self.change_list.clear()
            selected_row = -1
            for row, change_id in enumerate(self._change_order):
                change = self._changes[change_id]
                evidence_text = f"有证据（{len(change.navigable_evidence)}）" if change.navigable_evidence else "无有效证据"
                uncertainty_text = " · 有不确定标记" if change.uncertain else ""
                preview = _one_line(change.corrected_text) or "（精校结果为空）"
                if len(preview) > 58:
                    preview = preview[:57] + "…"
                text = (
                    f"{_DECISION_TEXT[change.status]} · {change.time_range} · {change.speaker or '未识别说话人'}\n"
                    f"{confidence_text(change.confidence)} · {evidence_text}{uncertainty_text}\n{preview}"
                )
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, change.id)
                item.setData(
                    Qt.ItemDataRole.AccessibleTextRole,
                    f"{_DECISION_TEXT[change.status]}，时间 {change.time_range}，说话人 {change.speaker or '未识别'}，"
                    f"{confidence_text(change.confidence)}，{evidence_text}{uncertainty_text}，精校结果 {preview}",
                )
                item.setSizeHint(QSize(0, 78))
                if change.status == "accepted":
                    item.setForeground(QColor("#236454"))
                elif change.status == "rejected":
                    item.setForeground(QColor("#8b3c50"))
                self.change_list.addItem(item)
                if change.id == selected_change_id:
                    selected_row = row
            self.change_list.blockSignals(False)
            if self.change_list.count():
                self.change_list.setCurrentRow(selected_row if selected_row >= 0 else 0)
            else:
                self._show_change(None)
            self._refresh_change_statistics()
            self._refresh_decision_actions()

        def _refresh_change_statistics(self) -> None:
            counts = {status: 0 for status in CHANGE_STATUSES}
            for change in self._changes.values():
                counts[change.status] += 1
            self.change_count_label.setText(
                f"变更：{len(self._changes)} 项 · 待确认 {counts['pending']} · "
                f"已接受 {counts['accepted']} · 已拒绝 {counts['rejected']}"
            )
            eligible = len(self.eligible_bulk_change_ids())
            self.bulk_accept_button.setText(f"批量接受：{eligible} 项")
            self.bulk_accept_button.setAccessibleDescription(
                f"当前有 {eligible} 项满足高置信、有证据、无不确定标记且仍待确认的批量条件"
            )

        @Slot(object, object)
        def _on_change_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
            change_id = str(current.data(Qt.ItemDataRole.UserRole)) if current is not None else ""
            self._show_change(self._changes.get(change_id))
            self._refresh_decision_actions()

        def _show_change(self, change: CorrectionChange | None) -> None:
            if change is None:
                self.time_range_label.setText("时间：—")
                self.speaker_label.setText("说话人：—")
                self.confidence_label.setText("置信度：—")
                self.uncertainty_label.setText("不确定标记：—")
                self.rationale_label.setText("校正依据：请选择一项变更。")
                self.evidence_browser.setPlainText("当前没有变更项；下方显示整份原始与精校文本。")
                self.raw_editor.setPlainText(self._document_raw_text)
                self.corrected_editor.setPlainText(self._document_corrected_text)
                return
            self.time_range_label.setText(f"时间：{change.time_range}")
            self.speaker_label.setText(f"说话人：{change.speaker or '未识别说话人'}")
            self.confidence_label.setText(f"置信度：{confidence_text(change.confidence)}")
            if change.uncertain:
                reason = change.uncertainty_reason or "需要人工听辨或补充证据"
                self.uncertainty_label.setText(f"不确定标记：有 · {reason}")
            else:
                self.uncertainty_label.setText("不确定标记：无")
            rationale = change.rationale or "未提供校正理由，请谨慎确认。"
            self.rationale_label.setText(f"校正依据：{rationale}")
            self.raw_editor.setPlainText(change.raw_text)
            self.corrected_editor.setPlainText(change.corrected_text)
            self._set_evidence(change)

        def _set_evidence(self, change: CorrectionChange) -> None:
            evidence = change.navigable_evidence
            if not evidence:
                self.evidence_browser.setPlainText("证据：无有效链接；此项不能批量接受，请逐项人工判断。")
                return
            links = "".join(
                f'<li><a href="{html.escape(item.url, quote=True)}">{html.escape(item.label)}</a></li>'
                for item in evidence
            )
            self.evidence_browser.setHtml(f"<strong>证据链接（{len(evidence)}）</strong><ul>{links}</ul>")

        def _refresh_decision_actions(self) -> None:
            change = self._changes.get(self.selected_change_id or "")
            can_decide = self._state == "completed" and change is not None
            self.accept_button.setEnabled(bool(can_decide and change and change.status != "accepted"))
            self.reject_button.setEnabled(bool(can_decide and change and change.status != "rejected"))
            self.bulk_accept_button.setEnabled(
                self._state == "completed" and bool(self.eligible_bulk_change_ids())
            )

        @Slot()
        def accept_selected(self) -> str | None:
            if not self.accept_button.isEnabled():
                return None
            change_id = self.selected_change_id
            if change_id is None:
                return None
            self.set_change_status(change_id, "accepted")
            self.acceptRequested.emit(change_id)
            self.feedback_label.setText(f"已接受变更 {change_id}；原始识别仍保持只读。")
            return change_id

        @Slot()
        def reject_selected(self) -> str | None:
            if not self.reject_button.isEnabled():
                return None
            change_id = self.selected_change_id
            if change_id is None:
                return None
            self.set_change_status(change_id, "rejected")
            self.rejectRequested.emit(change_id)
            self.feedback_label.setText(f"已拒绝变更 {change_id}；不会采纳对应精校文字。")
            return change_id

        @Slot()
        def accept_eligible_bulk(self) -> tuple[str, ...]:
            if self._state != "completed":
                return ()
            eligible = self.eligible_bulk_change_ids()
            for change_id in eligible:
                self._changes[change_id] = replace(self._changes[change_id], status="accepted")
            selected = self.selected_change_id
            self._refresh_change_list(selected)
            for change_id in eligible:
                self.acceptRequested.emit(change_id)
            if eligible:
                self.feedback_label.setText(
                    f"已批量接受 {len(eligible)} 项高置信、有证据且无不确定标记的变更。"
                )
            return eligible

        @Slot()
        def toggle_text_expansion(self) -> None:
            self._set_text_expanded(not self._expanded_text)

        def _set_text_expanded(self, expanded: bool) -> None:
            self._expanded_text = bool(expanded)
            maximum = 16_777_215 if expanded else 230
            self.raw_editor.setMaximumHeight(maximum)
            self.corrected_editor.setMaximumHeight(maximum)
            self.toggle_text_button.setText("收起长文本" if expanded else "展开长文本")
            self.toggle_text_button.setAccessibleName(
                "收起原始与精校长文本" if expanded else "展开原始与精校长文本"
            )

        @Slot()
        def _start_clicked(self) -> None:
            if not self.start_button.isEnabled():
                return
            self.mark_running(reset_elapsed=True)
            self.startRequested.emit()

        @Slot()
        def _cancel_clicked(self) -> None:
            if not self.cancel_button.isEnabled():
                return
            self.mark_cancelling()
            self.cancelRequested.emit()

        @Slot()
        def _retry_clicked(self) -> None:
            if not self.retry_button.isEnabled():
                return
            self.mark_running(reset_elapsed=False)
            self.retryRequested.emit()

        @Slot()
        def _export_clicked(self) -> None:
            if not self.export_button.isEnabled():
                return
            self.exportRequested.emit()
            self.feedback_label.setText("已提交导出请求；请选择位置保存完整精校稿。")

        def reject(self) -> None:
            if self._state == "running":
                self._cancel_clicked()
                return
            if self._state == "cancelling":
                return
            super().reject()

        @staticmethod
        def _repolish(widget: QWidget) -> None:
            widget.style().unpolish(widget)
            widget.style().polish(widget)


else:

    class DeepCorrectionDialog:  # pragma: no cover - only instantiated without desktop extras
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            detail = str(QT_WIDGETS_IMPORT_ERROR or "PySide6 不可用")
            raise RuntimeError(f"深度精校工作台需要桌面界面依赖 PySide6。技术信息：{detail}")


def _bounded_nonnegative(value: object) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _one_line(value: object) -> str:
    return " ".join(str(value or "").split())


__all__ = [
    "CHANGE_STATUSES",
    "DEEP_CORRECTION_STEPS",
    "HIGH_CONFIDENCE_THRESHOLD",
    "QT_WIDGETS_AVAILABLE",
    "CorrectionChange",
    "CorrectionEvidence",
    "DeepCorrectionDialog",
    "change_is_bulk_eligible",
    "confidence_text",
    "format_correction_timestamp",
    "format_elapsed",
    "format_time_range",
    "is_navigable_evidence_url",
]
