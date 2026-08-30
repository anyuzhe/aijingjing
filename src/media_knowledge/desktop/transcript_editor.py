from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..transcripts import TranscriptRepository, TranscriptSegment, TranscriptSpeaker, TranscriptV2


try:
    from PySide6.QtCore import Qt, Signal, Slot
    from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QComboBox,
        QDialog,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    QT_WIDGETS_AVAILABLE = True
    QT_WIDGETS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only without desktop extras
    QT_WIDGETS_AVAILABLE = False
    QT_WIDGETS_IMPORT_ERROR = exc


PlayCallback = Callable[[str, int, int], None]


def format_editor_timestamp(milliseconds: object) -> str:
    """Return an accessible, stable timestamp for a transcript segment."""

    try:
        total_seconds = max(0, int(float(milliseconds)) // 1000)
    except (TypeError, ValueError, OverflowError):
        total_seconds = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def transcript_route_text(transcript: TranscriptV2) -> str:
    """Build a concise route label without exposing private configuration data."""

    run = transcript.run
    parts = [run.profile or "未指定方案", run.provider or "未知引擎", run.model or "未知模型"]
    if run.language:
        parts.append(run.language)
    if run.diarization_provider:
        parts.append(f"说话人：{run.diarization_provider}")
    return "  ›  ".join(parts)


def speaker_label(speaker: TranscriptSpeaker | None, speaker_id: str | None = None) -> str:
    if speaker is not None:
        return speaker.display_name or speaker.id
    return speaker_id or "未识别说话人"


@dataclass(slots=True)
class _SegmentDraft:
    segment_id: str
    baseline: str
    text: str

    @property
    def dirty(self) -> bool:
        return self.text != self.baseline


if QT_WIDGETS_AVAILABLE:
    _EDITOR_STYLE = """
    QDialog#transcriptEditorDialog { background: #edf7fc; }
    QFrame#summaryCard, QFrame#editorCard, QGroupBox {
      background: #fbfeff;
      border: 1px solid #c5dfea;
      border-radius: 12px;
    }
    QGroupBox {
      color: #315e78;
      font-weight: 650;
      margin-top: 10px;
      padding-top: 12px;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
    QLabel#editorTitle { color: #1f506d; font-size: 18px; font-weight: 700; }
    QLabel#sourceName { color: #54768a; font-size: 12px; }
    QLabel#routeLabel { color: #2e647f; font-size: 12px; }
    QLabel#qualityPass {
      color: #236852; background: #e1f4ee; border: 1px solid #b5dfd1;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#qualityReview {
      color: #866027; background: #fff4dc; border: 1px solid #edd5a0;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#qualityFail {
      color: #994354; background: #fdebed; border: 1px solid #ebbdc5;
      border-radius: 9px; padding: 4px 9px; font-weight: 650;
    }
    QLabel#feedbackSuccess { color: #246a56; font-size: 12px; }
    QLabel#feedbackError { color: #9b4052; font-size: 12px; }
    QLabel#fieldHeading { color: #345f78; font-size: 13px; font-weight: 650; }
    QListWidget, QPlainTextEdit, QLineEdit, QComboBox {
      background: #fcfeff;
      color: #29485b;
      border: 1px solid #c5dce8;
      border-radius: 9px;
      padding: 5px 7px;
      selection-background-color: #d4edf8;
      selection-color: #174862;
    }
    QListWidget { outline: none; padding: 5px; }
    QListWidget::item { min-height: 38px; padding: 7px 8px; border-radius: 8px; }
    QListWidget::item:hover { background: #edf8fc; }
    QListWidget::item:selected { background: #d7edf7; color: #174c6a; }
    QPlainTextEdit#rawText { background: #f2f7f9; color: #627987; }
    QPushButton {
      min-height: 32px;
      background: #fbfdff;
      color: #29485b;
      border: 1px solid #c4dce8;
      border-radius: 8px;
      padding: 5px 11px;
    }
    QPushButton:hover { background: #e5f4fa; border-color: #81b8d1; }
    QPushButton:pressed { background: #d5eaf4; }
    QPushButton:focus, QComboBox:focus, QLineEdit:focus,
    QPlainTextEdit:focus, QListWidget:focus {
      border: 2px solid #398eb7;
    }
    QPushButton:disabled { color: #91a5b0; background: #eef4f7; border-color: #d7e4ea; }
    QPushButton#primaryButton {
      color: white; font-weight: 650; background: #26769c; border-color: #6db0cc;
    }
    QPushButton#primaryButton:hover { background: #3188af; border-color: #93cadf; }
    QListWidget#warningList {
      color: #785a2b; background: #fffaf0; border-color: #ead6aa; max-height: 78px;
    }
    """


    class TranscriptEditorDialog(QDialog):
        """Edit Transcript V2 facts without changing immutable ASR raw text."""

        playRequested = Signal(str, int, int)
        transcriptSaved = Signal(str, object)
        speakerChanged = Signal(str, object)
        reviewApproved = Signal(str)
        errorOccurred = Signal(str)

        def __init__(
            self,
            repository: TranscriptRepository,
            run_id: str,
            *,
            media_path: str | Path | None = None,
            actor: str = "user",
            play_callback: PlayCallback | None = None,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.repository = repository
            self.run_id = str(run_id or "").strip()
            if not self.run_id:
                raise ValueError("转写任务 ID 不能为空")
            transcript = repository.get_transcript(self.run_id)
            if transcript is None:
                raise KeyError(f"转写任务不存在：{self.run_id}")

            source_media = media_path if media_path is not None else transcript.source.original_uri
            self.media_path = str(source_media or "").strip()
            self.actor = str(actor or "user").strip() or "user"
            self.play_callback = play_callback
            self._transcript = transcript
            self._segments: dict[str, TranscriptSegment] = {}
            self._speakers: dict[str, TranscriptSpeaker] = {}
            self._drafts: dict[str, _SegmentDraft] = {}
            self._current_segment_id: str | None = None
            self._loading_editor = False
            self._last_error = ""
            self._shortcuts: list[QShortcut] = []

            self.setObjectName("transcriptEditorDialog")
            self.setWindowTitle(f"转写校订 · {transcript.source.name}")
            self.setMinimumSize(880, 640)
            self.resize(1120, 780)
            self.setModal(False)
            self.setStyleSheet(_EDITOR_STYLE)
            self._build_ui()
            self._connect_signals()
            self._install_shortcuts()
            self._reload(preserve_drafts=False)

        @property
        def transcript(self) -> TranscriptV2:
            return self._transcript

        @property
        def current_segment_id(self) -> str | None:
            return self._current_segment_id

        @property
        def dirty_segment_ids(self) -> tuple[str, ...]:
            return tuple(
                segment.id
                for segment in self._transcript.segments
                if (draft := self._drafts.get(segment.id)) is not None and draft.dirty
            )

        @property
        def last_error(self) -> str:
            return self._last_error

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(12)

            summary = QFrame()
            summary.setObjectName("summaryCard")
            summary_layout = QVBoxLayout(summary)
            summary_layout.setContentsMargins(14, 12, 14, 12)
            summary_layout.setSpacing(7)
            heading_row = QHBoxLayout()
            self.title_label = QLabel("转写校订")
            self.title_label.setObjectName("editorTitle")
            self.title_label.setAccessibleName("转写校订标题")
            heading_row.addWidget(self.title_label, 1)
            self.quality_label = QLabel()
            self.quality_label.setAccessibleName("转写质量状态")
            heading_row.addWidget(self.quality_label)
            summary_layout.addLayout(heading_row)
            self.source_label = QLabel()
            self.source_label.setObjectName("sourceName")
            self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.source_label.setAccessibleName("转写来源")
            summary_layout.addWidget(self.source_label)
            self.route_label = QLabel()
            self.route_label.setObjectName("routeLabel")
            self.route_label.setWordWrap(True)
            self.route_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.route_label.setAccessibleName("转写运行路线")
            summary_layout.addWidget(self.route_label)
            self.warning_list = QListWidget()
            self.warning_list.setObjectName("warningList")
            self.warning_list.setAccessibleName("转写质量警告")
            self.warning_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            summary_layout.addWidget(self.warning_list)
            root.addWidget(summary)

            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.setChildrenCollapsible(False)

            segment_frame = QFrame()
            segment_frame.setObjectName("editorCard")
            segment_layout = QVBoxLayout(segment_frame)
            segment_layout.setContentsMargins(11, 11, 11, 11)
            segment_heading_row = QHBoxLayout()
            self.segment_heading = QLabel("时间轴片段")
            self.segment_heading.setObjectName("fieldHeading")
            segment_heading_row.addWidget(self.segment_heading, 1)
            self.play_button = QPushButton("从这里播放")
            self.play_button.setAccessibleName("从当前片段时间点播放")
            self.play_button.setToolTip("在播放器中打开原始音视频并跳转到当前片段（Ctrl+P）")
            segment_heading_row.addWidget(self.play_button)
            segment_layout.addLayout(segment_heading_row)
            self.segment_list = QListWidget()
            self.segment_list.setAccessibleName("带时间和说话人的转写片段")
            self.segment_list.setAccessibleDescription("选择片段后可查看原始文字并进行定向校订")
            self.segment_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            segment_layout.addWidget(self.segment_list, 1)
            splitter.addWidget(segment_frame)

            editor_frame = QFrame()
            editor_frame.setObjectName("editorCard")
            editor_layout = QVBoxLayout(editor_frame)
            editor_layout.setContentsMargins(12, 11, 12, 11)
            editor_layout.setSpacing(8)

            assignment_row = QHBoxLayout()
            assignment_label = QLabel("此片段说话人")
            assignment_label.setObjectName("fieldHeading")
            assignment_row.addWidget(assignment_label)
            self.segment_speaker_combo = QComboBox()
            self.segment_speaker_combo.setAccessibleName("当前片段说话人")
            self.segment_speaker_combo.setMinimumWidth(180)
            assignment_row.addWidget(self.segment_speaker_combo, 1)
            self.assign_button = QPushButton("重新分配")
            self.assign_button.setAccessibleName("保存当前片段说话人")
            assignment_row.addWidget(self.assign_button)
            editor_layout.addLayout(assignment_row)

            raw_heading = QLabel("原始识别文字 · 只读证据")
            raw_heading.setObjectName("fieldHeading")
            editor_layout.addWidget(raw_heading)
            self.raw_editor = QPlainTextEdit()
            self.raw_editor.setObjectName("rawText")
            self.raw_editor.setReadOnly(True)
            self.raw_editor.setMaximumHeight(135)
            self.raw_editor.setAccessibleName("原始识别文字，只读")
            self.raw_editor.setAccessibleDescription("原始识别结果不会被人工校订覆盖")
            editor_layout.addWidget(self.raw_editor)

            corrected_heading = QLabel("人工校订文字")
            corrected_heading.setObjectName("fieldHeading")
            editor_layout.addWidget(corrected_heading)
            self.corrected_editor = QPlainTextEdit()
            self.corrected_editor.setAccessibleName("当前片段校订文字")
            self.corrected_editor.setAccessibleDescription("只修改当前片段；保存后会生成审计记录")
            self.corrected_editor.setPlaceholderText("在这里校订当前片段，不会修改原始识别文字……")
            editor_layout.addWidget(self.corrected_editor, 1)

            reason_row = QHBoxLayout()
            reason_label = QLabel("修改原因")
            reason_row.addWidget(reason_label)
            self.reason_edit = QLineEdit("人工核对并校订转写片段")
            self.reason_edit.setAccessibleName("校订原因")
            self.reason_edit.setPlaceholderText("写明为什么修改，便于以后追溯")
            reason_row.addWidget(self.reason_edit, 1)
            self.save_button = QPushButton("保存校订")
            self.save_button.setObjectName("primaryButton")
            self.save_button.setAccessibleName("保存全部待处理校订")
            self.save_button.setToolTip("逐片段保存并记录审计信息（Ctrl+S）")
            reason_row.addWidget(self.save_button)
            editor_layout.addLayout(reason_row)
            splitter.addWidget(editor_frame)
            splitter.setSizes([420, 680])
            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 3)
            root.addWidget(splitter, 1)

            speaker_group = QGroupBox("说话人管理")
            speaker_layout = QHBoxLayout(speaker_group)
            speaker_layout.setContentsMargins(11, 14, 11, 10)
            self.rename_speaker_combo = QComboBox()
            self.rename_speaker_combo.setAccessibleName("待重命名说话人")
            speaker_layout.addWidget(self.rename_speaker_combo, 1)
            self.speaker_name_edit = QLineEdit()
            self.speaker_name_edit.setAccessibleName("说话人新名称")
            self.speaker_name_edit.setPlaceholderText("例如：张老师")
            speaker_layout.addWidget(self.speaker_name_edit, 1)
            self.rename_button = QPushButton("确认名称")
            self.rename_button.setAccessibleName("重命名所选说话人")
            speaker_layout.addWidget(self.rename_button)
            speaker_layout.addSpacing(14)
            self.merge_source_combo = QComboBox()
            self.merge_source_combo.setAccessibleName("待合并说话人")
            speaker_layout.addWidget(self.merge_source_combo, 1)
            merge_arrow = QLabel("合并到")
            merge_arrow.setAccessibleName("合并方向")
            speaker_layout.addWidget(merge_arrow)
            self.merge_target_combo = QComboBox()
            self.merge_target_combo.setAccessibleName("目标说话人")
            speaker_layout.addWidget(self.merge_target_combo, 1)
            self.merge_button = QPushButton("合并")
            self.merge_button.setAccessibleName("合并所选说话人")
            speaker_layout.addWidget(self.merge_button)
            root.addWidget(speaker_group)

            feedback_row = QHBoxLayout()
            self.feedback_label = QLabel("选择一个片段开始核对。")
            self.feedback_label.setObjectName("feedbackSuccess")
            self.feedback_label.setWordWrap(True)
            self.feedback_label.setAccessibleName("保存和错误反馈")
            self.feedback_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            feedback_row.addWidget(self.feedback_label, 1)
            self.approve_button = QPushButton("确认复核完成并用于问答")
            self.approve_button.setObjectName("primaryButton")
            self.approve_button.setAccessibleName("确认转写复核完成并加入问答索引")
            self.approve_button.setToolTip("人工确认后，转写资料才会通过质量门禁并参与问答")
            feedback_row.addWidget(self.approve_button)
            close_button = QPushButton("完成")
            close_button.setAccessibleName("关闭转写校订")
            close_button.clicked.connect(self.close)
            feedback_row.addWidget(close_button)
            root.addLayout(feedback_row)

        def _connect_signals(self) -> None:
            self.segment_list.currentItemChanged.connect(self._on_segment_changed)
            self.corrected_editor.textChanged.connect(self._on_corrected_changed)
            self.save_button.clicked.connect(self._save_clicked)
            self.play_button.clicked.connect(self.request_play_current)
            self.assign_button.clicked.connect(self._assign_clicked)
            self.rename_button.clicked.connect(self._rename_clicked)
            self.rename_speaker_combo.currentIndexChanged.connect(self._load_speaker_name)
            self.merge_button.clicked.connect(self._merge_clicked)
            self.approve_button.clicked.connect(self._approve_review_clicked)

        def _install_shortcuts(self) -> None:
            for sequence, callback in (
                ("Ctrl+S", self._save_clicked),
                ("Ctrl+P", self.request_play_current),
            ):
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)

        def _reload(
            self,
            *,
            preserve_drafts: bool = True,
            selected_segment_id: str | None = None,
        ) -> None:
            transcript = self.repository.get_transcript(self.run_id)
            if transcript is None:
                raise KeyError(f"转写任务不存在：{self.run_id}")
            previous_drafts = self._drafts if preserve_drafts else {}
            selected = selected_segment_id or self._current_segment_id
            self._transcript = transcript
            self._segments = {segment.id: segment for segment in transcript.segments}
            self._speakers = {speaker.id: speaker for speaker in transcript.speakers}
            self._drafts = {}
            for segment in transcript.segments:
                previous = previous_drafts.get(segment.id)
                baseline = segment.effective_text
                self._drafts[segment.id] = _SegmentDraft(
                    segment.id,
                    baseline,
                    previous.text if previous is not None and previous.dirty else baseline,
                )

            self.source_label.setText(
                f"{transcript.source.name}  ·  {len(transcript.segments)} 个片段  ·  "
                f"{format_editor_timestamp(transcript.source.duration_ms)}"
            )
            self.route_label.setText(f"运行路线：{transcript_route_text(transcript)}")
            self._refresh_quality()
            self._populate_speakers()
            self._populate_segments(selected)
            self._update_save_state()

        def _refresh_quality(self) -> None:
            quality = self._transcript.quality
            status = quality.status if quality.status in {"pass", "review", "fail"} else "review"
            labels = {"pass": "质量通过", "review": "需要复核", "fail": "质量未通过"}
            objects = {"pass": "qualityPass", "review": "qualityReview", "fail": "qualityFail"}
            self.quality_label.setObjectName(objects[status])
            self.quality_label.setText(labels[status])
            self.quality_label.style().unpolish(self.quality_label)
            self.quality_label.style().polish(self.quality_label)
            self.warning_list.clear()
            for warning in quality.warnings:
                item = QListWidgetItem(f"⚠  {warning}")
                item.setData(Qt.ItemDataRole.AccessibleTextRole, f"质量警告：{warning}")
                self.warning_list.addItem(item)
            self.warning_list.setVisible(bool(quality.warnings))
            self.approve_button.setVisible(status != "pass")

        @Slot()
        def _approve_review_clicked(self) -> None:
            if self.dirty_segment_ids:
                self._show_error("请先保存当前校订，再确认复核完成。")
                return
            answer = QMessageBox.question(
                self,
                "确认转写复核完成",
                "确认已核对关键片段、专业术语和说话人？\n\n"
                "确认后，这份资料将重新加入高可信问答索引。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.reviewApproved.emit(self.run_id)
            self._set_feedback("已提交人工复核确认，正在更新问答索引。")
            self.approve_button.setEnabled(False)
            self.close()

        def _active_speakers(self) -> list[TranscriptSpeaker]:
            return [
                speaker
                for speaker in self._transcript.speakers
                if not speaker.metadata.get("merged_into")
            ]

        def _populate_speakers(self) -> None:
            speakers = self._active_speakers()
            combos = (
                self.segment_speaker_combo,
                self.rename_speaker_combo,
                self.merge_source_combo,
                self.merge_target_combo,
            )
            for combo in combos:
                previous = combo.currentData()
                combo.blockSignals(True)
                combo.clear()
                for speaker in speakers:
                    label = speaker_label(speaker)
                    combo.addItem(f"{label}  ·  {speaker.id}", speaker.id)
                index = combo.findData(previous)
                combo.setCurrentIndex(index if index >= 0 else (0 if combo.count() else -1))
                combo.blockSignals(False)
            if self.merge_target_combo.count() > 1 and self.merge_target_combo.currentIndex() == self.merge_source_combo.currentIndex():
                self.merge_target_combo.setCurrentIndex(1)
            self._load_speaker_name()

        def _segment_item_text(self, segment: TranscriptSegment) -> str:
            speaker = speaker_label(self._speakers.get(segment.speaker_id or ""), segment.speaker_id)
            draft = self._drafts[segment.id]
            marker = "● " if draft.dirty else ""
            flags = "  ⚠" if segment.flags else ""
            preview = " ".join(draft.text.split()) or "（此片段暂无文字）"
            if len(preview) > 74:
                preview = preview[:73] + "…"
            return (
                f"{marker}[{format_editor_timestamp(segment.start_ms)}–"
                f"{format_editor_timestamp(segment.end_ms)}]  {speaker}{flags}\n{preview}"
            )

        def _populate_segments(self, selected_segment_id: str | None) -> None:
            self.segment_list.blockSignals(True)
            self.segment_list.clear()
            selected_row = 0
            for row, segment in enumerate(self._transcript.segments):
                item = QListWidgetItem(self._segment_item_text(segment))
                item.setData(Qt.ItemDataRole.UserRole, segment.id)
                item.setData(Qt.ItemDataRole.AccessibleTextRole, self._segment_item_text(segment))
                item.setToolTip(self._segment_item_text(segment))
                self.segment_list.addItem(item)
                if segment.id == selected_segment_id:
                    selected_row = row
            self.segment_heading.setText(f"时间轴片段（{len(self._transcript.segments)}）")
            self.segment_list.blockSignals(False)
            if self.segment_list.count():
                self.segment_list.setCurrentRow(selected_row)
                self._show_segment(self.segment_list.currentItem())
            else:
                self._show_segment(None)

        def _refresh_segment_item(self, segment_id: str) -> None:
            for row in range(self.segment_list.count()):
                item = self.segment_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == segment_id:
                    segment = self._segments[segment_id]
                    text = self._segment_item_text(segment)
                    item.setText(text)
                    item.setToolTip(text)
                    item.setData(Qt.ItemDataRole.AccessibleTextRole, text)
                    break

        @Slot(QListWidgetItem, QListWidgetItem)
        def _on_segment_changed(
            self,
            current: QListWidgetItem | None,
            _previous: QListWidgetItem | None,
        ) -> None:
            self._show_segment(current)

        def _show_segment(self, item: QListWidgetItem | None) -> None:
            segment_id = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            segment = self._segments.get(str(segment_id)) if segment_id else None
            self._current_segment_id = segment.id if segment is not None else None
            self._loading_editor = True
            try:
                self.raw_editor.setPlainText(segment.raw_text if segment is not None else "")
                draft = self._drafts.get(segment.id) if segment is not None else None
                self.corrected_editor.setPlainText(draft.text if draft is not None else "")
                self.corrected_editor.setEnabled(segment is not None)
                self.assign_button.setEnabled(segment is not None and bool(self._active_speakers()))
                if segment is not None:
                    index = self.segment_speaker_combo.findData(segment.speaker_id)
                    self.segment_speaker_combo.setCurrentIndex(index)
                else:
                    self.segment_speaker_combo.setCurrentIndex(-1)
            finally:
                self._loading_editor = False
            self.play_button.setEnabled(segment is not None and bool(self.media_path))

        @Slot()
        def _on_corrected_changed(self) -> None:
            if self._loading_editor or self._current_segment_id is None:
                return
            draft = self._drafts.get(self._current_segment_id)
            if draft is None:
                return
            draft.text = self.corrected_editor.toPlainText()
            self._refresh_segment_item(draft.segment_id)
            self._update_save_state()

        def _update_save_state(self) -> None:
            count = len(self.dirty_segment_ids)
            self.save_button.setEnabled(count > 0)
            self.save_button.setText(f"保存校订（{count}）" if count else "保存校订")

        def save_changes(self, *, reason: str | None = None, actor: str | None = None) -> tuple[str, ...]:
            """Persist only dirty segments and create one audit row for each fact edit."""

            dirty_ids = self.dirty_segment_ids
            if not dirty_ids:
                self._set_feedback("没有需要保存的校订。")
                return ()
            audit_reason = str(reason if reason is not None else self.reason_edit.text()).strip()
            if not audit_reason:
                raise ValueError("请填写修改原因，便于以后追溯")
            audit_actor = str(actor or self.actor).strip() or "user"
            saved: list[str] = []
            for segment_id in dirty_ids:
                segment = self._segments[segment_id]
                draft = self._drafts[segment_id]
                corrected: str | None = draft.text
                if corrected == segment.raw_text:
                    corrected = None
                updated = self.repository.update_corrected_text(
                    segment_id,
                    corrected,
                    edit_type="manual_correction",
                    reason=audit_reason,
                    actor=audit_actor,
                    metadata={"editor": "desktop-transcript-editor"},
                )
                self._segments[segment_id] = updated
                draft.baseline = updated.effective_text
                draft.text = updated.effective_text
                saved.append(segment_id)
            selected = self._current_segment_id
            self._reload(preserve_drafts=True, selected_segment_id=selected)
            result = tuple(saved)
            self._set_feedback(f"已保存 {len(result)} 个片段；原始识别文字保持不变。")
            self.transcriptSaved.emit(self.run_id, result)
            return result

        @Slot()
        def _save_clicked(self) -> None:
            try:
                self.save_changes()
            except Exception as exc:  # noqa: BLE001 - desktop boundary turns failures into feedback
                self._show_error(f"保存失败：{exc}")

        def rename_speaker(
            self,
            speaker_id: str,
            display_name: str,
            *,
            reason: str = "人工确认说话人名称",
        ) -> TranscriptSpeaker:
            renamed = self.repository.rename_speaker(
                self.run_id,
                speaker_id,
                display_name,
                reason=reason,
                actor=self.actor,
            )
            self._reload(preserve_drafts=True, selected_segment_id=self._current_segment_id)
            self._set_feedback(f"已将 {speaker_id} 命名为“{renamed.display_name}”。")
            self.speakerChanged.emit(self.run_id, (speaker_id,))
            return renamed

        @Slot()
        def _rename_clicked(self) -> None:
            speaker_id = str(self.rename_speaker_combo.currentData() or "")
            name = self.speaker_name_edit.text().strip()
            try:
                if not speaker_id:
                    raise ValueError("请先选择说话人")
                self.rename_speaker(speaker_id, name)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"重命名失败：{exc}")

        @Slot()
        def _load_speaker_name(self) -> None:
            speaker_id = str(self.rename_speaker_combo.currentData() or "")
            speaker = self._speakers.get(speaker_id)
            self.speaker_name_edit.setText(speaker.display_name or "" if speaker else "")

        def reassign_segment(
            self,
            segment_id: str,
            speaker_id: str,
            *,
            reason: str = "人工调整说话人归属",
        ) -> TranscriptSegment:
            updated = self.repository.reassign_segment(
                segment_id,
                speaker_id,
                reason=reason,
                actor=self.actor,
            )
            self._reload(preserve_drafts=True, selected_segment_id=segment_id)
            self._set_feedback(f"已重新分配片段 {segment_id} 的说话人。")
            self.speakerChanged.emit(self.run_id, (segment_id,))
            return updated

        @Slot()
        def _assign_clicked(self) -> None:
            segment_id = self._current_segment_id
            speaker_id = str(self.segment_speaker_combo.currentData() or "")
            try:
                if not segment_id or not speaker_id:
                    raise ValueError("请先选择片段和说话人")
                self.reassign_segment(segment_id, speaker_id)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"重新分配失败：{exc}")

        def merge_speakers(
            self,
            source_speaker_id: str,
            target_speaker_id: str,
            *,
            reason: str = "人工合并说话人",
        ) -> int:
            affected = self.repository.merge_speakers(
                self.run_id,
                source_speaker_id,
                target_speaker_id,
                reason=reason,
                actor=self.actor,
            )
            self._reload(preserve_drafts=True, selected_segment_id=self._current_segment_id)
            self._set_feedback(f"说话人已合并，更新了 {affected} 个片段。")
            self.speakerChanged.emit(self.run_id, (source_speaker_id, target_speaker_id))
            return affected

        @Slot()
        def _merge_clicked(self) -> None:
            source = str(self.merge_source_combo.currentData() or "")
            target = str(self.merge_target_combo.currentData() or "")
            try:
                if not source or not target:
                    raise ValueError("至少需要两个可用的说话人")
                if source == target:
                    raise ValueError("不能把说话人合并到自身")
                answer = QMessageBox.question(
                    self,
                    "确认合并说话人",
                    f"将 {speaker_label(self._speakers.get(source), source)} 的全部片段合并到 "
                    f"{speaker_label(self._speakers.get(target), target)}？\n此操作会写入审计记录。",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                self.merge_speakers(source, target)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"合并失败：{exc}")

        @Slot()
        def request_play_current(self) -> bool:
            if self._current_segment_id is None:
                self._show_error("请先选择一个转写片段。")
                return False
            if not self.media_path:
                self._show_error("这个转写任务没有可用的原始音视频路径。")
                return False
            segment = self._segments[self._current_segment_id]
            try:
                if self.play_callback is not None:
                    self.play_callback(self.media_path, segment.start_ms, segment.end_ms)
                self.playRequested.emit(self.media_path, segment.start_ms, segment.end_ms)
            except Exception as exc:  # noqa: BLE001
                self._show_error(f"打开播放器失败：{exc}")
                return False
            self._set_feedback(f"正在从 {format_editor_timestamp(segment.start_ms)} 打开原始音视频。")
            return True

        def _set_feedback(self, message: str) -> None:
            self._last_error = ""
            self.feedback_label.setObjectName("feedbackSuccess")
            self.feedback_label.setText(message)
            self.feedback_label.style().unpolish(self.feedback_label)
            self.feedback_label.style().polish(self.feedback_label)

        def _show_error(self, message: str) -> None:
            self._last_error = message
            self.feedback_label.setObjectName("feedbackError")
            self.feedback_label.setText(message)
            self.feedback_label.style().unpolish(self.feedback_label)
            self.feedback_label.style().polish(self.feedback_label)
            self.errorOccurred.emit(message)

        def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
            if not self.dirty_segment_ids:
                event.accept()
                return
            answer = QMessageBox.warning(
                self,
                "存在未保存的校订",
                "还有片段没有保存。要在关闭前保存吗？",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                try:
                    self.save_changes()
                except Exception as exc:  # noqa: BLE001
                    self._show_error(f"保存失败：{exc}")
                    event.ignore()
                    return
            event.accept()


else:

    class TranscriptEditorDialog:  # pragma: no cover - depends on optional desktop runtime
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            detail = str(QT_WIDGETS_IMPORT_ERROR or "PySide6 不可用")
            raise RuntimeError(f"转写编辑器需要桌面界面依赖 PySide6。技术信息：{detail}")


__all__ = [
    "QT_WIDGETS_AVAILABLE",
    "TranscriptEditorDialog",
    "format_editor_timestamp",
    "speaker_label",
    "transcript_route_text",
]
