from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


try:  # Qt Multimedia lives in PySide6-Addons, not PySide6-Essentials.
    from PySide6.QtCore import Qt, QUrl, Signal, Slot
    from PySide6.QtGui import QCloseEvent, QFontDatabase, QKeySequence, QShortcut
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QSizePolicy,
        QSlider,
        QSplitter,
        QStyle,
        QVBoxLayout,
        QWidget,
    )

    QT_MULTIMEDIA_AVAILABLE = True
    QT_MULTIMEDIA_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on the optional desktop runtime
    QT_MULTIMEDIA_AVAILABLE = False
    QT_MULTIMEDIA_IMPORT_ERROR = exc


_VIDEO_SUFFIXES = frozenset(
    {
        ".3gp",
        ".avi",
        ".flv",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ts",
        ".webm",
        ".wmv",
    }
)


def _finite_number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _read_value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_value(value: object, names: Iterable[str], default: object = None) -> object:
    for name in names:
        candidate = _read_value(value, name, None)
        if candidate is not None:
            return candidate
    return default


def format_timestamp(milliseconds: object) -> str:
    """Format milliseconds as a stable player time without locale surprises."""

    total_seconds = max(0, int(_finite_number(milliseconds)) // 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def clamp_position_ms(position_ms: object, duration_ms: object = 0) -> int:
    """Clamp a requested seek position to a media duration when it is known."""

    position = max(0, int(_finite_number(position_ms)))
    duration = max(0, int(_finite_number(duration_ms)))
    return min(position, duration) if duration else position


def is_video_source(source: str | Path | object) -> bool:
    """Return whether the source name has a commonly supported video suffix."""

    text = str(source or "").strip()
    if not text:
        return False
    parsed = urlparse(text)
    candidate = parsed.path if parsed.scheme else text
    return Path(candidate).suffix.casefold() in _VIDEO_SUFFIXES


@dataclass(frozen=True, slots=True)
class TranscriptCue:
    """Small UI-facing projection of a transcript segment.

    The player deliberately accepts both Transcript V2 objects (``start_ms``) and
    legacy ASR dictionaries (``start`` in seconds), so opening old sources never
    requires rewriting their archived transcript.
    """

    id: str
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    ordinal: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def display_text(self) -> str:
        prefix = f"[{format_timestamp(self.start_ms)}]"
        speaker = f" {self.speaker}" if self.speaker else ""
        text = self.text.strip() or "（此片段暂无文字）"
        return f"{prefix}{speaker}  {text}"

    @classmethod
    def from_value(cls, value: object, *, fallback_ordinal: int = 0) -> "TranscriptCue":
        uses_milliseconds = any(
            _read_value(value, name, None) is not None
            for name in ("start_ms", "timestamp_start_ms")
        )
        start_value = _first_value(
            value,
            ("start_ms", "timestamp_start_ms", "start", "timestamp_start"),
            0,
        )
        end_value = _first_value(
            value,
            ("end_ms", "timestamp_end_ms", "end", "timestamp_end"),
            start_value,
        )
        multiplier = 1.0 if uses_milliseconds else 1000.0
        start_ms = max(0, int(round(_finite_number(start_value) * multiplier)))
        end_ms = max(start_ms, int(round(_finite_number(end_value) * multiplier)))

        corrected = _read_value(value, "corrected_text", None)
        if corrected is not None:
            text = str(corrected)
        else:
            text = str(
                _first_value(value, ("effective_text", "raw_text", "text", "description"), "")
                or ""
            )
        ordinal = int(
            _finite_number(
                _first_value(value, ("ordinal", "sequence", "index"), fallback_ordinal),
                float(fallback_ordinal),
            )
        )
        identifier = str(
            _first_value(value, ("id", "segment_id"), f"segment-{ordinal + 1}") or ""
        ).strip() or f"segment-{ordinal + 1}"
        speaker = str(
            _first_value(
                value,
                ("speaker_display_name", "speaker_name", "speaker", "speaker_id"),
                "",
            )
            or ""
        ).strip() or None
        return cls(identifier, start_ms, end_ms, text.strip(), speaker, ordinal)


def normalize_transcript_cues(values: Iterable[object] | None) -> tuple[TranscriptCue, ...]:
    """Coerce and chronologically order transcript values for player display."""

    cues = [
        value if isinstance(value, TranscriptCue) else TranscriptCue.from_value(value, fallback_ordinal=index)
        for index, value in enumerate(values or ())
    ]
    cues.sort(key=lambda cue: (cue.start_ms, cue.end_ms, cue.ordinal, cue.id))
    return tuple(cues)


def active_cue_index(cues: Iterable[TranscriptCue], position_ms: object) -> int | None:
    """Find the last cue containing a player position, including overlapping cues."""

    ordered = tuple(cues)
    if not ordered:
        return None
    position = max(0, int(_finite_number(position_ms)))
    starts = [cue.start_ms for cue in ordered]
    candidate = bisect_right(starts, position) - 1
    while candidate >= 0 and ordered[candidate].start_ms <= position:
        cue = ordered[candidate]
        if cue.start_ms <= position < max(cue.start_ms + 1, cue.end_ms):
            return candidate
        candidate -= 1
    return None


def multimedia_unavailable_message() -> str:
    detail = str(QT_MULTIMEDIA_IMPORT_ERROR or "Qt Multimedia 不可用")
    return (
        "媒体播放器组件未安装。请安装桌面多媒体依赖 PySide6-Addons 后重试。"
        f"\n技术信息：{detail}"
    )


if QT_MULTIMEDIA_AVAILABLE:
    _PLAYER_STYLE = """
    QDialog#mediaPlayerDialog { background: #eef7fb; }
    QFrame#mediaSurface, QFrame#transcriptPanel {
      background: #fbfeff;
      border: 1px solid #c7dfea;
      border-radius: 12px;
    }
    QLabel#playerTitle { color: #214f6b; font-size: 16px; font-weight: 650; }
    QLabel#playerStatus { color: #59798d; font-size: 12px; }
    QLabel#playerError { color: #9b4151; font-size: 12px; }
    QLabel#playerTime { color: #294f66; font-size: 13px; }
    QLabel#transcriptHeading { color: #345f78; font-size: 13px; font-weight: 600; }
    QPushButton, QComboBox {
      min-height: 32px;
      background: #fbfdff;
      color: #294457;
      border: 1px solid #c5dce8;
      border-radius: 8px;
      padding: 5px 10px;
    }
    QPushButton:hover, QComboBox:hover { background: #e6f4fa; border-color: #82b9d2; }
    QPushButton:pressed { background: #d8ebf5; }
    QPushButton:focus, QComboBox:focus, QCheckBox:focus, QListWidget:focus, QSlider:focus {
      border: 2px solid #3c91b8;
    }
    QPushButton:disabled { color: #91a4af; background: #edf3f6; border-color: #d8e4ea; }
    QPushButton#playButton {
      min-width: 84px;
      color: white;
      font-weight: 600;
      background: #236c91;
      border-color: #69abc8;
    }
    QPushButton#playButton:hover { background: #2d82a9; border-color: #92c9df; }
    QListWidget {
      background: #fcfeff;
      color: #294457;
      border: 1px solid #c8dce7;
      border-radius: 9px;
      padding: 4px;
      outline: none;
    }
    QListWidget::item { min-height: 30px; padding: 6px 8px; border-radius: 7px; }
    QListWidget::item:hover { background: #edf7fb; }
    QListWidget::item:selected { background: #d6ecf6; color: #174a6b; }
    QSlider::groove:horizontal { height: 6px; border-radius: 3px; background: #d8e8f0; }
    QSlider::sub-page:horizontal { border-radius: 3px; background: #65abc8; }
    QSlider::handle:horizontal {
      width: 16px; margin: -6px 0; border-radius: 8px;
      background: #fafdff; border: 2px solid #347fa4;
    }
    QCheckBox { color: #365d74; spacing: 7px; }
    QVideoWidget { background: #102839; border-radius: 9px; }
    """


    class MediaPlayerDialog(QDialog):
        """Local audio/video player with timestamp-aware transcript navigation."""

        segmentActivated = Signal(str, int, int)
        mediaReady = Signal(str)
        mediaError = Signal(str)

        def __init__(
            self,
            source: str | Path | QUrl | None = None,
            *,
            title: str = "音视频证据",
            segments: Iterable[object] | None = None,
            start_ms: int = 0,
            autoplay: bool = False,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("mediaPlayerDialog")
            self.setWindowTitle(title or "音视频证据")
            self.setMinimumSize(720, 560)
            self.resize(920, 720)
            self.setModal(False)
            self.setStyleSheet(_PLAYER_STYLE)

            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.audio_output.setVolume(1.0)

            self._source: QUrl | None = None
            self._source_display = ""
            self._pending_start_ms = max(0, int(start_ms))
            self._pending_autoplay = bool(autoplay)
            self._slider_dragging = False
            self._cues: tuple[TranscriptCue, ...] = ()
            self._cue_starts: tuple[int, ...] = ()
            self._cue_prefix_max_end: tuple[int, ...] = ()
            self._active_cue_index: int | None = None
            self._loop_bounds: tuple[int, int] | None = None
            self._last_error = ""
            self._shortcuts: list[QShortcut] = []

            self._build_ui(title or "音视频证据")
            self._connect_signals()
            self._install_shortcuts()
            self.set_segments(segments or ())
            self._set_controls_enabled(False)
            if source is not None:
                self.load_media(source, start_ms=start_ms, autoplay=autoplay)

        @property
        def cues(self) -> tuple[TranscriptCue, ...]:
            return self._cues

        @property
        def source_url(self) -> QUrl | None:
            return self._source

        @property
        def last_error(self) -> str:
            return self._last_error

        def _build_ui(self, title: str) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(12)

            heading_row = QHBoxLayout()
            heading_row.setSpacing(8)
            self.title_label = QLabel(title)
            self.title_label.setObjectName("playerTitle")
            self.title_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.title_label.setAccessibleName("当前媒体标题")
            heading_row.addWidget(self.title_label, 1)
            self.status_label = QLabel("尚未加载媒体")
            self.status_label.setObjectName("playerStatus")
            self.status_label.setAccessibleName("播放器状态")
            self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            heading_row.addWidget(self.status_label, 0, Qt.AlignmentFlag.AlignRight)
            root.addLayout(heading_row)

            self.splitter = QSplitter(Qt.Orientation.Vertical)
            self.splitter.setChildrenCollapsible(False)

            media_frame = QFrame()
            media_frame.setObjectName("mediaSurface")
            media_layout = QVBoxLayout(media_frame)
            media_layout.setContentsMargins(10, 10, 10, 10)
            media_layout.setSpacing(10)

            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(260)
            self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.video_widget.setAccessibleName("视频画面")
            self.video_widget.setVisible(False)
            self.player.setVideoOutput(self.video_widget)
            media_layout.addWidget(self.video_widget, 1)

            timeline_row = QHBoxLayout()
            timeline_row.setSpacing(10)
            self.position_slider = QSlider(Qt.Orientation.Horizontal)
            self.position_slider.setRange(0, 0)
            self.position_slider.setSingleStep(1000)
            self.position_slider.setPageStep(5000)
            self.position_slider.setAccessibleName("播放进度")
            self.position_slider.setAccessibleDescription("使用左右方向键调整播放位置")
            timeline_row.addWidget(self.position_slider, 1)
            self.time_label = QLabel("00:00 / 00:00")
            self.time_label.setObjectName("playerTime")
            self.time_label.setMinimumWidth(116)
            self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.time_label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            self.time_label.setAccessibleName("播放时间")
            timeline_row.addWidget(self.time_label)
            media_layout.addLayout(timeline_row)

            controls = QHBoxLayout()
            controls.setSpacing(8)
            self.rewind_button = QPushButton("后退 5 秒")
            self.rewind_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward))
            self.rewind_button.setAccessibleName("后退五秒")
            self.rewind_button.setToolTip("后退 5 秒（左方向键）")
            controls.addWidget(self.rewind_button)

            self.play_button = QPushButton("播放")
            self.play_button.setObjectName("playButton")
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.play_button.setAccessibleName("播放")
            self.play_button.setToolTip("播放或暂停（空格键）")
            controls.addWidget(self.play_button)

            self.forward_button = QPushButton("前进 5 秒")
            self.forward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
            self.forward_button.setAccessibleName("前进五秒")
            self.forward_button.setToolTip("前进 5 秒（右方向键）")
            controls.addWidget(self.forward_button)
            controls.addStretch(1)

            speed_label = QLabel("倍速")
            speed_label.setAccessibleName("播放速度标签")
            controls.addWidget(speed_label)
            self.speed_combo = QComboBox()
            self.speed_combo.setAccessibleName("播放速度")
            self.speed_combo.setToolTip("选择播放速度")
            for label, value in (("0.5×", 0.5), ("0.75×", 0.75), ("1.0×", 1.0), ("1.25×", 1.25), ("1.5×", 1.5), ("2.0×", 2.0)):
                self.speed_combo.addItem(label, value)
            self.speed_combo.setCurrentIndex(2)
            controls.addWidget(self.speed_combo)

            self.loop_checkbox = QCheckBox("循环当前片段")
            self.loop_checkbox.setAccessibleName("循环当前转写片段")
            self.loop_checkbox.setToolTip("循环所选片段（Ctrl+L）")
            self.loop_checkbox.setEnabled(False)
            controls.addWidget(self.loop_checkbox)
            media_layout.addLayout(controls)

            feedback_row = QHBoxLayout()
            self.feedback_label = QLabel("")
            self.feedback_label.setObjectName("playerError")
            self.feedback_label.setWordWrap(True)
            self.feedback_label.setAccessibleName("媒体错误信息")
            feedback_row.addWidget(self.feedback_label, 1)
            self.retry_button = QPushButton("重新加载")
            self.retry_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
            self.retry_button.setAccessibleName("重新加载媒体")
            self.retry_button.setVisible(False)
            feedback_row.addWidget(self.retry_button)
            media_layout.addLayout(feedback_row)
            self.splitter.addWidget(media_frame)

            transcript_frame = QFrame()
            transcript_frame.setObjectName("transcriptPanel")
            transcript_layout = QVBoxLayout(transcript_frame)
            transcript_layout.setContentsMargins(10, 10, 10, 10)
            transcript_layout.setSpacing(8)
            self.transcript_heading = QLabel("转写片段")
            self.transcript_heading.setObjectName("transcriptHeading")
            transcript_layout.addWidget(self.transcript_heading)
            self.segment_list = QListWidget()
            self.segment_list.setAccessibleName("带时间戳的转写片段")
            self.segment_list.setAccessibleDescription("点击片段可跳转到对应音视频位置")
            self.segment_list.setUniformItemSizes(True)
            self.segment_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            transcript_layout.addWidget(self.segment_list, 1)
            self.empty_label = QLabel("暂无带时间戳的转写片段")
            self.empty_label.setObjectName("playerStatus")
            self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.empty_label.setAccessibleName("转写片段为空")
            transcript_layout.addWidget(self.empty_label)
            self.splitter.addWidget(transcript_frame)
            self.splitter.setStretchFactor(0, 3)
            self.splitter.setStretchFactor(1, 2)
            root.addWidget(self.splitter, 1)

        def _connect_signals(self) -> None:
            self.play_button.clicked.connect(self.toggle_playback)
            self.rewind_button.clicked.connect(lambda: self.skip_by(-5000))
            self.forward_button.clicked.connect(lambda: self.skip_by(5000))
            self.retry_button.clicked.connect(self.reload)
            self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
            self.loop_checkbox.toggled.connect(self._on_loop_toggled)
            self.segment_list.itemClicked.connect(self._on_segment_clicked)
            self.segment_list.itemActivated.connect(self._on_segment_clicked)
            self.segment_list.currentItemChanged.connect(self._on_segment_selection_changed)
            self.position_slider.sliderPressed.connect(self._on_slider_pressed)
            self.position_slider.sliderReleased.connect(self._on_slider_released)
            self.position_slider.sliderMoved.connect(self._on_slider_moved)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.player.mediaStatusChanged.connect(self._on_media_status_changed)
            self.player.errorOccurred.connect(self._on_error)

        def _install_shortcuts(self) -> None:
            shortcuts = (
                ("Space", self.toggle_playback),
                ("Left", lambda: self.skip_by(-5000)),
                ("Right", lambda: self.skip_by(5000)),
                ("Ctrl+L", self._toggle_loop_shortcut),
            )
            for sequence, callback in shortcuts:
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(callback)
                self._shortcuts.append(shortcut)

        def _set_controls_enabled(self, enabled: bool) -> None:
            for widget in (
                self.play_button,
                self.rewind_button,
                self.forward_button,
                self.position_slider,
                self.speed_combo,
            ):
                widget.setEnabled(enabled)

        @staticmethod
        def _to_url(source: str | Path | QUrl) -> tuple[QUrl, str, bool]:
            if isinstance(source, QUrl):
                url = QUrl(source)
                display = url.toLocalFile() if url.isLocalFile() else url.toDisplayString()
                exists = not url.isLocalFile() or Path(url.toLocalFile()).expanduser().is_file()
                return url, display, exists
            raw = str(source).strip()
            parsed = QUrl(raw)
            if parsed.isValid() and parsed.scheme().casefold() in {"file", "http", "https"}:
                display = parsed.toLocalFile() if parsed.isLocalFile() else parsed.toDisplayString()
                exists = not parsed.isLocalFile() or Path(parsed.toLocalFile()).expanduser().is_file()
                return parsed, display, exists
            path = Path(raw).expanduser().resolve(strict=False)
            return QUrl.fromLocalFile(str(path)), str(path), path.is_file()

        def load_media(
            self,
            source: str | Path | QUrl,
            *,
            start_ms: int = 0,
            autoplay: bool = False,
        ) -> bool:
            """Load media without playing unless ``autoplay`` was explicitly requested."""

            url, display, exists = self._to_url(source)
            self._source = url
            self._source_display = display
            self._pending_start_ms = max(0, int(start_ms))
            self._pending_autoplay = bool(autoplay)
            self._last_error = ""
            self.feedback_label.clear()
            self.retry_button.setVisible(False)
            self.player.stop()
            self._set_controls_enabled(False)
            self.position_slider.setRange(0, 0)
            self.time_label.setText("00:00 / 00:00")
            self.video_widget.setVisible(is_video_source(display))
            self.splitter.setSizes([420, 260] if is_video_source(display) else [180, 500])
            if not url.isValid() or url.isEmpty():
                self._show_error("媒体地址无效。请选择本机已有的音频或视频文件。")
                return False
            if not exists:
                self._show_error(f"媒体文件不存在：{display}。请重新选择原始资料。")
                return False
            self.status_label.setText("正在加载媒体…")
            self.player.setSource(url)
            # QMediaPlayer never starts solely because a source was assigned. Keep
            # the explicit pause here as a documented guard for backend variance.
            if not autoplay:
                self.player.pause()
            return True

        @Slot()
        def reload(self) -> bool:
            if self._source is None:
                return False
            return self.load_media(
                self._source,
                start_ms=self.player.position() or self._pending_start_ms,
                autoplay=False,
            )

        def set_segments(self, segments: Iterable[object] | None) -> None:
            self._cues = normalize_transcript_cues(segments)
            self._cue_starts = tuple(cue.start_ms for cue in self._cues)
            prefix_ends: list[int] = []
            latest_end = 0
            for cue in self._cues:
                latest_end = max(latest_end, cue.end_ms)
                prefix_ends.append(latest_end)
            self._cue_prefix_max_end = tuple(prefix_ends)
            self._active_cue_index = None
            self._loop_bounds = None
            self.loop_checkbox.blockSignals(True)
            self.loop_checkbox.setChecked(False)
            self.loop_checkbox.blockSignals(False)
            self.loop_checkbox.setEnabled(False)
            self.segment_list.clear()
            for index, cue in enumerate(self._cues):
                item = QListWidgetItem(cue.display_text)
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setToolTip(cue.display_text)
                item.setData(Qt.ItemDataRole.AccessibleTextRole, cue.display_text)
                self.segment_list.addItem(item)
            self.transcript_heading.setText(f"转写片段（{len(self._cues)}）")
            self.empty_label.setVisible(not self._cues)
            self.segment_list.setVisible(bool(self._cues))

        def select_segment(self, index: int, *, seek: bool = True) -> TranscriptCue | None:
            if index < 0 or index >= len(self._cues):
                return None
            self.segment_list.setCurrentRow(index)
            cue = self._cues[index]
            if seek:
                self.seek_to(cue.start_ms)
                self.segmentActivated.emit(cue.id, cue.start_ms, cue.end_ms)
            return cue

        @Slot(QListWidgetItem)
        def _on_segment_clicked(self, item: QListWidgetItem) -> None:
            index = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(index, int):
                self.select_segment(index, seek=True)

        @Slot(QListWidgetItem, QListWidgetItem)
        def _on_segment_selection_changed(
            self,
            current: QListWidgetItem | None,
            _previous: QListWidgetItem | None,
        ) -> None:
            index = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
            cue = self._cues[index] if isinstance(index, int) and 0 <= index < len(self._cues) else None
            loopable = cue is not None and cue.duration_ms > 0
            self.loop_checkbox.setEnabled(loopable)
            if self.loop_checkbox.isChecked():
                self._loop_bounds = (cue.start_ms, cue.end_ms) if loopable and cue else None

        @Slot(bool)
        def _on_loop_toggled(self, enabled: bool) -> None:
            row = self.segment_list.currentRow()
            cue = self._cues[row] if 0 <= row < len(self._cues) else None
            if not enabled or cue is None or cue.duration_ms <= 0:
                self._loop_bounds = None
                if enabled:
                    self.loop_checkbox.blockSignals(True)
                    self.loop_checkbox.setChecked(False)
                    self.loop_checkbox.blockSignals(False)
                return
            self._loop_bounds = (cue.start_ms, cue.end_ms)
            if not (cue.start_ms <= self.player.position() < cue.end_ms):
                self.seek_to(cue.start_ms)

        @Slot()
        def _toggle_loop_shortcut(self) -> None:
            if self.loop_checkbox.isEnabled():
                self.loop_checkbox.toggle()

        @Slot()
        def toggle_playback(self) -> None:
            if self._source is None or not self.play_button.isEnabled():
                return
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            else:
                if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
                    self.seek_to(self._loop_bounds[0] if self._loop_bounds else 0)
                self.player.play()

        def skip_by(self, delta_ms: int) -> None:
            self.seek_to(self.player.position() + int(delta_ms))

        def seek_to(self, position_ms: int) -> int:
            target = clamp_position_ms(position_ms, self.player.duration())
            self.player.setPosition(target)
            return target

        @Slot(int)
        def _on_speed_changed(self, index: int) -> None:
            rate = self.speed_combo.itemData(index)
            self.player.setPlaybackRate(max(0.25, min(4.0, _finite_number(rate, 1.0))))

        @Slot()
        def _on_slider_pressed(self) -> None:
            self._slider_dragging = True

        @Slot()
        def _on_slider_released(self) -> None:
            self._slider_dragging = False
            self.seek_to(self.position_slider.value())

        @Slot(int)
        def _on_slider_moved(self, value: int) -> None:
            self._update_time_label(value, self.player.duration())

        @Slot(int)
        def _on_position_changed(self, position: int) -> None:
            if self._loop_bounds is not None:
                loop_start, loop_end = self._loop_bounds
                if position >= loop_end:
                    self.player.setPosition(loop_start)
                    return
            if not self._slider_dragging:
                self.position_slider.setValue(position)
                self._update_time_label(position, self.player.duration())
            self._sync_active_cue(position)

        @Slot(int)
        def _on_duration_changed(self, duration: int) -> None:
            safe_duration = max(0, int(duration))
            self.position_slider.setRange(0, safe_duration)
            self._update_time_label(self.player.position(), safe_duration)
            self._apply_pending_start()

        def _apply_pending_start(self) -> None:
            if self._pending_start_ms:
                self.seek_to(self._pending_start_ms)
            self._pending_start_ms = 0

        def _update_time_label(self, position: int, duration: int) -> None:
            text = f"{format_timestamp(position)} / {format_timestamp(duration)}"
            self.time_label.setText(text)
            self.time_label.setAccessibleDescription(f"当前 {format_timestamp(position)}，总时长 {format_timestamp(duration)}")

        def _sync_active_cue(self, position: int) -> None:
            if not self._cues:
                return
            candidate = bisect_right(self._cue_starts, position) - 1
            index: int | None = None
            while candidate >= 0:
                cue = self._cues[candidate]
                if cue.start_ms <= position < max(cue.start_ms + 1, cue.end_ms):
                    index = candidate
                    break
                if candidate == 0 or self._cue_prefix_max_end[candidate - 1] <= position:
                    break
                candidate -= 1
            if index is None or index == self._active_cue_index:
                return
            self._active_cue_index = index
            self.segment_list.setCurrentRow(index)
            item = self.segment_list.item(index)
            if item is not None:
                self.segment_list.scrollToItem(item)

        @Slot(QMediaPlayer.PlaybackState)
        def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
            playing = state == QMediaPlayer.PlaybackState.PlayingState
            self.play_button.setText("暂停" if playing else "播放")
            self.play_button.setAccessibleName("暂停" if playing else "播放")
            icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
            self.play_button.setIcon(self.style().standardIcon(icon))
            if playing:
                self.status_label.setText("正在播放")
            elif state == QMediaPlayer.PlaybackState.PausedState:
                self.status_label.setText("已暂停")

        @Slot(QMediaPlayer.MediaStatus)
        def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
            if status == QMediaPlayer.MediaStatus.LoadingMedia:
                self.status_label.setText("正在加载媒体…")
            elif status in {
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferingMedia,
                QMediaPlayer.MediaStatus.BufferedMedia,
            }:
                self._set_controls_enabled(True)
                self.status_label.setText("媒体已就绪")
                self._apply_pending_start()
                self.mediaReady.emit(self._source_display)
                if self._pending_autoplay:
                    self._pending_autoplay = False
                    self.player.play()
            elif status == QMediaPlayer.MediaStatus.StalledMedia:
                self.status_label.setText("媒体读取较慢，正在缓冲…")
            elif status == QMediaPlayer.MediaStatus.EndOfMedia:
                if self._loop_bounds is not None:
                    self.player.setPosition(self._loop_bounds[0])
                    self.player.play()
                else:
                    self.status_label.setText("播放结束")
            elif status == QMediaPlayer.MediaStatus.InvalidMedia:
                self._show_error(
                    "无法读取这个媒体文件。请确认文件未损坏，并尝试转换为 MP3、M4A、WAV 或 MP4 格式。"
                )

        @Slot(QMediaPlayer.Error, str)
        def _on_error(self, _error: QMediaPlayer.Error, error_string: str) -> None:
            detail = str(error_string or self.player.errorString() or "系统未提供更多信息").strip()
            self._show_error(
                f"媒体播放失败：{detail}。请确认文件完整且系统支持它的编码格式。"
            )

        def _show_error(self, message: str) -> None:
            self._last_error = str(message).strip()
            self.player.pause()
            self._set_controls_enabled(False)
            self.status_label.setText("媒体不可用")
            self.feedback_label.setText(self._last_error)
            self.retry_button.setVisible(self._source is not None)
            self.mediaError.emit(self._last_error)

        def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
            self.player.stop()
            self.player.setSource(QUrl())
            super().closeEvent(event)


    AudioPlayerDialog = MediaPlayerDialog

else:

    class MediaPlayerDialog:  # pragma: no cover - instantiated only without desktop extras
        """Helpful placeholder when PySide6-Addons is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(multimedia_unavailable_message())


    AudioPlayerDialog = MediaPlayerDialog


__all__ = [
    "AudioPlayerDialog",
    "MediaPlayerDialog",
    "QT_MULTIMEDIA_AVAILABLE",
    "QT_MULTIMEDIA_IMPORT_ERROR",
    "TranscriptCue",
    "active_cue_index",
    "clamp_position_ms",
    "format_timestamp",
    "is_video_source",
    "multimedia_unavailable_message",
    "normalize_transcript_cues",
]
