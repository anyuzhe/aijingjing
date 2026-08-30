from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from media_knowledge.desktop.audio_player import (  # noqa: E402
    QT_MULTIMEDIA_AVAILABLE,
    MediaPlayerDialog,
    TranscriptCue,
    active_cue_index,
    clamp_position_ms,
    format_timestamp,
    is_video_source,
    multimedia_unavailable_message,
    normalize_transcript_cues,
)


class AudioPlayerHelperTests(unittest.TestCase):
    def test_timestamp_and_position_helpers_are_defensive(self) -> None:
        self.assertEqual(format_timestamp(-200), "00:00")
        self.assertEqual(format_timestamp(65_432), "01:05")
        self.assertEqual(format_timestamp(3_661_000), "01:01:01")
        self.assertEqual(format_timestamp(float("nan")), "00:00")
        self.assertEqual(clamp_position_ms(-10, 1000), 0)
        self.assertEqual(clamp_position_ms(1200, 1000), 1000)
        self.assertEqual(clamp_position_ms(1200, 0), 1200)

    def test_cues_accept_v2_objects_and_legacy_seconds(self) -> None:
        class V2Segment:
            id = "v2"
            ordinal = 9
            start_ms = 3000
            end_ms = 4500
            speaker_id = "speaker_1"
            raw_text = "原始文字"
            corrected_text = "校订文字"

        cues = normalize_transcript_cues(
            [
                V2Segment(),
                {"id": "old", "start": 1.25, "end": 2.5, "text": "旧版", "speaker": "讲师"},
            ]
        )
        self.assertEqual([cue.id for cue in cues], ["old", "v2"])
        self.assertEqual(cues[0].start_ms, 1250)
        self.assertEqual(cues[0].end_ms, 2500)
        self.assertEqual(cues[1].text, "校订文字")
        self.assertEqual(cues[1].speaker, "speaker_1")
        self.assertIn("[00:01]", cues[0].display_text)

    def test_active_cue_prefers_latest_overlapping_segment(self) -> None:
        cues = (
            TranscriptCue("first", 0, 4000, "第一段"),
            TranscriptCue("overlap", 2500, 5000, "重叠段"),
            TranscriptCue("last", 6000, 7000, "最后一段"),
        )
        self.assertEqual(active_cue_index(cues, 1000), 0)
        self.assertEqual(active_cue_index(cues, 3000), 1)
        self.assertIsNone(active_cue_index(cues, 5500))
        self.assertEqual(active_cue_index(cues, 6000), 2)

    def test_video_detection_uses_url_path_without_query(self) -> None:
        self.assertTrue(is_video_source("https://example.test/video.MP4?token=1"))
        self.assertTrue(is_video_source(Path("training.mov")))
        self.assertFalse(is_video_source("meeting.m4a"))

    @unittest.skipIf(QT_MULTIMEDIA_AVAILABLE, "仅验证缺少可选组件时的错误路径")
    def test_missing_multimedia_dependency_has_actionable_message(self) -> None:
        message = multimedia_unavailable_message()
        self.assertIn("PySide6-Addons", message)
        with self.assertRaisesRegex(RuntimeError, "PySide6-Addons"):
            MediaPlayerDialog()


@unittest.skipUnless(QT_MULTIMEDIA_AVAILABLE, "当前环境未安装 PySide6-Addons")
class MediaPlayerDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.dialog = MediaPlayerDialog(
            segments=[
                {"id": "s1", "start_ms": 1000, "end_ms": 2500, "raw_text": "第一段"},
                {"id": "s2", "start_ms": 3000, "end_ms": 6000, "raw_text": "第二段"},
            ]
        )

    def tearDown(self) -> None:
        self.dialog.close()
        self.dialog.deleteLater()
        self.application.processEvents()

    def test_dialog_defaults_to_stopped_and_has_accessible_controls(self) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        self.assertEqual(self.dialog.player.playbackState(), QMediaPlayer.PlaybackState.StoppedState)
        self.assertFalse(self.dialog.play_button.isEnabled())
        self.assertEqual(self.dialog.play_button.accessibleName(), "播放")
        self.assertEqual(self.dialog.position_slider.accessibleName(), "播放进度")
        self.assertEqual(self.dialog.segment_list.accessibleName(), "带时间戳的转写片段")
        self.assertEqual(self.dialog.segment_list.count(), 2)

    def test_select_segment_seeks_and_enables_loop(self) -> None:
        with patch.object(self.dialog, "seek_to", return_value=3000) as seek:
            selected = self.dialog.select_segment(1)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.id, "s2")
        seek.assert_called_once_with(3000)
        self.assertTrue(self.dialog.loop_checkbox.isEnabled())
        self.dialog.loop_checkbox.setChecked(True)
        self.assertEqual(self.dialog._loop_bounds, (3000, 6000))

    def test_missing_local_file_shows_recoverable_error_without_playing(self) -> None:
        missing = Path(tempfile.gettempdir()) / "ai-jingjing-definitely-missing.mp3"
        missing.unlink(missing_ok=True)
        self.assertFalse(self.dialog.load_media(missing, autoplay=True))
        self.assertIn("文件不存在", self.dialog.last_error)
        self.assertFalse(self.dialog.retry_button.isHidden())
        self.assertFalse(self.dialog.play_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
