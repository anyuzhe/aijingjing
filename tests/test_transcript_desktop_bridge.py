from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from media_knowledge.desktop.controller import DesktopController
from media_knowledge.embedding import HashEmbeddingProvider
from media_knowledge.indexing import IndexingService
from media_knowledge.models import ContentSegment, KnowledgeDocument, SourceReference
from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.transcripts import (
    TranscriptQuality,
    TranscriptRepository,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptSpeaker,
    TranscriptV2,
)


class TranscriptDesktopBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.controller = DesktopController(self.root / "data", migrate_legacy=False)
        self.controller.settings.embedding_provider = "hash"
        self.controller.settings.embedding_model = "hash-384-v1"
        self.media = self.root / "meeting.mp4"
        self.media.write_bytes(b"test-media")
        self.run_id = "run-desktop-bridge"
        source = SourceReference(
            source_id="file:meeting",
            media_type="video",
            title="项目会议",
            original_uri=str(self.media),
            local_path=str(self.media),
            checksum="a" * 64,
        )
        segments = [
            ContentSegment(
                "run-desktop-bridge-seg-1",
                0.0,
                "speech",
                text="微压需要提高",
                location={"timestamp_start": 0.0, "timestamp_end": 1.5, "speaker_id": "S1"},
                metadata={
                    "speaker_id": "S1",
                    "speaker_name": "张工",
                    "asr_run_id": self.run_id,
                    "quality_status": "review",
                },
            ),
            ContentSegment(
                "frame-1",
                2.0,
                "image",
                description="画面展示了一张结构示意图",
                location={"timestamp_start": 2.0, "timestamp_end": 2.0},
                asset=str(self.root / "frame.jpg"),
            ),
        ]
        document = KnowledgeDocument(
            "file:meeting",
            "项目会议",
            "video",
            segments,
            source,
        )
        with KnowledgeDatabase(self.controller.paths.database) as database:
            report = IndexingService(
                database,
                HashEmbeddingProvider(dimensions=384, model="hash-384-v1"),
            ).index_document(document)
            self.document_id = report.document_id
            transcript = TranscriptV2(
                TranscriptSource(
                    self.media.name,
                    "a" * 64,
                    2000,
                    str(self.media),
                ),
                TranscriptRun(
                    self.run_id,
                    "chinese-accuracy",
                    "qwen3-mlx",
                    "Qwen3-ASR-1.7B",
                    "zh",
                    True,
                    "pyannote",
                ),
                [TranscriptSpeaker("S1", "张工")],
                [
                    TranscriptSegment(
                        "run-desktop-bridge-seg-1",
                        0,
                        0,
                        1500,
                        "S1",
                        "微压需要提高",
                        confidence=0.91,
                    )
                ],
                TranscriptQuality("review", ("请人工确认术语",), {"coverage": 1.0}),
            )
            TranscriptRepository(database).save_transcript(
                transcript,
                document_id=self.document_id,
            )
            database.set_document_enabled(self.document_id, False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_latest_transcript_exposes_local_media_and_selected_model(self) -> None:
        result = self.controller.latest_transcript(self.document_id)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["media_path"], str(self.media))
        self.assertEqual(result["run"]["model"], "Qwen3-ASR-1.7B")
        self.assertEqual(result["transcript"]["format"], "ai-jingjing-transcript-v2")

    def test_correction_rebuilds_speech_index_and_preserves_visual_chunk(self) -> None:
        with self.assertRaisesRegex(ValueError, "尚未通过人工复核"):
            self.controller.refresh_transcript_index(self.run_id)

        self.controller.approve_transcript_for_retrieval(self.run_id)
        with KnowledgeDatabase(self.controller.paths.database) as database:
            TranscriptRepository(database).update_corrected_text(
                "run-desktop-bridge-seg-1",
                "围压需要提高",
                reason="人工确认专业术语",
                actor="user",
            )

        report = self.controller.refresh_transcript_index(
            self.run_id,
            affected_segment_ids=("run-desktop-bridge-seg-1",),
        )

        self.assertEqual(report["affected_segment_ids"], ["run-desktop-bridge-seg-1"])
        with KnowledgeDatabase(self.controller.paths.database) as database:
            contents = [str(row["content"]) for row in database.get_chunks(self.document_id).values()]
        self.assertTrue(any("围压需要提高" in content for content in contents))
        self.assertTrue(any("结构示意图" in content for content in contents))
        self.assertFalse(any("微压需要提高" in content for content in contents))

    def test_human_approval_updates_quality_and_enables_document(self) -> None:
        result = self.controller.approve_transcript_for_retrieval(self.run_id)

        self.assertTrue(result["enabled"])
        with KnowledgeDatabase(self.controller.paths.database) as database:
            run = TranscriptRepository(database).get_run(self.run_id)
            document = database.get_document(self.document_id)
        assert run is not None and document is not None
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.quality["status"], "pass")
        self.assertTrue(run.quality["metrics"]["human_reviewed"])
        self.assertTrue(document["enabled"])


if __name__ == "__main__":
    unittest.main()
