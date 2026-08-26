from __future__ import annotations

import unittest

from media_knowledge.chunking import MediaAwareChunker
from media_knowledge.models import ContentSegment, KnowledgeDocument, SourceReference


def make_document(media_type: str, segments: list[ContentSegment]) -> KnowledgeDocument:
    source = SourceReference(source_id=f"src-{media_type}", media_type=media_type, title=media_type)
    return KnowledgeDocument(
        source_id=source.source_id,
        title=media_type,
        media_type=media_type,
        segments=segments,
        source=source,
        document_id=f"doc-{media_type}",
    )


class MediaAwareChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunker = MediaAwareChunker(target_tokens=64, max_tokens=96)

    def test_pdf_chunks_never_lose_page_number(self) -> None:
        document = make_document(
            "pdf",
            [
                ContentSegment("p1-a", 1, "text", text="第一页介绍系统背景。", location={"page": 1}),
                ContentSegment("p1-b", 2, "image", description="系统输入输出图。", location={"page": 1}),
                ContentSegment("p2-a", 3, "text", text="第二页介绍实验。", location={"page": 2}),
            ],
        )
        chunks = self.chunker.chunk(document)
        self.assertEqual([chunk.source_reference.page_number for chunk in chunks], [1, 2])
        self.assertIn("系统输入输出图", chunks[0].content)

    def test_presentation_chunks_preserve_slide_and_render(self) -> None:
        document = make_document(
            "presentation",
            [
                ContentSegment(
                    "s6", 6, "slide", text="FAST-LIVO2 system architecture", location={"slide": 6}, asset="slides/slide-006.png"
                ),
                ContentSegment("s7", 7, "slide", text="Experiment results", location={"slide": 7}),
            ],
        )
        chunks = self.chunker.chunk(document)
        self.assertEqual([chunk.source_reference.slide_number for chunk in chunks], [6, 7])
        self.assertEqual(chunks[0].source_reference.image_path, "slides/slide-006.png")

    def test_video_chunk_preserves_time_range(self) -> None:
        document = make_document(
            "video",
            [
                ContentSegment(
                    "v1", 1, "speech", text="The lecturer introduces the sensor stack.",
                    location={"timestamp_start": 10.0, "timestamp_end": 20.0},
                ),
                ContentSegment(
                    "v2", 2, "video_frame", description="LiDAR, IMU and camera feed the fusion node.",
                    location={"timestamp_start": 20.0, "timestamp_end": 28.0},
                ),
            ],
        )
        chunks = self.chunker.chunk(document)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_reference.timestamp_start, 10.0)
        self.assertEqual(chunks[0].source_reference.timestamp_end, 28.0)

    def test_markdown_heading_path_is_preserved(self) -> None:
        document = make_document(
            "document",
            [ContentSegment("m1", 1, "text", text="# Project\n\nIntro.\n\n## Architecture\n\nPlanner calls tools.")],
        )
        chunks = self.chunker.chunk(document)
        self.assertEqual(chunks[0].heading_path, ["Project"])
        self.assertEqual(chunks[1].heading_path, ["Project", "Architecture"])


if __name__ == "__main__":
    unittest.main()
