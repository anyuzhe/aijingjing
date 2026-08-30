from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from media_knowledge.storage import KnowledgeDatabase
from media_knowledge.transcripts import (
    CorrectionSuggestion,
    GlossaryTermInput,
    TranscriptQuality,
    TranscriptRepository,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptSpeaker,
    TranscriptV2,
    TranscriptWord,
    apply_confirmed_corrections,
    evaluate_transcript_quality,
    transcript_from_dict,
)


class TranscriptSchemaTests(unittest.TestCase):
    def test_v1_remains_readable_and_round_trips_as_v2(self) -> None:
        legacy = {
            "format": "ai-jingjing-transcript-v1",
            "source": "项目会议.wav",
            "engine": "faster-whisper",
            "model": "small",
            "language": "zh",
            "duration_seconds": 3.5,
            "integrity": {"status": "warn", "warnings": ["尾部静音"]},
            "segments": [
                {"start": 0.0, "end": 1.25, "text": "结构面参数", "confidence": 0.81},
                {"start": 1.25, "end": 3.5, "text": "需要检查"},
            ],
        }

        converted = transcript_from_dict(legacy)

        self.assertEqual(converted.format, "ai-jingjing-transcript-v2")
        self.assertEqual(converted.source.name, "项目会议.wav")
        self.assertEqual(converted.source.duration_ms, 3500)
        self.assertEqual(converted.run.provider, "faster-whisper")
        self.assertEqual(converted.segments[0].raw_text, "结构面参数")
        self.assertEqual((converted.segments[0].start_ms, converted.segments[0].end_ms), (0, 1250))
        self.assertEqual(converted.quality.status, "review")
        self.assertEqual(
            TranscriptV2.from_dict(converted.to_dict()).to_dict(),
            converted.to_dict(),
        )

    def test_v2_keeps_raw_corrected_words_speakers_run_and_quality(self) -> None:
        transcript = _sample_transcript()
        restored = TranscriptV2.from_json(transcript.to_json())

        self.assertEqual(restored.segments[0].raw_text, "微压需要提高")
        self.assertEqual(restored.segments[0].corrected_text, "围压需要提高。")
        self.assertEqual(restored.segments[0].words[0].text, "微压")
        self.assertEqual(restored.speakers[0].display_name, "张工")
        self.assertEqual(restored.run.model, "Qwen3-ASR-1.7B")
        self.assertEqual(restored.quality.status, "review")


class TranscriptRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = KnowledgeDatabase(Path(self.temporary.name) / "knowledge.db")
        self.repository = TranscriptRepository(self.database)

    def tearDown(self) -> None:
        self.database.close()
        self.temporary.cleanup()

    def test_schema_12_migrates_all_transcript_fact_tables(self) -> None:
        versions = {
            int(row["version"])
            for row in self.database.connection.execute(
                "SELECT version FROM schema_migrations"
            )
        }
        self.assertIn(12, versions)
        for table in (
            "transcription_runs",
            "transcript_speakers",
            "transcript_segments",
            "transcript_edits",
            "asr_glossaries",
            "asr_glossary_terms",
        ):
            self.assertEqual(self.database.status()[table], 0)

    def test_save_load_and_edit_never_overwrite_raw_text(self) -> None:
        transcript = _sample_transcript()
        self.repository.save_transcript(transcript, transcript_path="transcripts/run.json")

        updated = self.repository.update_corrected_text(
            "seg_0001",
            "围压需要进一步提高。",
            edit_type="terminology",
            reason="人工确认专业词并补充程度词",
            actor="user",
        )

        self.assertEqual(updated.raw_text, "微压需要提高")
        self.assertEqual(updated.corrected_text, "围压需要进一步提高。")
        loaded = self.repository.get_transcript("asr-run-001")
        assert loaded is not None
        self.assertEqual(loaded.segments[0].raw_text, "微压需要提高")
        self.assertEqual(loaded.segments[0].corrected_text, "围压需要进一步提高。")
        edits = self.repository.list_edits(run_id="asr-run-001")
        self.assertEqual(edits[0].before_text, "围压需要提高。")
        self.assertEqual(edits[0].after_text, "围压需要进一步提高。")
        self.assertEqual(edits[0].edit_type, "terminology")
        self.assertEqual(edits[0].actor, "user")

        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.connection:
                self.database.connection.execute(
                    "UPDATE transcript_segments SET raw_text='被覆盖' WHERE id='seg_0001'"
                )

    def test_speaker_rename_merge_and_segment_reassignment_are_audited(self) -> None:
        self.repository.save_transcript(_sample_transcript(two_segments=True))

        created = self.repository.create_speaker("asr-run-001", "spk_02")
        self.assertEqual(created.name_source, "automatic")
        updated_meta = self.repository.update_speaker(
            "asr-run-001", "spk_02", metadata={"channel": "remote"}
        )
        self.assertEqual(updated_meta.metadata["channel"], "remote")
        self.assertTrue(self.repository.delete_speaker("asr-run-001", "spk_02"))

        renamed = self.repository.rename_speaker("asr-run-001", "spk_00", "王工")
        self.assertEqual((renamed.display_name, renamed.name_source), ("王工", "manual"))
        reassigned = self.repository.reassign_segment("seg_0002", "spk_00")
        self.assertEqual(reassigned.speaker_id, "spk_00")
        affected = self.repository.merge_speakers(
            "asr-run-001", "spk_01", "spk_00", reason="同一人的误分段"
        )
        self.assertEqual(affected, 0)
        source = self.repository.get_speaker("asr-run-001", "spk_01")
        assert source is not None
        self.assertEqual(source.metadata["merged_into"], "spk_00")

        edit_types = {item.edit_type for item in self.repository.list_edits(run_id="asr-run-001")}
        self.assertEqual(
            edit_types,
            {
                "speaker_create", "speaker_update", "speaker_delete",
                "speaker_rename", "speaker_assignment", "speaker_merge",
            },
        )

    def test_glossary_crud_and_context_terms_keep_scope_order(self) -> None:
        global_book = self.repository.create_glossary("全局术语", scope="global")
        space_book = self.repository.create_glossary(
            "岩体力学", scope="knowledge_space", scope_id="rock"
        )
        source_book = self.repository.create_glossary(
            "本次会议", scope="source", scope_id="sha256:abc"
        )
        self.repository.add_glossary_term(
            global_book.id,
            GlossaryTermInput("FLAC3D", variants=("flac 3d",)),
        )
        self.repository.add_glossary_term(space_book.id, GlossaryTermInput("围压"))
        self.repository.add_glossary_term(source_book.id, GlossaryTermInput("SMRM"))

        self.assertEqual(
            self.repository.context_terms(
                knowledge_space_id="rock", source_id="sha256:abc"
            ),
            ("FLAC3D", "围压", "SMRM"),
        )
        self.assertTrue(self.repository.delete_glossary(source_book.id))
        self.assertIsNone(self.repository.get_glossary(source_book.id))


class TranscriptCorrectionAndQualityTests(unittest.TestCase):
    def test_glossary_suggestion_requires_confirmation_and_exact_span(self) -> None:
        text = "本次试验采用微压加载，微压传感器正常。"
        suggestion = CorrectionSuggestion(
            before="微压",
            after="围压",
            reason="命中岩体力学词库",
            confidence=0.84,
            confirmed=False,
            start_char=6,
            end_char=8,
        )

        self.assertEqual(apply_confirmed_corrections(text, [suggestion]), text)
        confirmed = suggestion.confirm()
        self.assertEqual(
            apply_confirmed_corrections(text, [confirmed]),
            "本次试验采用围压加载，微压传感器正常。",
        )
        with self.assertRaises(ValueError):
            apply_confirmed_corrections(
                text,
                [CorrectionSuggestion("微压", "围压", "无定位", 0.9, True)],
            )

    def test_quality_gate_distinguishes_pass_review_and_fail(self) -> None:
        passing = _sample_transcript()
        passing.quality = TranscriptQuality(status="pass")
        passing.segments[0].corrected_text = "结构面参数需要检查。"
        passing.segments[0].raw_text = "结构面参数需要检查"
        self.assertEqual(evaluate_transcript_quality(passing).status, "pass")

        review = _sample_transcript(two_segments=True)
        review.segments[1].speaker_id = "speaker_unknown"
        review.segments[1].flags = ("truncated",)
        report = evaluate_transcript_quality(review, expected_speakers=2)
        self.assertEqual(report.status, "review")
        self.assertIn("truncated", {issue.code for issue in report.issues})
        self.assertIn("speaker_unknown", {issue.code for issue in report.issues})

        failing = _sample_transcript()
        failing.segments[0].end_ms = 40000
        report = evaluate_transcript_quality(failing)
        self.assertEqual(report.status, "fail")
        self.assertIn("timestamp_out_of_bounds", {issue.code for issue in report.issues})

        audio_failure = evaluate_transcript_quality(
            passing,
            audio_metrics={"decode_ok": False, "duration_ms": 0, "clipping_ratio": 0.4},
        )
        self.assertEqual(audio_failure.status, "fail")
        self.assertIn("audio_decode_failed", {issue.code for issue in audio_failure.issues})


def _sample_transcript(*, two_segments: bool = False) -> TranscriptV2:
    segments = [
        TranscriptSegment(
            id="seg_0001",
            ordinal=0,
            start_ms=0,
            end_ms=1500,
            speaker_id="spk_00",
            raw_text="微压需要提高",
            corrected_text="围压需要提高。",
            confidence=0.86,
            words=(TranscriptWord(0, 500, "微压", 0.82),),
        )
    ]
    if two_segments:
        segments.append(
            TranscriptSegment(
                id="seg_0002",
                ordinal=1,
                start_ms=1500,
                end_ms=3000,
                speaker_id="spk_01",
                raw_text="先做对照计算",
                corrected_text="先做对照计算。",
                confidence=0.9,
            )
        )
    return TranscriptV2(
        source=TranscriptSource("项目会议.wav", "a" * 64, 3000),
        run=TranscriptRun(
            id="asr-run-001",
            profile="chinese-accuracy",
            provider="qwen3-mlx",
            model="Qwen3-ASR-1.7B",
            language="Chinese",
            word_timestamps=True,
            diarization_provider="pyannote",
            context_profile="岩体力学",
        ),
        speakers=[
            TranscriptSpeaker("spk_00", "张工", "manual"),
            TranscriptSpeaker("spk_01", "李工", "manual"),
        ],
        segments=segments,
        quality=TranscriptQuality(status="review", warnings=("等待术语确认",)),
    )


if __name__ == "__main__":
    unittest.main()
