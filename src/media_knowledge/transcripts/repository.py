from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from ..models import utcnow_iso
from ..storage.database import KnowledgeDatabase
from .correction import CorrectionSuggestion, apply_confirmed_corrections
from .schema import (
    QUALITY_STATUSES,
    SPEAKER_NAME_SOURCES,
    TranscriptQuality,
    TranscriptRun,
    TranscriptSegment,
    TranscriptSource,
    TranscriptSpeaker,
    TranscriptV2,
    TranscriptWord,
)


RUN_STATUSES = frozenset({"queued", "running", "completed", "review", "failed", "cancelled"})
GLOSSARY_SCOPES = frozenset({"global", "knowledge_space", "source"})
_SPACE_RE = re.compile(r"\s+")
_UNSET = object()


def _clean(value: object, *, required: bool = False, field_name: str = "文本") -> str:
    result = str(value or "").strip()
    if required and not result:
        raise ValueError(f"{field_name}不能为空")
    return result


def _normalize(value: object) -> str:
    return _SPACE_RE.sub(" ", _clean(value)).casefold()


def _dump(value: object, *, default: object) -> str:
    candidate = default if value is None else value
    try:
        return json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("数据必须可以序列化为 JSON") from exc


def _load_mapping(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_sequence(value: object) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


@dataclass(frozen=True, slots=True)
class TranscriptionRunRecord:
    id: str
    document_id: str | None
    source_name: str
    source_checksum: str
    source_duration_ms: int
    profile: str
    provider: str
    model: str
    language: str | None
    status: str
    config: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    transcript_path: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TranscriptEdit:
    id: str
    run_id: str
    segment_id: str | None
    target_type: str
    target_id: str
    before_text: str
    after_text: str
    edit_type: str
    confirmed: bool
    reason: str
    actor: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Glossary:
    id: str
    name: str
    scope: str
    scope_id: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GlossaryTermInput:
    canonical_term: str
    variants: tuple[str, ...] = ()
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    id: str
    glossary_id: str
    canonical_term: str
    variants: tuple[str, ...] = ()
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["variants"] = list(self.variants)
        return value


class TranscriptRepository:
    """SQLite fact repository for immutable ASR output and audited corrections."""

    def __init__(self, database: KnowledgeDatabase):
        self.database = database

    @property
    def connection(self) -> sqlite3.Connection:
        return self.database.connection

    def save_transcript(
        self,
        transcript: TranscriptV2,
        *,
        document_id: str | None = None,
        transcript_path: str | None = None,
        status: str | None = None,
    ) -> TranscriptV2:
        self._validate_transcript(transcript)
        run_id = transcript.run.id
        existing = self.get_run(run_id)
        if existing is not None:
            current = self.get_transcript(run_id)
            assert current is not None
            self._assert_same_raw_facts(current, transcript)
            return current
        if document_id and self.database.get_document(document_id) is None:
            raise ValueError(f"文档不存在：{document_id}")
        run_status = status or {
            "pass": "completed",
            "review": "review",
            "fail": "failed",
        }.get(transcript.quality.status, "review")
        if run_status not in RUN_STATUSES:
            raise ValueError(f"不支持的转写任务状态：{run_status}")
        now = utcnow_iso()
        config = dict(transcript.run.config)
        config["_transcript_v2"] = {
            "word_timestamps": transcript.run.word_timestamps,
            "diarization_provider": transcript.run.diarization_provider,
            "context_profile": transcript.run.context_profile,
            "fallback": transcript.run.fallback,
            "source_original_uri": transcript.source.original_uri,
            "source_metadata": transcript.source.metadata,
            "transcript_metadata": transcript.metadata,
        }
        speakers = {speaker.id: speaker for speaker in transcript.speakers}
        for segment in transcript.segments:
            if segment.speaker_id and segment.speaker_id not in speakers:
                speakers[segment.speaker_id] = TranscriptSpeaker(segment.speaker_id)
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO transcription_runs(
                           id, document_id, source_name, source_checksum, source_duration_ms,
                           profile, provider, model, language, status, config_json,
                           quality_json, transcript_path, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        document_id,
                        transcript.source.name,
                        transcript.source.sha256,
                        transcript.source.duration_ms,
                        transcript.run.profile,
                        transcript.run.provider,
                        transcript.run.model,
                        transcript.run.language,
                        run_status,
                        _dump(config, default={}),
                        _dump(transcript.quality.to_dict(), default={}),
                        _clean(transcript_path) or None,
                        now,
                        now,
                    ),
                )
                for speaker in speakers.values():
                    self._insert_speaker(run_id, speaker, now)
                for segment in sorted(transcript.segments, key=lambda item: item.ordinal):
                    self.connection.execute(
                        """INSERT INTO transcript_segments(
                               id, run_id, ordinal, start_ms, end_ms, speaker_id,
                               raw_text, corrected_text, confidence, flags_json,
                               words_json, metadata_json, created_at, updated_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            segment.id,
                            run_id,
                            segment.ordinal,
                            segment.start_ms,
                            segment.end_ms,
                            segment.speaker_id,
                            segment.raw_text,
                            segment.corrected_text,
                            segment.confidence,
                            _dump(list(segment.flags), default=[]),
                            _dump([word.to_dict() for word in segment.words], default=[]),
                            _dump(segment.metadata, default={}),
                            now,
                            now,
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法保存转写事实：{exc}") from exc
        saved = self.get_transcript(run_id)
        assert saved is not None
        return saved

    def get_run(self, run_id: str) -> TranscriptionRunRecord | None:
        row = self.connection.execute(
            "SELECT * FROM transcription_runs WHERE id=?", (_clean(run_id),)
        ).fetchone()
        return self._run_from_row(row) if row is not None else None

    def list_runs(
        self,
        *,
        document_id: str | None = None,
        source_checksum: str | None = None,
        statuses: Sequence[str] = (),
        limit: int = 100,
        offset: int = 0,
    ) -> list[TranscriptionRunRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if document_id:
            clauses.append("document_id=?")
            params.append(document_id)
        if source_checksum:
            clauses.append("source_checksum=?")
            params.append(source_checksum)
        clean_statuses = tuple(dict.fromkeys(_clean(item) for item in statuses if _clean(item)))
        if clean_statuses:
            if any(item not in RUN_STATUSES for item in clean_statuses):
                raise ValueError("包含不支持的转写任务状态")
            clauses.append("status IN (" + ",".join("?" for _ in clean_statuses) + ")")
            params.extend(clean_statuses)
        safe_limit = min(1000, max(1, int(limit)))
        safe_offset = max(0, int(offset))
        query = "SELECT * FROM transcription_runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, id LIMIT ? OFFSET ?"
        params.extend((safe_limit, safe_offset))
        return [self._run_from_row(row) for row in self.connection.execute(query, params).fetchall()]

    def get_transcript(self, run_id: str) -> TranscriptV2 | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        private = run.config.get("_transcript_v2")
        private = private if isinstance(private, Mapping) else {}
        public_config = dict(run.config)
        public_config.pop("_transcript_v2", None)
        source = TranscriptSource(
            run.source_name,
            run.source_checksum,
            run.source_duration_ms,
            _clean(private.get("source_original_uri")) or None,
            dict(private.get("source_metadata")) if isinstance(private.get("source_metadata"), Mapping) else {},
        )
        transcript_run = TranscriptRun(
            run.id,
            run.profile,
            run.provider,
            run.model,
            run.language,
            bool(private.get("word_timestamps", False)),
            _clean(private.get("diarization_provider")) or None,
            _clean(private.get("context_profile")) or None,
            dict(private.get("fallback")) if isinstance(private.get("fallback"), Mapping) else None,
            public_config,
        )
        quality = TranscriptQuality.from_dict(run.quality)
        metadata = dict(private.get("transcript_metadata")) if isinstance(private.get("transcript_metadata"), Mapping) else {}
        return TranscriptV2(
            source,
            transcript_run,
            self.list_speakers(run.id),
            self.list_segments(run.id),
            quality,
            metadata,
        )

    def update_run_status(
        self,
        run_id: str,
        status: str,
        *,
        quality: TranscriptQuality | Mapping[str, Any] | None = None,
        transcript_path: str | None = None,
    ) -> TranscriptionRunRecord:
        clean_status = _clean(status)
        if clean_status not in RUN_STATUSES:
            raise ValueError(f"不支持的转写任务状态：{clean_status}")
        current = self._require_run(run_id)
        quality_value = current.quality
        if isinstance(quality, TranscriptQuality):
            quality_value = quality.to_dict()
        elif isinstance(quality, Mapping):
            quality_value = dict(quality)
        path_value = current.transcript_path if transcript_path is None else (_clean(transcript_path) or None)
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE transcription_runs
                   SET status=?, quality_json=?, transcript_path=?, updated_at=? WHERE id=?""",
                (clean_status, _dump(quality_value, default={}), path_value, now, current.id),
            )
        return self._require_run(current.id)

    def delete_run(self, run_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM transcription_runs WHERE id=?", (_clean(run_id),)
            )
        return cursor.rowcount > 0

    def write_latest_v2(
        self,
        run_id: str,
        path: str | Path,
        *,
        overwrite: bool = False,
        expected_existing_checksum: str | None = None,
        allowed_root: str | Path | None = None,
    ) -> Path:
        """Atomically regenerate V2 JSON from current editable database facts.

        Overwrite requires an explicit flag. Callers may additionally provide the
        previous file checksum for compare-and-swap protection against stale UI
        state or another process exporting concurrently.
        """

        from .exporter import atomic_write_bytes, safe_output_path

        transcript = self.get_transcript(run_id)
        if transcript is None:
            raise KeyError(f"转写任务不存在：{run_id}")
        target = safe_output_path(path, suffixes=(".json",), allowed_root=allowed_root)
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"转写文件已存在：{target}")
            if target.is_symlink():
                raise ValueError("拒绝覆盖符号链接")
            if expected_existing_checksum is not None:
                actual = hashlib.sha256(target.read_bytes()).hexdigest()
                if actual != _clean(expected_existing_checksum):
                    raise RuntimeError("现有 V2 文件已发生变化，拒绝覆盖")
        elif expected_existing_checksum is not None:
            raise RuntimeError("预期覆盖的 V2 文件不存在")
        atomic_write_bytes(target, transcript.to_json().encode("utf-8"), overwrite=overwrite)
        return target

    def get_segment(self, segment_id: str) -> TranscriptSegment | None:
        row = self.connection.execute(
            "SELECT * FROM transcript_segments WHERE id=?", (_clean(segment_id),)
        ).fetchone()
        return self._segment_from_row(row) if row is not None else None

    def list_segments(
        self,
        run_id: str,
        *,
        speaker_id: str | None = None,
    ) -> list[TranscriptSegment]:
        if speaker_id is None:
            rows = self.connection.execute(
                "SELECT * FROM transcript_segments WHERE run_id=? ORDER BY ordinal, id",
                (_clean(run_id),),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """SELECT * FROM transcript_segments
                   WHERE run_id=? AND speaker_id=? ORDER BY ordinal, id""",
                (_clean(run_id), _clean(speaker_id)),
            ).fetchall()
        return [self._segment_from_row(row) for row in rows]

    def update_corrected_text(
        self,
        segment_id: str,
        corrected_text: str | None,
        *,
        edit_type: str = "manual_correction",
        reason: str = "",
        actor: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> TranscriptSegment:
        row = self._require_segment_row(segment_id)
        before = row["corrected_text"] if row["corrected_text"] is not None else row["raw_text"]
        after_value = None if corrected_text is None else str(corrected_text)
        after = row["raw_text"] if after_value is None else after_value
        if before == after:
            return self._segment_from_row(row)
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                "UPDATE transcript_segments SET corrected_text=?, updated_at=? WHERE id=?",
                (after_value, now, row["id"]),
            )
            self._insert_edit(
                run_id=row["run_id"],
                segment_id=row["id"],
                target_type="segment_text",
                target_id=row["id"],
                before_text=str(before),
                after_text=str(after),
                edit_type=edit_type,
                reason=reason,
                actor=actor,
                metadata=metadata,
                created_at=now,
            )
            self._touch_run(row["run_id"], now)
        updated = self.get_segment(row["id"])
        assert updated is not None
        return updated

    def apply_correction_suggestion(
        self,
        segment_id: str,
        suggestion: CorrectionSuggestion,
        *,
        actor: str = "user",
    ) -> TranscriptSegment:
        if suggestion.segment_id and suggestion.segment_id != segment_id:
            raise ValueError("校订建议不属于指定片段")
        segment = self.get_segment(segment_id)
        if segment is None:
            raise KeyError(f"转写片段不存在：{segment_id}")
        corrected = apply_confirmed_corrections(segment.effective_text, (suggestion,))
        return self.update_corrected_text(
            segment_id,
            corrected,
            edit_type="terminology",
            reason=suggestion.reason,
            actor=actor,
            metadata={"suggestion": suggestion.to_dict()},
        )

    def get_speaker(self, run_id: str, speaker_id: str) -> TranscriptSpeaker | None:
        row = self.connection.execute(
            "SELECT * FROM transcript_speakers WHERE run_id=? AND speaker_id=?",
            (_clean(run_id), _clean(speaker_id)),
        ).fetchone()
        return self._speaker_from_row(row) if row is not None else None

    def create_speaker(
        self,
        run_id: str,
        speaker_id: str,
        *,
        display_name: str | None = None,
        name_source: str = "automatic",
        metadata: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> TranscriptSpeaker:
        run = self._require_run(run_id)
        clean_id = _clean(speaker_id, required=True, field_name="说话人 ID")
        clean_source = _clean(name_source) or "automatic"
        if clean_source not in SPEAKER_NAME_SOURCES:
            raise ValueError(f"不支持的说话人名称来源：{clean_source}")
        speaker = TranscriptSpeaker(
            clean_id, _clean(display_name) or None, clean_source, dict(metadata or {})
        )
        now = utcnow_iso()
        try:
            with self.connection:
                self._insert_speaker(run.id, speaker, now)
                self._insert_edit(
                    run_id=run.id, segment_id=None, target_type="speaker_record",
                    target_id=clean_id, before_text="", after_text=_dump(speaker.to_dict(), default={}),
                    edit_type="speaker_create", reason="新增说话人标签", actor=actor,
                    created_at=now,
                )
                self._touch_run(run.id, now)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法创建说话人：{exc}") from exc
        result = self.get_speaker(run.id, clean_id)
        assert result is not None
        return result

    def update_speaker(
        self,
        run_id: str,
        speaker_id: str,
        *,
        display_name: str | None | object = _UNSET,
        metadata: dict[str, Any] | None = None,
        actor: str = "user",
        reason: str = "更新说话人信息",
    ) -> TranscriptSpeaker:
        row = self._require_speaker_row(run_id, speaker_id)
        current = self._speaker_from_row(row)
        next_name = current.display_name if display_name is _UNSET else (_clean(display_name) or None)
        next_source = current.name_source
        if display_name is not _UNSET and next_name != current.display_name:
            # A user-supplied identity is never persisted as an automatic claim.
            next_source = "manual"
        next_metadata = current.metadata if metadata is None else dict(metadata)
        updated = TranscriptSpeaker(current.id, next_name, next_source, next_metadata)
        if updated == current:
            return current
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE transcript_speakers SET display_name=?, name_source=?,
                       metadata_json=?, updated_at=? WHERE run_id=? AND speaker_id=?""",
                (updated.display_name, updated.name_source, _dump(updated.metadata, default={}),
                 now, row["run_id"], row["speaker_id"]),
            )
            self._insert_edit(
                run_id=row["run_id"], segment_id=None, target_type="speaker_record",
                target_id=row["speaker_id"], before_text=_dump(current.to_dict(), default={}),
                after_text=_dump(updated.to_dict(), default={}), edit_type="speaker_update",
                reason=reason, actor=actor, created_at=now,
            )
            self._touch_run(row["run_id"], now)
        result = self.get_speaker(row["run_id"], row["speaker_id"])
        assert result is not None
        return result

    def delete_speaker(
        self,
        run_id: str,
        speaker_id: str,
        *,
        actor: str = "user",
        reason: str = "删除未使用的说话人标签",
    ) -> bool:
        row = self.connection.execute(
            "SELECT * FROM transcript_speakers WHERE run_id=? AND speaker_id=?",
            (_clean(run_id), _clean(speaker_id)),
        ).fetchone()
        if row is None:
            return False
        referenced = self.connection.execute(
            """SELECT COUNT(*) FROM transcript_segments
               WHERE run_id=? AND speaker_id=?""",
            (row["run_id"], row["speaker_id"]),
        ).fetchone()[0]
        if int(referenced) > 0:
            raise ValueError("说话人仍有关联片段，请先重分配或合并")
        speaker = self._speaker_from_row(row)
        now = utcnow_iso()
        with self.connection:
            self._insert_edit(
                run_id=row["run_id"], segment_id=None, target_type="speaker_record",
                target_id=row["speaker_id"], before_text=_dump(speaker.to_dict(), default={}),
                after_text="", edit_type="speaker_delete", reason=reason,
                actor=actor, created_at=now,
            )
            self.connection.execute(
                "DELETE FROM transcript_speakers WHERE run_id=? AND speaker_id=?",
                (row["run_id"], row["speaker_id"]),
            )
            self._touch_run(row["run_id"], now)
        return True

    def list_speakers(self, run_id: str, *, include_merged: bool = True) -> list[TranscriptSpeaker]:
        rows = self.connection.execute(
            "SELECT * FROM transcript_speakers WHERE run_id=? ORDER BY speaker_id",
            (_clean(run_id),),
        ).fetchall()
        speakers = [self._speaker_from_row(row) for row in rows]
        if not include_merged:
            speakers = [speaker for speaker in speakers if not speaker.metadata.get("merged_into")]
        return speakers

    def rename_speaker(
        self,
        run_id: str,
        speaker_id: str,
        display_name: str,
        *,
        reason: str = "人工确认说话人名称",
        actor: str = "user",
    ) -> TranscriptSpeaker:
        clean_name = _clean(display_name, required=True, field_name="说话人名称")
        row = self._require_speaker_row(run_id, speaker_id)
        before = str(row["display_name"] or "")
        if before == clean_name and row["name_source"] == "manual":
            return self._speaker_from_row(row)
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                """UPDATE transcript_speakers
                   SET display_name=?, name_source='manual', updated_at=?
                   WHERE run_id=? AND speaker_id=?""",
                (clean_name, now, row["run_id"], row["speaker_id"]),
            )
            self._insert_edit(
                run_id=row["run_id"], segment_id=None, target_type="speaker_name",
                target_id=row["speaker_id"], before_text=before, after_text=clean_name,
                edit_type="speaker_rename", reason=reason, actor=actor, created_at=now,
            )
            self._touch_run(row["run_id"], now)
        updated = self.get_speaker(row["run_id"], row["speaker_id"])
        assert updated is not None
        return updated

    def reassign_segment(
        self,
        segment_id: str,
        speaker_id: str,
        *,
        reason: str = "人工调整说话人归属",
        actor: str = "user",
    ) -> TranscriptSegment:
        row = self._require_segment_row(segment_id)
        speaker = self._require_speaker_row(row["run_id"], speaker_id)
        before = str(row["speaker_id"] or "")
        after = str(speaker["speaker_id"])
        if before == after:
            return self._segment_from_row(row)
        now = utcnow_iso()
        with self.connection:
            self.connection.execute(
                "UPDATE transcript_segments SET speaker_id=?, updated_at=? WHERE id=?",
                (after, now, row["id"]),
            )
            self._insert_edit(
                run_id=row["run_id"], segment_id=row["id"],
                target_type="speaker_assignment", target_id=row["id"],
                before_text=before, after_text=after, edit_type="speaker_assignment",
                reason=reason, actor=actor, created_at=now,
            )
            self._touch_run(row["run_id"], now)
        updated = self.get_segment(row["id"])
        assert updated is not None
        return updated

    def merge_speakers(
        self,
        run_id: str,
        source_speaker_id: str,
        target_speaker_id: str,
        *,
        reason: str = "人工合并说话人",
        actor: str = "user",
    ) -> int:
        if _clean(source_speaker_id) == _clean(target_speaker_id):
            raise ValueError("不能把说话人合并到自身")
        source = self._require_speaker_row(run_id, source_speaker_id)
        target = self._require_speaker_row(run_id, target_speaker_id)
        source_metadata = _load_mapping(source["metadata_json"])
        if source_metadata.get("merged_into"):
            raise ValueError("源说话人已经被合并")
        now = utcnow_iso()
        segment_ids = [
            str(row["id"])
            for row in self.connection.execute(
                "SELECT id FROM transcript_segments WHERE run_id=? AND speaker_id=? ORDER BY ordinal",
                (source["run_id"], source["speaker_id"]),
            ).fetchall()
        ]
        source_metadata.update({"merged_into": target["speaker_id"], "merged_at": now})
        with self.connection:
            cursor = self.connection.execute(
                """UPDATE transcript_segments SET speaker_id=?, updated_at=?
                   WHERE run_id=? AND speaker_id=?""",
                (target["speaker_id"], now, source["run_id"], source["speaker_id"]),
            )
            self.connection.execute(
                """UPDATE transcript_speakers SET metadata_json=?, updated_at=?
                   WHERE run_id=? AND speaker_id=?""",
                (_dump(source_metadata, default={}), now, source["run_id"], source["speaker_id"]),
            )
            self._insert_edit(
                run_id=source["run_id"], segment_id=None, target_type="speaker_merge",
                target_id=source["speaker_id"], before_text=source["speaker_id"],
                after_text=target["speaker_id"], edit_type="speaker_merge",
                reason=reason, actor=actor,
                metadata={"affected_segment_ids": segment_ids}, created_at=now,
            )
            self._touch_run(source["run_id"], now)
        return max(0, cursor.rowcount)

    def list_edits(
        self,
        *,
        run_id: str | None = None,
        segment_id: str | None = None,
        limit: int = 1000,
    ) -> list[TranscriptEdit]:
        clauses: list[str] = []
        params: list[object] = []
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if segment_id:
            clauses.append("segment_id=?")
            params.append(segment_id)
        query = "SELECT * FROM transcript_edits"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, id LIMIT ?"
        params.append(min(10000, max(1, int(limit))))
        return [self._edit_from_row(row) for row in self.connection.execute(query, params).fetchall()]

    def create_glossary(
        self,
        name: str,
        *,
        scope: str = "global",
        scope_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        glossary_id: str | None = None,
    ) -> Glossary:
        clean_name = _clean(name, required=True, field_name="术语库名称")
        clean_scope = _clean(scope)
        if clean_scope not in GLOSSARY_SCOPES:
            raise ValueError(f"不支持的术语库范围：{clean_scope}")
        clean_scope_id = "" if clean_scope == "global" else _clean(
            scope_id, required=True, field_name="术语库范围 ID"
        )
        clean_id = _clean(glossary_id or f"glossary-{uuid4().hex}")
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO asr_glossaries(
                           id, name, scope, scope_id, enabled, metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (clean_id, clean_name, clean_scope, clean_scope_id, 1 if enabled else 0,
                     _dump(metadata, default={}), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法创建术语库：{exc}") from exc
        return self._require_glossary(clean_id)

    def get_glossary(self, glossary_id: str) -> Glossary | None:
        row = self.connection.execute(
            "SELECT * FROM asr_glossaries WHERE id=?", (_clean(glossary_id),)
        ).fetchone()
        return self._glossary_from_row(row) if row is not None else None

    def list_glossaries(
        self,
        *,
        scope: str | None = None,
        scope_id: str | None = None,
        enabled_only: bool = False,
    ) -> list[Glossary]:
        clauses: list[str] = []
        params: list[object] = []
        if scope:
            if scope not in GLOSSARY_SCOPES:
                raise ValueError(f"不支持的术语库范围：{scope}")
            clauses.append("scope=?")
            params.append(scope)
        if scope_id is not None:
            clauses.append("scope_id=?")
            params.append(scope_id)
        if enabled_only:
            clauses.append("enabled=1")
        query = "SELECT * FROM asr_glossaries"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY CASE scope WHEN 'global' THEN 0 WHEN 'knowledge_space' THEN 1 ELSE 2 END, name, id"
        return [self._glossary_from_row(row) for row in self.connection.execute(query, params).fetchall()]

    def update_glossary(
        self,
        glossary_id: str,
        *,
        name: str | None = None,
        enabled: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Glossary:
        current = self._require_glossary(glossary_id)
        next_name = current.name if name is None else _clean(name, required=True, field_name="术语库名称")
        next_enabled = current.enabled if enabled is None else bool(enabled)
        next_metadata = current.metadata if metadata is None else metadata
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """UPDATE asr_glossaries
                       SET name=?, enabled=?, metadata_json=?, updated_at=? WHERE id=?""",
                    (next_name, 1 if next_enabled else 0, _dump(next_metadata, default={}), now, current.id),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法更新术语库：{exc}") from exc
        return self._require_glossary(current.id)

    def delete_glossary(self, glossary_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM asr_glossaries WHERE id=?", (_clean(glossary_id),)
            )
        return cursor.rowcount > 0

    def add_glossary_term(
        self,
        glossary_id: str,
        term: GlossaryTermInput | str,
        *,
        term_id: str | None = None,
    ) -> GlossaryTerm:
        self._require_glossary(glossary_id)
        value = term if isinstance(term, GlossaryTermInput) else GlossaryTermInput(str(term))
        canonical = _clean(value.canonical_term, required=True, field_name="标准术语")
        variants = self._unique_texts(value.variants, excluding=canonical)
        clean_id = _clean(term_id or f"term-{uuid4().hex}")
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """INSERT INTO asr_glossary_terms(
                           id, glossary_id, canonical_term, normalized_term, variants_json,
                           notes, metadata_json, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (clean_id, glossary_id, canonical, _normalize(canonical),
                     _dump(list(variants), default=[]), _clean(value.notes),
                     _dump(value.metadata, default={}), now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法添加术语：{exc}") from exc
        result = self.get_glossary_term(clean_id)
        assert result is not None
        return result

    def get_glossary_term(self, term_id: str) -> GlossaryTerm | None:
        row = self.connection.execute(
            "SELECT * FROM asr_glossary_terms WHERE id=?", (_clean(term_id),)
        ).fetchone()
        return self._term_from_row(row) if row is not None else None

    def list_glossary_terms(self, glossary_id: str) -> list[GlossaryTerm]:
        return [
            self._term_from_row(row)
            for row in self.connection.execute(
                """SELECT * FROM asr_glossary_terms
                   WHERE glossary_id=? ORDER BY normalized_term, id""",
                (_clean(glossary_id),),
            ).fetchall()
        ]

    def update_glossary_term(
        self,
        term_id: str,
        *,
        canonical_term: str | None = None,
        variants: Sequence[str] | None = None,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GlossaryTerm:
        current = self.get_glossary_term(term_id)
        if current is None:
            raise KeyError(f"术语不存在：{term_id}")
        canonical = current.canonical_term if canonical_term is None else _clean(
            canonical_term, required=True, field_name="标准术语"
        )
        next_variants = current.variants if variants is None else self._unique_texts(variants, excluding=canonical)
        next_notes = current.notes if notes is None else _clean(notes)
        next_metadata = current.metadata if metadata is None else metadata
        now = utcnow_iso()
        try:
            with self.connection:
                self.connection.execute(
                    """UPDATE asr_glossary_terms SET canonical_term=?, normalized_term=?,
                           variants_json=?, notes=?, metadata_json=?, updated_at=? WHERE id=?""",
                    (canonical, _normalize(canonical), _dump(list(next_variants), default=[]),
                     next_notes, _dump(next_metadata, default={}), now, current.id),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"无法更新术语：{exc}") from exc
        result = self.get_glossary_term(current.id)
        assert result is not None
        return result

    def delete_glossary_term(self, term_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM asr_glossary_terms WHERE id=?", (_clean(term_id),)
            )
        return cursor.rowcount > 0

    def context_terms(
        self,
        *,
        knowledge_space_id: str | None = None,
        source_id: str | None = None,
    ) -> tuple[str, ...]:
        clauses = ["g.enabled=1", "(g.scope='global'"]
        params: list[object] = []
        if knowledge_space_id:
            clauses[-1] += " OR (g.scope='knowledge_space' AND g.scope_id=?)"
            params.append(knowledge_space_id)
        if source_id:
            clauses[-1] += " OR (g.scope='source' AND g.scope_id=?)"
            params.append(source_id)
        clauses[-1] += ")"
        rows = self.connection.execute(
            """SELECT t.canonical_term FROM asr_glossary_terms t
               JOIN asr_glossaries g ON g.id=t.glossary_id
               WHERE """ + " AND ".join(clauses) + """
               ORDER BY CASE g.scope WHEN 'global' THEN 0
                                     WHEN 'knowledge_space' THEN 1 ELSE 2 END,
                        g.name, t.normalized_term, t.id""",
            params,
        ).fetchall()
        result: list[str] = []
        seen: set[str] = set()
        for row in rows:
            term = str(row["canonical_term"])
            normalized = _normalize(term)
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(term)
        return tuple(result)

    def _validate_transcript(self, transcript: TranscriptV2) -> None:
        _clean(transcript.run.id, required=True, field_name="ASR run ID")
        _clean(transcript.run.profile, required=True, field_name="转写配置")
        _clean(transcript.run.provider, required=True, field_name="ASR Provider")
        _clean(transcript.run.model, required=True, field_name="ASR 模型")
        _clean(transcript.source.sha256, required=True, field_name="源文件校验值")
        if transcript.source.duration_ms < 0:
            raise ValueError("源文件时长不能为负数")
        if transcript.quality.status not in QUALITY_STATUSES:
            raise ValueError(f"不支持的转写质量状态：{transcript.quality.status}")
        speaker_ids = [_clean(item.id, required=True, field_name="说话人 ID") for item in transcript.speakers]
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ValueError("说话人 ID 不能重复")
        segment_ids: set[str] = set()
        ordinals: set[int] = set()
        for segment in transcript.segments:
            if not _clean(segment.id):
                raise ValueError("转写片段 ID 不能为空")
            if segment.id in segment_ids or segment.ordinal in ordinals:
                raise ValueError("转写片段 ID 和 ordinal 必须唯一")
            if segment.ordinal < 0 or segment.start_ms < 0 or segment.end_ms <= segment.start_ms:
                raise ValueError(f"转写片段时间范围无效：{segment.id}")
            if segment.confidence is not None and not 0 <= segment.confidence <= 1:
                raise ValueError(f"转写片段置信度无效：{segment.id}")
            segment_ids.add(segment.id)
            ordinals.add(segment.ordinal)

    @staticmethod
    def _assert_same_raw_facts(current: TranscriptV2, incoming: TranscriptV2) -> None:
        current_raw = [(item.id, item.ordinal, item.start_ms, item.end_ms, item.raw_text) for item in current.segments]
        incoming_raw = [(item.id, item.ordinal, item.start_ms, item.end_ms, item.raw_text) for item in incoming.segments]
        if current.source.sha256 != incoming.source.sha256 or current_raw != incoming_raw:
            raise ValueError("同一 ASR run 已存在，禁止覆盖原始转写事实")

    def _insert_speaker(self, run_id: str, speaker: TranscriptSpeaker, now: str) -> None:
        source = speaker.name_source if speaker.name_source in SPEAKER_NAME_SOURCES else "automatic"
        self.connection.execute(
            """INSERT INTO transcript_speakers(
                   run_id, speaker_id, display_name, name_source, metadata_json, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, speaker.id, speaker.display_name, source,
             _dump(speaker.metadata, default={}), now, now),
        )

    def _insert_edit(
        self,
        *,
        run_id: str,
        segment_id: str | None,
        target_type: str,
        target_id: str,
        before_text: str,
        after_text: str,
        edit_type: str,
        reason: str,
        actor: str,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        confirmed: bool = True,
    ) -> None:
        self.connection.execute(
            """INSERT INTO transcript_edits(
                   id, run_id, segment_id, target_type, target_id, before_text,
                   after_text, edit_type, confirmed, reason, actor, metadata_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f"edit-{uuid4().hex}", run_id, segment_id, target_type, target_id,
             before_text, after_text, _clean(edit_type) or target_type, 1 if confirmed else 0,
             _clean(reason),
             _clean(actor) or "user", _dump(metadata, default={}), created_at or utcnow_iso()),
        )

    def _touch_run(self, run_id: str, now: str) -> None:
        self.connection.execute(
            "UPDATE transcription_runs SET updated_at=? WHERE id=?", (now, run_id)
        )

    def _require_run(self, run_id: str) -> TranscriptionRunRecord:
        result = self.get_run(run_id)
        if result is None:
            raise KeyError(f"转写任务不存在：{run_id}")
        return result

    def _require_segment_row(self, segment_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM transcript_segments WHERE id=?", (_clean(segment_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"转写片段不存在：{segment_id}")
        return row

    def _require_speaker_row(self, run_id: str, speaker_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM transcript_speakers WHERE run_id=? AND speaker_id=?",
            (_clean(run_id), _clean(speaker_id)),
        ).fetchone()
        if row is None:
            raise KeyError(f"说话人不存在：{run_id}/{speaker_id}")
        return row

    def _require_glossary(self, glossary_id: str) -> Glossary:
        result = self.get_glossary(glossary_id)
        if result is None:
            raise KeyError(f"术语库不存在：{glossary_id}")
        return result

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> TranscriptionRunRecord:
        return TranscriptionRunRecord(
            id=str(row["id"]), document_id=row["document_id"],
            source_name=str(row["source_name"]), source_checksum=str(row["source_checksum"]),
            source_duration_ms=int(row["source_duration_ms"]), profile=str(row["profile"]),
            provider=str(row["provider"]), model=str(row["model"]), language=row["language"],
            status=str(row["status"]), config=_load_mapping(row["config_json"]),
            quality=_load_mapping(row["quality_json"]), transcript_path=row["transcript_path"],
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _speaker_from_row(row: sqlite3.Row) -> TranscriptSpeaker:
        return TranscriptSpeaker(
            str(row["speaker_id"]), row["display_name"], str(row["name_source"]),
            _load_mapping(row["metadata_json"]),
        )

    @staticmethod
    def _segment_from_row(row: sqlite3.Row) -> TranscriptSegment:
        words = tuple(
            TranscriptWord.from_dict(item)
            for item in _load_sequence(row["words_json"])
            if isinstance(item, Mapping)
        )
        return TranscriptSegment(
            id=str(row["id"]), ordinal=int(row["ordinal"]),
            start_ms=int(row["start_ms"]), end_ms=int(row["end_ms"]),
            speaker_id=row["speaker_id"], raw_text=str(row["raw_text"]),
            corrected_text=row["corrected_text"], confidence=row["confidence"],
            flags=tuple(str(item) for item in _load_sequence(row["flags_json"])),
            words=words, metadata=_load_mapping(row["metadata_json"]),
        )

    @staticmethod
    def _edit_from_row(row: sqlite3.Row) -> TranscriptEdit:
        return TranscriptEdit(
            id=str(row["id"]), run_id=str(row["run_id"]), segment_id=row["segment_id"],
            target_type=str(row["target_type"]), target_id=str(row["target_id"]),
            before_text=str(row["before_text"]), after_text=str(row["after_text"]),
            edit_type=str(row["edit_type"]), confirmed=bool(row["confirmed"]),
            reason=str(row["reason"]), actor=str(row["actor"]),
            metadata=_load_mapping(row["metadata_json"]), created_at=str(row["created_at"]),
        )

    @staticmethod
    def _glossary_from_row(row: sqlite3.Row) -> Glossary:
        return Glossary(
            id=str(row["id"]), name=str(row["name"]), scope=str(row["scope"]),
            scope_id=str(row["scope_id"]), enabled=bool(row["enabled"]),
            metadata=_load_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _term_from_row(row: sqlite3.Row) -> GlossaryTerm:
        return GlossaryTerm(
            id=str(row["id"]), glossary_id=str(row["glossary_id"]),
            canonical_term=str(row["canonical_term"]),
            variants=tuple(str(item) for item in _load_sequence(row["variants_json"])),
            notes=str(row["notes"]), metadata=_load_mapping(row["metadata_json"]),
            created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _unique_texts(values: Iterable[str], *, excluding: str = "") -> tuple[str, ...]:
        result: list[str] = []
        seen = {_normalize(excluding)} if excluding else set()
        for value in values:
            clean = _clean(value)
            normalized = _normalize(clean)
            if not clean or normalized in seen:
                continue
            seen.add(normalized)
            result.append(clean)
        return tuple(result)


__all__ = [
    "GLOSSARY_SCOPES",
    "RUN_STATUSES",
    "Glossary",
    "GlossaryTerm",
    "GlossaryTermInput",
    "TranscriptEdit",
    "TranscriptRepository",
    "TranscriptionRunRecord",
]
