from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Iterable

from ..models import (
    ContentSegment,
    KnowledgeChunk,
    KnowledgeDocument,
    SourceReference,
    estimate_tokens,
    sha256_text,
)


class MediaAwareChunker:
    def __init__(self, target_tokens: int = 320, max_tokens: int = 480, max_time_span: float = 120.0):
        if target_tokens < 32 or max_tokens < target_tokens:
            raise ValueError("chunk token limits are invalid")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.max_time_span = max_time_span

    def chunk(self, document: KnowledgeDocument) -> list[KnowledgeChunk]:
        media_type = document.media_type.lower()
        if media_type in {"pdf"}:
            pieces = self._group_by_locator(document.segments, "page")
        elif media_type in {"presentation", "ppt", "pptx", "slides"}:
            pieces = self._group_by_locator(document.segments, "slide")
        elif media_type in {"audio", "video"}:
            pieces = self._temporal(document.segments)
        elif media_type == "image":
            pieces = self._image(document.segments)
        else:
            pieces = self._structured_text(document.segments)
        return [self._build_chunk(document, index, *piece) for index, piece in enumerate(pieces)]

    def _split_text(self, value: str) -> list[tuple[str, int, int]]:
        value = value.strip()
        if not value:
            return []
        raw_units = [unit.strip() for unit in re.split(r"(?<=[。！？.!?])\s+|\n{2,}", value) if unit.strip()]
        units: list[str] = []
        for unit in raw_units:
            if estimate_tokens(unit) <= self.max_tokens:
                units.append(unit)
                continue
            cursor = 0
            while cursor < len(unit):
                low, high = cursor + 1, len(unit)
                best = low
                while low <= high:
                    middle = (low + high) // 2
                    if estimate_tokens(unit[cursor:middle]) <= self.max_tokens:
                        best = middle
                        low = middle + 1
                    else:
                        high = middle - 1
                units.append(unit[cursor:best].strip())
                cursor = best

        result: list[tuple[str, int, int]] = []
        buffer: list[str] = []
        search_from = 0
        start = 0
        for unit in units:
            candidate = "\n\n".join([*buffer, unit])
            if buffer and estimate_tokens(candidate) > self.target_tokens:
                content = "\n\n".join(buffer)
                end = start + len(content)
                result.append((content, start, end))
                buffer = [unit]
                found = value.find(unit, max(search_from, end))
                start = found if found >= 0 else end
            else:
                if not buffer:
                    found = value.find(unit, search_from)
                    start = found if found >= 0 else search_from
                buffer.append(unit)
            search_from = start
        if buffer:
            content = "\n\n".join(buffer)
            result.append((content, start, start + len(content)))
        return result

    def _structured_text(
        self, segments: Iterable[ContentSegment]
    ) -> list[tuple[str, str, list[str], SourceReference | None, dict[str, Any]]]:
        pieces: list[tuple[str, str, list[str], SourceReference | None, dict[str, Any]]] = []
        for segment in sorted(segments, key=lambda item: (item.sequence, item.id)):
            text = segment.retrieval_text
            if not text:
                continue
            heading_stack = list(segment.heading_path)
            blocks: list[tuple[list[str], str, int, int]] = []
            if re.search(r"(?m)^#{1,6}\s+", text):
                current_lines: list[str] = []
                current_start = 0
                cursor = 0
                for line in text.splitlines(keepends=True):
                    match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
                    if match:
                        if current_lines:
                            body = "".join(current_lines).strip()
                            if body:
                                blocks.append((list(heading_stack), body, current_start, cursor))
                            current_lines = []
                        level = len(match.group(1))
                        heading_stack = heading_stack[: level - 1] + [match.group(2).strip()]
                        current_start = cursor + len(line)
                    else:
                        if not current_lines:
                            current_start = cursor
                        current_lines.append(line)
                    cursor += len(line)
                body = "".join(current_lines).strip()
                if body:
                    blocks.append((list(heading_stack), body, current_start, len(text)))
            else:
                blocks.append((heading_stack, text, 0, len(text)))
            for block_index, (path, body, block_start, _) in enumerate(blocks):
                for part_index, (part, start, end) in enumerate(self._split_text(body)):
                    key = f"segment:{segment.id}:block:{block_index}:part:{part_index}"
                    metadata = {"segment_ids": [segment.id], "modality": segment.modality}
                    reference = self._segment_reference(segment, text_start=block_start + start, text_end=block_start + end)
                    pieces.append((key, part, path, reference, metadata))
        return pieces

    def _group_by_locator(
        self, segments: Iterable[ContentSegment], locator: str
    ) -> list[tuple[str, str, list[str], SourceReference | None, dict[str, Any]]]:
        groups: dict[tuple[Any, Any], list[ContentSegment]] = {}
        for segment in sorted(segments, key=lambda item: (item.sequence, item.id)):
            value = segment.location.get(locator)
            section = segment.location.get("section")
            key = (value if value is not None else f"segment:{segment.id}", section)
            groups.setdefault(key, []).append(segment)
        pieces: list[tuple[str, str, list[str], SourceReference | None, dict[str, Any]]] = []
        for (location, section), members in groups.items():
            text = "\n\n".join(member.retrieval_text for member in members if member.retrieval_text)
            if not text:
                continue
            heading = [str(section)] if section else next((member.heading_path for member in members if member.heading_path), [])
            for part_index, (part, start, end) in enumerate(self._split_text(text)):
                stable_members = ",".join(member.id for member in members)
                key = f"{locator}:{location}:section:{section or '-'}:members:{stable_members}:part:{part_index}"
                first = members[0]
                reference = self._segment_reference(first, text_start=start, text_end=end)
                if locator == "page" and isinstance(location, int):
                    reference = replace(reference, page_number=location)
                if locator == "slide" and isinstance(location, int):
                    reference = replace(reference, slide_number=location)
                image_path = next((member.asset for member in members if member.asset), None)
                reference = replace(reference, image_path=image_path)
                pieces.append(
                    (
                        key,
                        part,
                        list(heading),
                        reference,
                        {"segment_ids": [member.id for member in members], "locator": locator, "locator_value": location},
                    )
                )
        return pieces

    def _temporal(
        self, segments: Iterable[ContentSegment]
    ) -> list[tuple[str, str, list[str], SourceReference | None, dict[str, Any]]]:
        ordered = [
            segment
            for segment in sorted(segments, key=lambda item: (item.sequence, item.id))
            if self._temporal_text(segment)
        ]
        groups: list[list[ContentSegment]] = []
        current: list[ContentSegment] = []
        for segment in ordered:
            start = segment.location.get("timestamp_start")
            previous_end = current[-1].location.get("timestamp_end") if current else None
            group_start = current[0].location.get("timestamp_start") if current else None
            current_speaker = next(
                (self._speaker_id(item) for item in reversed(current) if self._speaker_id(item)),
                None,
            )
            incoming_speaker = self._speaker_id(segment)
            speaker_changed = bool(
                current
                and incoming_speaker
                and current_speaker
                and incoming_speaker != current_speaker
            )
            quality_changed = bool(
                current
                and self._quality_status(segment) != self._quality_status(current[-1])
            )
            modality_changed = bool(
                current and segment.modality != current[-1].modality
            )
            section_changed = bool(
                current
                and segment.location.get("section")
                and segment.location.get("section") != current[-1].location.get("section")
            )
            gap = start - previous_end if isinstance(start, (int, float)) and isinstance(previous_end, (int, float)) else 0
            span = start - group_start if isinstance(start, (int, float)) and isinstance(group_start, (int, float)) else 0
            candidate_tokens = estimate_tokens("\n".join(self._temporal_text(item) for item in [*current, segment]))
            if current and (
                speaker_changed
                or quality_changed
                or modality_changed
                or gap > 8
                or span > self.max_time_span
                or section_changed
                or candidate_tokens > self.target_tokens
            ):
                groups.append(current)
                current = []
            current.append(segment)
        if current:
            groups.append(current)

        pieces = []
        for group_index, members in enumerate(groups):
            body = "\n".join(self._temporal_text(member) for member in members if self._temporal_text(member))
            starts = [member.location.get("timestamp_start") for member in members]
            ends = [member.location.get("timestamp_end") for member in members]
            start = next((float(value) for value in starts if isinstance(value, (int, float))), None)
            end = next((float(value) for value in reversed(ends) if isinstance(value, (int, float))), start)
            speaker_ids = list(
                dict.fromkeys(
                    speaker for member in members if (speaker := self._speaker_id(member))
                )
            )
            speaker_names = list(
                dict.fromkeys(
                    name for member in members if (name := self._speaker_name(member))
                )
            )
            speaker_label = speaker_names[0] if len(speaker_names) == 1 else (
                speaker_ids[0] if len(speaker_ids) == 1 else ""
            )
            # This is the text sent to lexical/vector indexing.  A human speaker
            # label improves “who said what” retrieval, while timestamps stay in
            # provenance metadata so repeated numbers do not pollute embeddings.
            text = f"{speaker_label}：{body}" if speaker_label else body
            heading = next((member.heading_path for member in members if member.heading_path), [])
            reference = replace(
                self._segment_reference(members[0]), timestamp_start=start, timestamp_end=end
            )
            member_key = ",".join(member.id for member in members)
            run_ids = list(
                dict.fromkeys(
                    run_id for member in members if (run_id := self._asr_run_id(member))
                )
            )
            quality_status = self._quality_status(members[0])
            quality_flags = list(
                dict.fromkeys(
                    str(flag)
                    for member in members
                    for flag in self._quality_flags(member)
                    if str(flag)
                )
            )
            display_label = speaker_label or "未知说话人"
            display_text = (
                f"[{self._display_timestamp(start)}] {display_label}\n{body}"
                if start is not None
                else f"{display_label}\n{body}"
            )
            pieces.append(
                (
                    f"time:{start}:{end}:speaker:{','.join(speaker_ids) or '-'}:quality:{quality_status}:members:{member_key}:group:{group_index}",
                    text,
                    list(heading),
                    reference,
                    {
                        "segment_ids": [member.id for member in members],
                        "time_span": [start, end],
                        "timestamp_start": start,
                        "timestamp_end": end,
                        "speaker_ids": speaker_ids,
                        "speaker_names": speaker_names,
                        "asr_run_ids": run_ids,
                        "asr_run_id": run_ids[0] if len(run_ids) == 1 else None,
                        "quality_status": quality_status,
                        "quality_flags": quality_flags,
                        "overlap": any(bool(member.metadata.get("overlap")) for member in members),
                        "display_text": display_text,
                    },
                )
            )
        return pieces

    @staticmethod
    def _speaker_id(segment: ContentSegment) -> str | None:
        value = segment.metadata.get("speaker_id") or segment.location.get("speaker_id")
        if value is None and segment.metadata.get("speaker"):
            value = segment.metadata.get("speaker")
        compact = str(value or "").strip()
        return compact or None

    @staticmethod
    def _speaker_name(segment: ContentSegment) -> str | None:
        value = (
            segment.metadata.get("speaker_name")
            or segment.metadata.get("display_name")
            or segment.location.get("speaker_name")
        )
        compact = str(value or "").strip()
        return compact or None

    @staticmethod
    def _quality_status(segment: ContentSegment) -> str:
        value: object = segment.metadata.get("quality_status", "pass")
        if isinstance(segment.metadata.get("quality"), dict):
            value = segment.metadata["quality"].get("status", value)  # type: ignore[union-attr]
        compact = str(value or "pass").strip().casefold()
        return compact if compact in {"pass", "review", "fail"} else "review"

    @staticmethod
    def _quality_flags(segment: ContentSegment) -> list[object]:
        value = segment.metadata.get("quality_flags", [])
        if isinstance(value, (tuple, list, set)):
            return list(value)
        return [value] if value else []

    @staticmethod
    def _asr_run_id(segment: ContentSegment) -> str | None:
        value = segment.metadata.get("asr_run_id") or segment.metadata.get("run_id")
        compact = str(value or "").strip()
        return compact or None

    @staticmethod
    def _temporal_text(segment: ContentSegment) -> str:
        # ContentSegment.retrieval_text historically appends an English
        # ``Speaker:`` line.  Speaker-aware chunks add one stable label at the
        # chunk boundary instead, avoiding repeated labels and timestamp noise.
        return "\n".join(
            value for value in (segment.text.strip(), segment.description.strip()) if value
        )

    @staticmethod
    def _display_timestamp(value: float | None) -> str:
        seconds = max(0, int(value or 0))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _image(
        self, segments: Iterable[ContentSegment]
    ) -> list[tuple[str, str, list[str], SourceReference | None, dict[str, Any]]]:
        members = [segment for segment in sorted(segments, key=lambda item: (item.sequence, item.id)) if segment.retrieval_text]
        if not members:
            return []
        text = "\n\n".join(member.retrieval_text for member in members)
        result = []
        for part_index, (part, start, end) in enumerate(self._split_text(text)):
            reference = replace(
                self._segment_reference(members[0], text_start=start, text_end=end),
                image_path=next((member.asset for member in members if member.asset), None),
            )
            result.append(
                (
                    f"image:members:{','.join(member.id for member in members)}:part:{part_index}",
                    part,
                    [],
                    reference,
                    {"segment_ids": [member.id for member in members], "modalities": [member.modality for member in members]},
                )
            )
        return result

    @staticmethod
    def _segment_reference(
        segment: ContentSegment, text_start: int | None = None, text_end: int | None = None
    ) -> SourceReference:
        location = segment.location
        return SourceReference(
            source_id="",
            media_type="",
            title="",
            page_number=location.get("page"),
            slide_number=location.get("slide"),
            timestamp_start=location.get("timestamp_start"),
            timestamp_end=location.get("timestamp_end"),
            section=location.get("section"),
            image_path=segment.asset,
            text_start=text_start,
            text_end=text_end,
        )

    def _build_chunk(
        self,
        document: KnowledgeDocument,
        ordinal: int,
        chunk_key: str,
        content: str,
        heading_path: list[str],
        partial_reference: SourceReference | None,
        metadata: dict[str, Any],
    ) -> KnowledgeChunk:
        document_id = document.document_id or f"doc-{sha256_text(document.source_id)[:20]}"
        chunk_id = f"chk-{sha256_text(document_id + '|' + chunk_key)[:24]}"
        partial = partial_reference or SourceReference(source_id="", media_type="", title="")
        base = document.source
        reference = replace(
            base,
            document_id=document_id,
            chunk_id=chunk_id,
            page_number=partial.page_number,
            slide_number=partial.slide_number,
            timestamp_start=partial.timestamp_start,
            timestamp_end=partial.timestamp_end,
            section=partial.section,
            image_path=partial.image_path,
            text_start=partial.text_start,
            text_end=partial.text_end,
        )
        return KnowledgeChunk(
            id=chunk_id,
            document_id=document_id,
            chunk_key=chunk_key,
            ordinal=ordinal,
            content=content,
            heading_path=heading_path,
            source_reference=reference,
            token_count=estimate_tokens(content),
            content_hash=sha256_text(content),
            metadata=metadata,
        )
