from __future__ import annotations

import hashlib
import html
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlsplit, urlunsplit

from .deep_repository import DeepCorrectionRepository
from .repository import TranscriptRepository


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MARKDOWN_META_RE = re.compile(r"([\\`*_{}\[\]()#+.!|>~-])")


@dataclass(frozen=True, slots=True)
class MarkdownExportResult:
    path: Path
    sha256: str
    bytes_written: int


def safe_output_path(
    path: str | Path,
    *,
    suffixes: Sequence[str],
    allowed_root: str | Path | None = None,
) -> Path:
    raw = os.fspath(path)
    if not raw or "\x00" in raw:
        raise ValueError("导出路径无效")
    target = Path(raw).expanduser()
    if target.name in {"", ".", ".."}:
        raise ValueError("导出文件名无效")
    if target.suffix.casefold() not in {item.casefold() for item in suffixes}:
        raise ValueError("导出文件扩展名不受支持")
    resolved = target.resolve(strict=False)
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("导出路径超出允许目录") from exc
    if target.is_symlink():
        raise ValueError("拒绝写入符号链接")
    return target


def atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            if path.is_symlink():
                raise ValueError("拒绝覆盖符号链接")
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(f"导出文件已存在：{path}") from None
            temporary.unlink(missing_ok=True)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _md(value: object) -> str:
    clean = _CONTROL_RE.sub("", str(value or "")).replace("\r\n", "\n").replace("\r", "\n")
    escaped = html.escape(clean, quote=False)
    escaped = _MARKDOWN_META_RE.sub(r"\\\1", escaped)
    return escaped.replace("\n", "<br>")


def _table(value: object) -> str:
    return _md(value).replace("|", "\\|")


def _timestamp(milliseconds: int) -> str:
    value = max(0, int(milliseconds))
    hours, remainder = divmod(value, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _safe_http_url(value: object) -> str | None:
    raw = _CONTROL_RE.sub("", str(value or "").strip())
    if not raw or len(raw) > 4096:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return None
    if parsed.username or parsed.password:
        return None
    path = quote(parsed.path, safe="/%:@!$&'+,;=~%")
    query = quote(parsed.query, safe="=&%:@!$'+,;/?~")
    fragment = quote(parsed.fragment, safe="=&%:@!$'+,;/?~")
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc, path, query, fragment))


def _mermaid_label(value: object) -> str:
    clean = _CONTROL_RE.sub("", str(value or "")).replace("\r", " ").replace("\n", " ")
    clean = re.sub(r"[`\"{}\[\]<>|%]", "'", clean)
    return html.escape(clean[:240], quote=False)


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


class DeepCorrectionMarkdownExporter:
    def __init__(self, repository: DeepCorrectionRepository):
        self.repository = repository

    def export(
        self,
        run_id: str,
        path: str | Path,
        *,
        overwrite: bool = False,
        allowed_root: str | Path | None = None,
    ) -> MarkdownExportResult:
        run = self.repository.get_run(run_id)
        if run is None:
            raise KeyError(f"深度精校任务不存在：{run_id}")
        if run.status != "completed":
            raise ValueError("只有已完成的精校任务可以导出")
        transcript = TranscriptRepository(self.repository.database).get_transcript(run.transcript_run_id)
        if transcript is None:
            raise ValueError("精校任务关联的原转写不存在")
        target = safe_output_path(path, suffixes=(".md",), allowed_root=allowed_root)
        if target.exists() and not overwrite:
            raise FileExistsError(f"导出文件已存在：{target}")
        snapshot = self.repository.snapshot(run.id)
        markdown = self._render(snapshot, transcript)
        payload = markdown.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        atomic_write_bytes(target, payload, overwrite=overwrite)
        self.repository.set_export_info(run.id, str(target), digest)
        return MarkdownExportResult(target, digest, len(payload))

    def _render(self, snapshot: Mapping[str, Any], transcript: object) -> str:
        run = snapshot["run"]
        paragraphs = _sequence(snapshot.get("paragraphs"))
        changes = _sequence(snapshot.get("changes"))
        evidence = _sequence(snapshot.get("evidence"))
        events = snapshot.get("change_events")
        events = events if isinstance(events, Mapping) else {}
        result = run.get("result") if isinstance(run, Mapping) else {}
        result = result if isinstance(result, Mapping) else {}
        source = transcript.source
        speaker_map = {item.id: (item.display_name or item.id) for item in transcript.speakers}
        lines: list[str] = [f"# {_md(source.name)} 深度精校稿", ""]

        lines.extend([
            "## 处理元数据", "",
            f"- 原转写任务：`{_md(run.get('transcript_run_id'))}`",
            f"- 精校任务：`{_md(run.get('id'))}`",
            f"- 精校模型：{_md(run.get('provider'))} / {_md(run.get('model'))}",
            f"- 源文件 SHA-256：`{_md(source.sha256)}`",
            f"- 源媒体时长：{_timestamp(source.duration_ms)}",
            f"- 结果校验摘要：`{_md(run.get('result_checksum') or '—')}`",
            "",
            "## 处理边界", "",
            "- 原始 ASR 文本作为不可变证据保存，本导出不覆盖原始稿。",
            "- 精校正文是可审核派生稿；每项修改的待确认/已接受/已拒绝状态以下方差异审计表为准。",
            "- 未确认身份、术语和外部事实保持不确定标记。",
        ])
        for boundary in _sequence(result.get("processing_boundaries")):
            lines.append(f"- {_md(boundary)}")

        lines.extend(["", "## 说话人", "", "| 匿名 ID | 显示名称 | 名称来源 |", "|---|---|---|"])
        for speaker in transcript.speakers:
            lines.append(
                f"| {_table(speaker.id)} | {_table(speaker.display_name or '未确认')} | {_table(speaker.name_source)} |"
            )
        if not transcript.speakers:
            lines.append("| — | 未进行说话人区分 | — |")

        lines.extend([
            "", "## 术语校正表", "",
            "| 原文 | 建议/校订 | 状态 | 理由 |", "|---|---|---|---|",
        ])
        terminology = [item for item in changes if isinstance(item, Mapping) and item.get("change_type") == "terminology"]
        for item in terminology:
            lines.append(
                f"| {_table(item.get('before_text'))} | {_table(item.get('after_text'))} | "
                f"{_table(item.get('status'))} | {_table(item.get('reason'))} |"
            )
        if not terminology:
            lines.append("| — | — | 无术语校正记录 | — |")

        lines.extend(["", "## 完整精校正文", ""])
        changes_by_paragraph: dict[str, list[Mapping[str, Any]]] = {}
        for change in changes:
            if not isinstance(change, Mapping):
                continue
            paragraph_id = str(change.get("paragraph_id") or "")
            if paragraph_id:
                changes_by_paragraph.setdefault(paragraph_id, []).append(change)
        for paragraph in paragraphs:
            if not isinstance(paragraph, Mapping):
                continue
            speaker_id = str(paragraph.get("speaker_id") or "speaker_unknown")
            speaker = speaker_map.get(speaker_id, speaker_id)
            rendered_text = str(paragraph.get("corrected_text") or "")
            # Paragraphs retain the model proposal as immutable audit facts.
            # A rejected item must nevertheless disappear from a later export;
            # restore its exact pre-change text without mutating stored facts.
            for change in changes_by_paragraph.get(str(paragraph.get("id") or ""), []):
                if str(change.get("status") or "") != "rejected":
                    continue
                before = str(change.get("before_text") or "")
                after = str(change.get("after_text") or "")
                if after and after in rendered_text:
                    rendered_text = rendered_text.replace(after, before, 1)
            lines.extend([
                f"### [{_timestamp(int(paragraph.get('start_ms', 0)))} → "
                f"{_timestamp(int(paragraph.get('end_ms', 0)))}] {_md(speaker)}",
                "",
                _md(rendered_text),
                "",
            ])
        if not paragraphs:
            lines.append("无可导出的精校正文。")

        segment_lookup = {item.id: item for item in transcript.segments}
        evidence_counts: dict[str, int] = {}
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            change_id = str(item.get("change_id") or "")
            if change_id:
                evidence_counts[change_id] = evidence_counts.get(change_id, 0) + 1
        status_labels = {
            "proposed": "待确认",
            "accepted": "已接受",
            "rejected": "已拒绝",
        }
        lines.extend([
            "", "## 原稿 / 精校稿差异审计", "",
            "| 时间 | 说话人 | 原稿 | 精校建议 | 状态 | 置信度 | 证据 | 理由 |",
            "|---|---|---|---|---|---:|---:|---|",
        ])
        for item in changes:
            if not isinstance(item, Mapping):
                continue
            source_ids = _sequence(item.get("source_segment_ids"))
            segment = segment_lookup.get(str(source_ids[0])) if source_ids else None
            time_text = (
                f"{_timestamp(segment.start_ms)}–{_timestamp(segment.end_ms)}"
                if segment is not None else "—"
            )
            speaker_id = str(segment.speaker_id or "") if segment is not None else ""
            speaker = speaker_map.get(speaker_id, speaker_id or "未确认")
            raw_confidence = item.get("confidence")
            try:
                confidence = f"{max(0.0, min(1.0, float(raw_confidence))):.0%}"
            except (TypeError, ValueError, OverflowError):
                confidence = "—"
            change_id = str(item.get("id") or "")
            lines.append(
                f"| {_table(time_text)} | {_table(speaker)} | "
                f"{_table(item.get('before_text'))} | {_table(item.get('after_text'))} | "
                f"{_table(status_labels.get(str(item.get('status') or ''), item.get('status') or '—'))} | "
                f"{_table(confidence)} | {evidence_counts.get(change_id, 0)} | "
                f"{_table(item.get('reason'))} |"
            )
        if not changes:
            lines.append("| — | — | — | 无修改建议 | — | — | 0 | — |")

        uncertain = _sequence(result.get("uncertain_items"))
        uncertain.extend(
            f"{item.get('before_text')} → {item.get('after_text')}（{item.get('status')}）"
            for item in changes
            if isinstance(item, Mapping) and item.get("status") != "accepted"
        )
        lines.extend(["", "## 不确定项", ""])
        if uncertain:
            lines.extend(f"- {_md(item)}" for item in uncertain)
        else:
            lines.append("- 未记录额外不确定项。")

        lines.extend(["", "## 外部证据", ""])
        external = [item for item in evidence if isinstance(item, Mapping) and item.get("evidence_type") == "external"]
        for item in external:
            label = _md(item.get("title") or "外部证据")
            url = _safe_http_url(item.get("url"))
            summary = _md(item.get("summary"))
            if url:
                lines.append(f"- [{label}]({url})" + (f" — {summary}" if summary else ""))
            else:
                lines.append(f"- {label} — 链接因安全策略未导出" + (f"；{summary}" if summary else ""))
        if not external:
            lines.append("- 未记录外部证据。")

        cards = [item for item in _sequence(result.get("knowledge_cards")) if isinstance(item, Mapping)]
        lines.extend(["", "## 知识卡", ""])
        for card in cards:
            lines.extend([
                f"### {_md(card.get('title') or '未命名知识卡')}", "",
                _md(card.get("summary") or card.get("content")), "",
            ])
            tags = _sequence(card.get("tags"))
            if tags:
                lines.append("标签：" + "、".join(_md(item) for item in tags))
                lines.append("")
        if not cards:
            lines.append("未生成知识卡。")

        if result.get("include_mermaid", True):
            lines.extend(["", "## 知识关系图", "", "```mermaid", "flowchart TD"])
            card_ids: dict[str, str] = {}
            for index, card in enumerate(cards):
                title = str(card.get("title") or f"知识卡 {index + 1}")
                node_id = f"N{index}"
                card_ids[title] = node_id
                label = _mermaid_label(title)
                lines.append(f'    {node_id}["{label}"]')
            for relation in _sequence(result.get("relations")):
                if not isinstance(relation, Mapping):
                    continue
                source_id = card_ids.get(str(relation.get("source") or ""))
                target_id = card_ids.get(str(relation.get("target") or ""))
                if not source_id or not target_id:
                    continue
                label = _mermaid_label(relation.get("label") or "关联")
                lines.append(f"    {source_id} -->|{label}| {target_id}")
            if not cards:
                lines.append('    N0["暂无知识关系"]')
            lines.append("```")
        lines.extend(["", "## 审计摘要", ""])
        status_counts = {
            status: sum(1 for item in changes if isinstance(item, Mapping) and item.get("status") == status)
            for status in ("proposed", "accepted", "rejected")
        }
        event_count = sum(len(_sequence(value)) for value in events.values())
        lines.extend([
            f"- 修改建议总数：{len(changes)}",
            f"- 待审核：{status_counts['proposed']}",
            f"- 已接受：{status_counts['accepted']}",
            f"- 已拒绝：{status_counts['rejected']}",
            f"- 状态审计事件：{event_count}",
            f"- 导出正文段落：{len(paragraphs)}",
            "",
        ])
        return "\n".join(lines)


__all__ = [
    "DeepCorrectionMarkdownExporter",
    "MarkdownExportResult",
    "atomic_write_bytes",
    "safe_output_path",
]
