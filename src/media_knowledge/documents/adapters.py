from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import ContentSegment, KnowledgeDocument, SourceReference, sha256_text


def _structured_description(attributes: dict[str, Any]) -> str:
    if not attributes:
        return ""
    selected = {
        key: attributes[key]
        for key in ("rows", "latex", "visual_type", "slide_type", "language")
        if key in attributes
    }
    return json.dumps(selected, ensure_ascii=False, sort_keys=True) if selected else ""


def documents_from_ucb(bundle: dict[str, Any]) -> list[KnowledgeDocument]:
    """Adapt Knowledge Ingestor UCB 1.0 records without coupling to its scripts."""
    if bundle.get("schema_version") != "1.0":
        raise ValueError("unsupported UCB schema_version; expected '1.0'")
    source_records = bundle.get("sources")
    content_records = bundle.get("content")
    if not isinstance(source_records, list) or not isinstance(content_records, list):
        raise ValueError("UCB sources and content must be arrays")

    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in content_records:
        if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
            raise ValueError("every UCB content record needs a source_id")
        by_source.setdefault(item["source_id"], []).append(item)

    documents: list[KnowledgeDocument] = []
    for source in source_records:
        if not isinstance(source, dict):
            raise ValueError("every UCB source must be an object")
        source_id = str(source.get("id", "")).strip()
        title = str(source.get("title", "")).strip()
        media_type = str(source.get("kind", "document")).strip() or "document"
        if not source_id or not title:
            raise ValueError("every UCB source needs a non-empty id and title")
        origin = source.get("origin") if isinstance(source.get("origin"), dict) else {}
        source_metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        segments: list[ContentSegment] = []
        for index, item in enumerate(sorted(by_source.get(source_id, []), key=lambda row: (row.get("sequence", 0), row.get("id", "")))):
            location = item.get("location") if isinstance(item.get("location"), dict) else {}
            attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
            description = str(item.get("description") or "")
            if not item.get("text") and not description:
                description = _structured_description(attributes)
            section = location.get("section")
            heading_path = [str(section)] if section else []
            segments.append(
                ContentSegment(
                    id=str(item.get("id") or f"segment-{index + 1}"),
                    sequence=float(item.get("sequence", index)),
                    modality=str(item.get("modality") or "text"),
                    text=str(item.get("text") or ""),
                    description=description,
                    location=dict(location),
                    heading_path=heading_path,
                    asset=str(item["asset"]) if item.get("asset") else None,
                    metadata={
                        "confidence": item.get("confidence"),
                        "speaker": item.get("speaker"),
                        "language": item.get("language"),
                        "derived_from": item.get("derived_from", []),
                        "attributes": attributes,
                    },
                )
            )
        reference = SourceReference(
            source_id=source_id,
            media_type=media_type,
            title=title,
            original_uri=origin.get("uri"),
            local_path=origin.get("local_path"),
            obsidian_path=source_metadata.get("obsidian_path"),
            checksum=origin.get("sha256"),
        )
        documents.append(
            KnowledgeDocument(
                source_id=source_id,
                title=title,
                media_type=media_type,
                segments=segments,
                source=reference,
                collections=list(source_metadata.get("collections", [])),
                tags=list(source_metadata.get("tags", [])),
                metadata={
                    "bundle_id": bundle.get("bundle_id"),
                    "source_metadata": source_metadata,
                    "extraction": source.get("extraction", {}),
                },
            )
        )
    return documents


def document_from_text(
    text: str,
    *,
    title: str,
    source_id: str | None = None,
    media_type: str = "text",
    local_path: str | None = None,
    original_uri: str | None = None,
    obsidian_path: str | None = None,
    collections: list[str] | None = None,
    tags: list[str] | None = None,
) -> KnowledgeDocument:
    source_id = source_id or f"src-{sha256_text((local_path or original_uri or title) + text)[:20]}"
    checksum = sha256_text(text)
    reference = SourceReference(
        source_id=source_id,
        media_type=media_type,
        title=title,
        original_uri=original_uri,
        local_path=local_path,
        obsidian_path=obsidian_path,
        checksum=checksum,
    )
    return KnowledgeDocument(
        source_id=source_id,
        title=title,
        media_type=media_type,
        segments=[ContentSegment(id="segment-1", sequence=0, modality="text", text=text)],
        source=reference,
        collections=collections or [],
        tags=tags or [],
    )


def load_documents(
    path: str | Path,
    *,
    title: str | None = None,
    media_type: str | None = None,
    collections: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[KnowledgeDocument]:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path.suffix.lower() == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        documents = documents_from_ucb(data)
        for document in documents:
            if collections:
                document.collections = sorted(set(document.collections + collections))
            if tags:
                document.tags = sorted(set(document.tags + tags))
        return documents
    if source_path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError("V4 indexes UCB JSON, Markdown, or text; binary parsing remains in V1-V3")
    text = source_path.read_text(encoding="utf-8")
    kind = media_type or ("document" if source_path.suffix.lower() in {".md", ".markdown"} else "text")
    return [
        document_from_text(
            text,
            title=title or source_path.stem,
            source_id=f"file-{sha256_text(str(source_path))[:20]}",
            media_type=kind,
            local_path=str(source_path),
            collections=collections,
            tags=tags,
        )
    ]
