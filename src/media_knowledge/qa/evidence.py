from __future__ import annotations

import json
import re

from ..models import SearchResult, SourceReference
from ..providers.web import WebSearchHit
from .models import Evidence


class EvidenceBuilder:
    _INSTRUCTION_PATTERNS = (
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
        re.compile(r"(?:system|developer)\s*(?:prompt|message)\s*:", re.I),
        re.compile(r"(?:忽略|无视).{0,12}(?:指令|要求|提示词)"),
        re.compile(r"(?:系统|开发者)(?:提示词|消息|指令)\s*[：:]"),
        re.compile(r"(?:执行|运行).{0,16}(?:命令|脚本|代码)"),
    )

    @classmethod
    def contains_instruction_like_text(cls, content: str) -> bool:
        return any(pattern.search(content) for pattern in cls._INSTRUCTION_PATTERNS)

    def build(
        self,
        knowledge_results: list[SearchResult],
        web_results: list[WebSearchHit] | None = None,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        seen_chunks: set[str] = set()
        for result in knowledge_results:
            if result.chunk_id in seen_chunks:
                continue
            seen_chunks.add(result.chunk_id)
            evidence.append(
                Evidence(
                    evidence_id="",
                    content=result.content,
                    title=result.title,
                    score=result.score,
                    source=result.source,
                    source_kind="knowledge",
                    instruction_risk=self.contains_instruction_like_text(result.content),
                )
            )
        for hit in web_results or []:
            if not hit.url or not hit.content.strip():
                continue
            reference = SourceReference(
                source_id=f"web:{hit.url}",
                media_type="web",
                title=hit.title,
                original_uri=hit.url,
            )
            evidence.append(
                Evidence(
                    evidence_id="",
                    content=hit.content,
                    title=hit.title,
                    score=hit.score,
                    source=reference,
                    source_kind="web",
                    instruction_risk=self.contains_instruction_like_text(hit.content),
                )
            )
        evidence.sort(key=lambda item: (-item.score, item.title, item.source.source_id))
        for index, item in enumerate(evidence, 1):
            item.evidence_id = f"S{index}"
        return evidence

    @staticmethod
    def source_count(evidence: list[Evidence]) -> int:
        identities = {
            item.source.document_id
            or item.source.original_uri
            or item.source.local_path
            or item.source.source_id
            for item in evidence
        }
        return len(identities)

    @staticmethod
    def context(evidence: list[Evidence], max_characters_per_item: int = 3600) -> str:
        blocks: list[str] = []
        for item in evidence:
            locator = item.locator()
            blocks.append(
                json.dumps(
                    {
                        "evidence_id": item.evidence_id,
                        "title": item.title,
                        "media_type": item.source.media_type,
                        "locator": locator or None,
                        "uri": item.source.original_uri,
                        "instruction_like_content_detected": item.instruction_risk,
                        "content": item.content[:max_characters_per_item],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        if not blocks:
            return ""
        return (
            "BEGIN_UNTRUSTED_EVIDENCE_JSONL\n"
            + "\n".join(blocks)
            + "\nEND_UNTRUSTED_EVIDENCE_JSONL"
        )
