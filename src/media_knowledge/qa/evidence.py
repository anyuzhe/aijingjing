from __future__ import annotations

from ..models import SearchResult, SourceReference
from ..providers.web import WebSearchHit
from .models import Evidence


class EvidenceBuilder:
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
                    evidence_id=f"S{len(evidence) + 1}",
                    content=result.content,
                    title=result.title,
                    score=result.score,
                    source=result.source,
                    source_kind="knowledge",
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
                    evidence_id=f"S{len(evidence) + 1}",
                    content=hit.content,
                    title=hit.title,
                    score=hit.score,
                    source=reference,
                    source_kind="web",
                )
            )
        return evidence

    @staticmethod
    def context(evidence: list[Evidence], max_characters_per_item: int = 3600) -> str:
        blocks = []
        for item in evidence:
            locator = item.locator()
            header = f"[{item.evidence_id}] title={item.title!r} type={item.source.media_type}"
            if locator:
                header += f" locator={locator!r}"
            if item.source.original_uri:
                header += f" uri={item.source.original_uri!r}"
            blocks.append(header + "\n" + item.content[:max_characters_per_item])
        return "\n\n".join(blocks)
