from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..providers.web import WebSearchProvider
from .deep_correction import ExternalEvidence
from .schema import TranscriptV2


_LATIN_TERM_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.+_-]{1,}|[A-Za-z]+\d+[A-Za-z0-9.+_-]*)\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9_-]{1,15}\b")
_TERM_TRIGGER_RE = re.compile(
    r"(?:称为|叫做|简称|专业术语|模型|框架|插件|工具|系统)[：:、，,\s]*("
    r"[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9.+_-]{1,23})"
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|倍|万|亿|GB|MB|ms|秒|分钟|小时|条|页)", re.I)


@dataclass(frozen=True, slots=True)
class ExternalEvidenceCollection:
    evidence: tuple[ExternalEvidence, ...]
    queries: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def _short_claim(text: str, match_start: int, match_end: int, *, radius: int = 36) -> str:
    start = max(0, match_start - radius)
    end = min(len(text), match_end + radius)
    return " ".join(text[start:end].split()).strip(" ，,。；;：:")


def plan_external_queries(
    transcript: TranscriptV2,
    *,
    known_terms: Mapping[str, Sequence[str]] | None = None,
    max_queries: int = 12,
) -> tuple[str, ...]:
    """Create bounded verification queries without uploading a whole transcript."""

    limit = max(0, min(50, int(max_queries)))
    if not limit:
        return ()
    full_text = "\n".join(segment.raw_text for segment in transcript.segments)
    folded = full_text.casefold()
    candidates: list[str] = []
    for canonical, variants in (known_terms or {}).items():
        values = [str(canonical).strip(), *(str(item).strip() for item in variants)]
        if any(value and value.casefold() in folded for value in values):
            candidates.append(str(canonical).strip())
    latin = Counter(_LATIN_TERM_RE.findall(full_text))
    acronyms = Counter(_ACRONYM_RE.findall(full_text))
    candidates.extend(term for term, _count in (latin + acronyms).most_common(limit * 2))
    candidates.extend(match.group(1) for match in _TERM_TRIGGER_RE.finditer(full_text))

    # Numeric claims are searched with only a short local context window.  The
    # caller enables this explicit networking stage; full paragraphs are never
    # sent to the search provider.
    claims: list[str] = []
    for segment in transcript.segments:
        for match in _NUMBER_RE.finditer(segment.raw_text):
            claim = _short_claim(segment.raw_text, match.start(), match.end())
            if 4 <= len(claim) <= 100:
                claims.append(claim)

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in [*candidates, *claims]:
        clean = " ".join(str(candidate or "").split()).strip()
        normalized = clean.casefold()
        if len(clean) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        suffix = " 官方 文档" if len(clean) <= 36 else " 来源"
        queries.append((clean + suffix)[:160])
        if len(queries) >= limit:
            break
    return tuple(queries)


def collect_external_evidence(
    transcript: TranscriptV2,
    provider: WebSearchProvider,
    *,
    known_terms: Mapping[str, Sequence[str]] | None = None,
    max_queries: int = 12,
    results_per_query: int = 2,
) -> ExternalEvidenceCollection:
    queries = plan_external_queries(
        transcript, known_terms=known_terms, max_queries=max_queries
    )
    if not provider.available:
        return ExternalEvidenceCollection((), queries, ("外部检索 Provider 不可用",))
    evidence: list[ExternalEvidence] = []
    warnings: list[str] = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            hits = provider.search(query, top_k=max(1, min(5, results_per_query)))
        except Exception as exc:
            warnings.append(f"外部核验查询失败：{query[:60]}（{type(exc).__name__}）")
            continue
        for hit in hits:
            title = " ".join(str(hit.title or "").split())[:300]
            snippet = " ".join(str(hit.content or "").split())[:1500]
            url = str(hit.url or "").strip()
            if not title or not snippet or not url or url in seen_urls:
                continue
            digest = hashlib.sha256(f"{query}\0{url}".encode("utf-8")).hexdigest()[:20]
            try:
                item = ExternalEvidence(
                    id=f"web-{digest}", title=title, snippet=snippet, url=url, query=query
                )
            except ValueError:
                warnings.append(f"外部检索返回了无效证据链接：{title[:50]}")
                continue
            seen_urls.add(url)
            evidence.append(item)
    if queries and not evidence:
        warnings.append("没有取得可逐字核验的外部证据；相关内容必须保守标注")
    return ExternalEvidenceCollection(tuple(evidence), queries, tuple(dict.fromkeys(warnings)))


__all__ = [
    "ExternalEvidenceCollection",
    "collect_external_evidence",
    "plan_external_queries",
]
