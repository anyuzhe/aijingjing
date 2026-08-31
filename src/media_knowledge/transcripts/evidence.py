from __future__ import annotations

import hashlib
import math
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
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+_-]{1,}", re.I)
_HAN_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
_QUERY_SUFFIXES = ("官方", "文档", "来源")
_GENERIC_LATIN_TOKENS = frozenset({"ai", "app", "agent", "agents", "ok"})
_GENERIC_HAN_NGRAMS = frozenset({
    "这个系统", "一个系统", "这种情况", "可以看到", "我们可以", "所以这个",
    "就是这个", "相关信息", "官方文档", "完整内容", "主要内容", "最新消息",
})


@dataclass(frozen=True, slots=True)
class ExternalEvidenceCollection:
    evidence: tuple[ExternalEvidence, ...]
    queries: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def _hit_relevant(query: str, title: str, snippet: str) -> bool:
    """Reject obvious search-engine drift before a hit reaches the LLM.

    Search snippets remain untrusted even after this filter.  The gate merely
    requires an exact technical/number token or a non-generic four-character
    Chinese phrase shared with the bounded query.
    """

    query_core = str(query or "")
    for suffix in _QUERY_SUFFIXES:
        query_core = query_core.replace(suffix, " ")
    haystack = f"{title}\n{snippet}".casefold()
    tokens = {
        token.casefold()
        for token in _LATIN_TOKEN_RE.findall(query_core)
        if len(token) >= 3 and token.casefold() not in _GENERIC_LATIN_TOKENS
    }
    if any(token in haystack for token in tokens):
        return True

    query_han = "".join(_HAN_RUN_RE.findall(query_core))
    hit_han = "".join(_HAN_RUN_RE.findall(f"{title}{snippet}"))
    if len(query_han) < 4 or len(hit_han) < 4:
        return False
    required = min(12, max(5, math.ceil(len(query_han) * 0.35)))
    if len(query_han) < required:
        required = len(query_han)
    for size in range(min(len(query_han), 18), required - 1, -1):
        for index in range(len(query_han) - size + 1):
            phrase = query_han[index:index + size]
            if phrase not in _GENERIC_HAN_NGRAMS and phrase in hit_han:
                return True
    return False


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
    filtered_hits = 0
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
            if not _hit_relevant(query, title, snippet):
                filtered_hits += 1
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
    if filtered_hits:
        warnings.append(f"已过滤 {filtered_hits} 条与核验查询缺少明确词项交集的候选网页")
    if queries and not evidence:
        warnings.append("没有取得可逐字核验的外部证据；相关内容必须保守标注")
    return ExternalEvidenceCollection(tuple(evidence), queries, tuple(dict.fromkeys(warnings)))


__all__ = [
    "ExternalEvidenceCollection",
    "collect_external_evidence",
    "plan_external_queries",
]
