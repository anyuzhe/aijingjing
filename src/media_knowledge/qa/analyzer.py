from __future__ import annotations

import re

from .models import QuestionAnalysis


class QuestionAnalyzer:
    FOLLOW_UP_PATTERN = re.compile(
        r"^(?:那|那么|所以|还有|然后)|(?:它|这个|那个|该|其|这些|上述|前面|刚才)|\b(?:it|that|those|they|this)\b",
        re.IGNORECASE,
    )

    def analyze(self, question: str) -> QuestionAnalysis:
        normalized = re.sub(r"\s+", " ", question).strip()
        if not normalized:
            raise ValueError("question must not be empty")
        lowered = normalized.casefold()
        if any(token in normalized for token in ("比较", "区别", "异同", "对比")) or "compare" in lowered:
            task_type = "compare"
        elif any(token in normalized for token in ("综合", "总结", "到底", "归纳")) or "synthesize" in lowered:
            task_type = "synthesis"
        elif any(token in normalized for token in ("为什么", "如何", "怎么")) or any(
            token in lowered for token in ("why", "how")
        ):
            task_type = "explanation"
        else:
            task_type = "factual"
        keywords = re.findall(r"[A-Za-z0-9_-]{2,}|[\u3400-\u9fff]{2,8}", normalized)
        return QuestionAnalysis(
            original_question=question,
            normalized_question=normalized,
            is_follow_up=bool(self.FOLLOW_UP_PATTERN.search(normalized)),
            task_type=task_type,
            keywords=list(dict.fromkeys(keywords))[:16],
        )
