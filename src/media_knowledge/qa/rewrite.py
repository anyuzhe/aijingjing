from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .models import ConversationContext, QuestionAnalysis


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, analysis: QuestionAnalysis, context: ConversationContext) -> str:
        raise NotImplementedError


class ContextualQueryRewriter(QueryRewriter):
    def rewrite(self, analysis: QuestionAnalysis, context: ConversationContext) -> str:
        if not analysis.is_follow_up or (not context.summary and not context.recent_messages):
            return analysis.normalized_question
        subjects = context.subject_candidates()
        if subjects:
            return re.sub(r"\s+", " ", " ".join([*subjects, analysis.normalized_question])).strip()
        previous_user = next(
            (message.content for message in reversed(context.recent_messages) if message.role == "user"),
            context.summary,
        )
        compact = re.sub(r"\s+", " ", previous_user).strip()[:500]
        return f"{compact}；{analysis.normalized_question}" if compact else analysis.normalized_question
