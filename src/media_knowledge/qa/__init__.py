from .models import Citation, Evidence, KnowledgeAnswer, QuestionAnalysis, TokenUsage
from .quality import EvidenceQuality

__all__ = [
    "KnowledgeQAEngine",
    "Citation",
    "Evidence",
    "EvidenceQuality",
    "KnowledgeAnswer",
    "QuestionAnalysis",
    "TokenUsage",
]


def __getattr__(name: str):
    if name == "KnowledgeQAEngine":
        from .engine import KnowledgeQAEngine

        return KnowledgeQAEngine
    raise AttributeError(name)
