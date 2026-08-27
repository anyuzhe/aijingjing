"""AI知识库-AI静静 local-first ingestion, retrieval, and grounded QA core."""

from .indexing.service import IndexingService
from .qa.engine import KnowledgeQAEngine
from .retrieval.hybrid import KnowledgeRetriever

__all__ = ["IndexingService", "KnowledgeRetriever", "KnowledgeQAEngine"]
__version__ = "2.0.5"
