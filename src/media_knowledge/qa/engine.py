from __future__ import annotations

import logging
from time import perf_counter
from typing import Callable

from ..citations import CitationValidationError, CitationValidator
from ..providers import AnswerProvider, DisabledWebSearchProvider, ExtractiveGroundedProvider, WebSearchProvider
from ..retrieval import KnowledgeRetriever
from ..storage import ConversationRepository, KnowledgeDatabase
from .analyzer import QuestionAnalyzer
from .evidence import EvidenceBuilder
from .models import AnswerRequest, AnswerResponse, KnowledgeAnswer, TokenUsage, new_id
from .prompt import build_answer_prompt
from .rewrite import ContextualQueryRewriter, QueryRewriter


logger = logging.getLogger(__name__)


class KnowledgeQAEngine:
    def __init__(
        self,
        database: KnowledgeDatabase,
        retriever: KnowledgeRetriever,
        *,
        answer_provider: AnswerProvider | None = None,
        web_search_provider: WebSearchProvider | None = None,
        analyzer: QuestionAnalyzer | None = None,
        query_rewriter: QueryRewriter | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        recent_context_limit: int = 6,
    ):
        self.database = database
        self.retriever = retriever
        self.answer_provider = answer_provider or ExtractiveGroundedProvider()
        self.web_search_provider = web_search_provider or DisabledWebSearchProvider()
        self.analyzer = analyzer or QuestionAnalyzer()
        self.query_rewriter = query_rewriter or ContextualQueryRewriter()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.recent_context_limit = recent_context_limit
        self.conversations = ConversationRepository(database)
        self.citation_validator = CitationValidator(self.conversations)

    @property
    def web_search_available(self) -> bool:
        return self.web_search_provider.available

    def ask(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        mode: str = "knowledge",
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        media_types: list[str] | None = None,
        folders: list[str] | None = None,
        document_ids: list[str] | None = None,
        date_range: tuple[str | None, str | None] | None = None,
        top_k: int = 10,
        response_language: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> KnowledgeAnswer:
        pipeline_started = perf_counter()
        normalized_mode = mode.casefold().replace("_", "+").replace(" ", "")
        if normalized_mode in {"knowledgeonly", "knowledge-only"}:
            normalized_mode = "knowledge"
        if normalized_mode not in {"knowledge", "knowledge+web"}:
            raise ValueError("mode must be 'knowledge' or 'knowledge+web'")
        if top_k < 1 or top_k > 12:
            raise ValueError("V5 evidence top_k must be between 1 and 12")

        conversation_id = self.conversations.ensure_conversation(
            conversation_id, title=question.strip()[:100] or None
        )
        context = self.conversations.context(conversation_id, self.recent_context_limit)
        analysis = self.analyzer.analyze(question)
        rewritten_query = self.query_rewriter.rewrite(analysis, context)
        question_message = self.conversations.add_message(
            conversation_id,
            "user",
            analysis.normalized_question,
            {
                "analysis": analysis.to_dict(),
                "rewritten_query": rewritten_query,
                "requested_mode": normalized_mode,
            },
        )

        retrieval_started = perf_counter()
        knowledge_results = self.retriever.search_knowledge(
            rewritten_query,
            collections=collections,
            tags=tags,
            media_types=media_types,
            folders=folders,
            document_ids=document_ids,
            date_range=date_range,
            top_k=top_k,
        )
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        web_requested = normalized_mode == "knowledge+web"
        web_started = perf_counter()
        web_results = (
            self.web_search_provider.search(rewritten_query, top_k=min(5, top_k))
            if web_requested and self.web_search_provider.available
            else []
        )
        web_ms = (perf_counter() - web_started) * 1000
        evidence = self.evidence_builder.build(knowledge_results, web_results)

        if progress_callback:
            if evidence:
                source_count = self.evidence_builder.source_count(evidence)
                progress_callback(
                    "answering",
                    f"已按相关性选出 {len(evidence)} 个候选片段，来自 {source_count} 份资料，正在生成中文回答",
                )
            else:
                progress_callback("answering", "没有找到达到相关性要求的知识片段，正在整理结果")

        answer_started = perf_counter()
        if evidence:
            response = self._generate_validated(
                question, context, evidence, response_language=response_language
            )
            validation = self.citation_validator.validate(response.markdown, evidence)
            citations = self.citation_validator.citations(validation, evidence)
        else:
            response = AnswerResponse(
                markdown="知识库中没有足够资料回答这个问题。",
                model="none",
                provider="system",
                token_usage=TokenUsage(),
            )
            citations = []
        answer_ms = (perf_counter() - answer_started) * 1000

        retrieval_info = {
            "original_question": analysis.normalized_question,
            "rewritten_query": rewritten_query,
            "question_analysis": analysis.to_dict(),
            "knowledge_result_count": len(knowledge_results),
            "web_result_count": len(web_results),
            "evidence_count": len(evidence),
            "requested_mode": normalized_mode,
            "effective_mode": "knowledge+web" if web_results else "knowledge",
            "web_provider": self.web_search_provider.name,
            "web_available": self.web_search_provider.available,
            "response_language": response_language,
            "filters": {
                "collections": collections or [],
                "tags": tags or [],
                "media_types": media_types or [],
                "folders": folders or [],
                "document_ids": document_ids or [],
                "date_range": list(date_range) if date_range else None,
            },
            "knowledge_candidates": [
                {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "score": result.score,
                    "page": result.page,
                    "slide": result.slide,
                    "timestamp_start": result.timestamp_start,
                    "timestamp_end": result.timestamp_end,
                    "debug": result.debug,
                }
                for result in knowledge_results
            ],
            "latency_ms": {
                "retrieval_and_rerank": round(retrieval_ms, 3),
                "web_search": round(web_ms, 3),
                "answer_and_citation_validation": round(answer_ms, 3),
            },
        }
        confidence = self._confidence(len(evidence), len(citations), response.provider)
        answer = KnowledgeAnswer(
            answer_id=new_id("answer"),
            conversation_id=conversation_id,
            markdown=response.markdown,
            citations=citations,
            evidence=evidence,
            model=response.model,
            provider=response.provider,
            token_usage=response.token_usage,
            retrieval_info=retrieval_info,
            confidence=confidence,
        )
        answer.retrieval_info["latency_ms"]["total"] = round(
            (perf_counter() - pipeline_started) * 1000, 3
        )
        self.conversations.save_answer(answer, question_message.message_id)
        self.conversations.refresh_summary(conversation_id, self.recent_context_limit)
        logger.info(
            "knowledge QA complete answer_id=%s evidence=%d citations=%d provider=%s",
            answer.answer_id,
            len(evidence),
            len(citations),
            answer.provider,
        )
        return answer

    def _generate_validated(
        self, question: str, context, evidence, *, response_language: str | None = None
    ) -> AnswerResponse:
        system_prompt, user_prompt = build_answer_prompt(
            question, context, evidence, response_language=response_language
        )
        response = self.answer_provider.generate(
            AnswerRequest(question, system_prompt, user_prompt, evidence, response_language)
        )
        validation = self.citation_validator.validate(response.markdown, evidence)
        if validation.valid:
            return response

        logger.warning("citation validation requested repair errors=%s", validation.errors)

        repair_system, repair_user = build_answer_prompt(
            question,
            context,
            evidence,
            correction_errors=validation.errors,
            response_language=response_language,
        )
        repaired = self.answer_provider.generate(
            AnswerRequest(question, repair_system, repair_user, evidence, response_language)
        )
        repaired.token_usage = TokenUsage(
            input_tokens=response.token_usage.input_tokens + repaired.token_usage.input_tokens,
            output_tokens=response.token_usage.output_tokens + repaired.token_usage.output_tokens,
        )
        final_validation = self.citation_validator.validate(repaired.markdown, evidence)
        if not final_validation.valid:
            raise CitationValidationError(final_validation.errors)
        return repaired

    @staticmethod
    def _confidence(evidence_count: int, citation_count: int, provider: str) -> float:
        if evidence_count == 0:
            return 0.0
        coverage = citation_count / evidence_count
        base = 0.5 if provider == "local-extractive" else 0.55
        return round(min(0.95, base + 0.25 * coverage + 0.02 * min(evidence_count, 5)), 3)
