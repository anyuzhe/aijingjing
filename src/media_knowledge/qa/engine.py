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
from .models import AnswerRequest, AnswerResponse, ImageAttachment, KnowledgeAnswer, TokenUsage, new_id
from .prompt import build_answer_prompt
from .quality import evaluate_evidence_quality
from .rewrite import ContextualQueryRewriter, QueryRewriter
from .strategy import AdaptiveRetrievalPlanner


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
        retrieval_planner: AdaptiveRetrievalPlanner | None = None,
        recent_context_limit: int = 6,
    ):
        self.database = database
        self.retriever = retriever
        self.answer_provider = answer_provider or ExtractiveGroundedProvider()
        self.web_search_provider = web_search_provider or DisabledWebSearchProvider()
        self.analyzer = analyzer or QuestionAnalyzer()
        self.query_rewriter = query_rewriter or ContextualQueryRewriter()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.retrieval_planner = retrieval_planner or AdaptiveRetrievalPlanner(database)
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
        delta_callback: Callable[[str], None] | None = None,
        image_attachments: list[ImageAttachment] | None = None,
    ) -> KnowledgeAnswer:
        pipeline_started = perf_counter()
        normalized_mode = mode.casefold().replace("_", "+").replace(" ", "")
        if normalized_mode in {"knowledgeonly", "knowledge-only"}:
            normalized_mode = "knowledge"
        if normalized_mode not in {"knowledge", "knowledge+web"}:
            raise ValueError("mode must be 'knowledge' or 'knowledge+web'")
        if top_k < 1 or top_k > 12:
            raise ValueError("V5 evidence top_k must be between 1 and 12")

        supplied_images = list(image_attachments or [])[:4]
        effective_question = question.strip() or (
            "请仔细分析这张图片，并结合知识库说明其中的内容。" if supplied_images else ""
        )
        if not effective_question:
            raise ValueError("请输入问题或添加图片")
        conversation_id = self.conversations.ensure_conversation(
            conversation_id, title=effective_question[:100] or None
        )
        context = self.conversations.context(conversation_id, self.recent_context_limit)
        analysis = self.analyzer.analyze(effective_question)
        active_images = supplied_images or (
            context.latest_image_attachments() if analysis.is_follow_up else []
        )
        rewritten_query = self.query_rewriter.rewrite(analysis, context)
        question_message = self.conversations.add_message(
            conversation_id,
            "user",
            analysis.normalized_question,
            {
                "analysis": analysis.to_dict(),
                "rewritten_query": rewritten_query,
                "requested_mode": normalized_mode,
                "image_attachments": [item.to_dict() for item in supplied_images],
                "reused_previous_images": bool(active_images and not supplied_images),
            },
        )

        retrieval_started = perf_counter()
        focused_results = self.retriever.search_knowledge(
            rewritten_query,
            collections=collections,
            tags=tags,
            media_types=media_types,
            folders=folders,
            document_ids=document_ids,
            date_range=date_range,
            top_k=top_k,
        )
        retrieval_selection = self.retrieval_planner.select(
            analysis,
            focused_results,
            document_ids=document_ids,
            top_k=top_k,
            collections=collections,
            tags=tags,
            media_types=media_types,
            folders=folders,
            date_range=date_range,
        )
        knowledge_results = retrieval_selection.results
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
                if retrieval_selection.strategy == "full_context":
                    message = (
                        f"已在上下文预算内载入该资料的全部 {len(evidence)} 个片段，"
                        "正在生成中文回答"
                    )
                elif retrieval_selection.strategy == "hierarchical":
                    message = (
                        f"资料较长，已按章节与位置选出 {len(evidence)} 个代表片段，"
                        "正在生成中文回答"
                    )
                else:
                    message = (
                        f"已按相关性选出 {len(evidence)} 个候选片段，来自 "
                        f"{source_count} 份资料，正在生成中文回答"
                    )
                progress_callback(
                    "answering",
                    message,
                )
            else:
                progress_callback("answering", "没有找到达到相关性要求的知识片段，正在整理结果")

        answer_started = perf_counter()
        if evidence or active_images:
            response = self._generate_validated(
                effective_question,
                context,
                evidence,
                response_language=response_language,
                image_attachments=active_images,
                delta_callback=delta_callback,
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
            if delta_callback:
                delta_callback(response.markdown)
        answer_ms = (perf_counter() - answer_started) * 1000

        retrieval_info = {
            "original_question": analysis.normalized_question,
            "rewritten_query": rewritten_query,
            "question_analysis": analysis.to_dict(),
            "knowledge_result_count": len(knowledge_results),
            "focused_result_count": len(focused_results),
            "candidate_count": len(focused_results),
            "retrieved_count": len(knowledge_results),
            "web_result_count": len(web_results),
            "evidence_count": len(evidence),
            "requested_mode": normalized_mode,
            "effective_mode": "knowledge+web" if web_results else "knowledge",
            "web_provider": self.web_search_provider.name,
            "web_available": self.web_search_provider.available,
            "response_language": response_language,
            "image_count": len(active_images),
            "new_image_count": len(supplied_images),
            "retrieval_strategy": retrieval_selection.strategy,
            "retrieval_strategy_details": retrieval_selection.details,
            "untrusted_evidence_boundary": True,
            "instruction_risk_evidence_count": sum(item.instruction_risk for item in evidence),
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
        evidence_quality = evaluate_evidence_quality(
            response.markdown,
            evidence,
            citations,
            retrieval_strategy=retrieval_selection.strategy,
            image_count=len(active_images),
        )
        retrieval_info["evidence_quality"] = evidence_quality.to_dict()
        retrieval_info["confidence_semantics"] = (
            "compatibility alias for citation_coverage; this is not a probability of truth"
        )
        confidence = evidence_quality.citation_coverage
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
            evidence_quality=evidence_quality,
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
        self,
        question: str,
        context,
        evidence,
        *,
        response_language: str | None = None,
        image_attachments: list[ImageAttachment] | None = None,
        delta_callback: Callable[[str], None] | None = None,
    ) -> AnswerResponse:
        images = list(image_attachments or [])
        system_prompt, user_prompt = build_answer_prompt(
            question,
            context,
            evidence,
            response_language=response_language,
            image_count=len(images),
        )
        response = self.answer_provider.generate(
            AnswerRequest(
                question=question,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                evidence=evidence,
                response_language=response_language,
                image_attachments=images,
                delta_callback=delta_callback,
            )
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
            image_count=len(images),
        )
        repaired = self.answer_provider.generate(
            AnswerRequest(
                question=question,
                system_prompt=repair_system,
                user_prompt=repair_user,
                evidence=evidence,
                response_language=response_language,
                image_attachments=images,
            )
        )
        repaired.token_usage = TokenUsage(
            input_tokens=response.token_usage.input_tokens + repaired.token_usage.input_tokens,
            output_tokens=response.token_usage.output_tokens + repaired.token_usage.output_tokens,
        )
        final_validation = self.citation_validator.validate(repaired.markdown, evidence)
        if not final_validation.valid:
            raise CitationValidationError(final_validation.errors)
        return repaired
