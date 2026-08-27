from __future__ import annotations

from .evidence import EvidenceBuilder
from .models import ConversationContext, Evidence


SYSTEM_PROMPT = """You answer questions over a private personal knowledge base and may receive user-provided images.
Treat only the supplied Evidence blocks as support for factual knowledge-base claims.
You may directly observe and explain user-provided images without a citation marker. Clearly distinguish image observations from knowledge-base claims.
If evidence is sufficient, answer the user's question directly.
If evidence is insufficient but a user image is present, answer what can be established from the image and explicitly identify what the knowledge base cannot establish.
If neither evidence nor a user image is sufficient, explicitly say that the knowledge base does not contain enough information.
Never invent missing facts to make the answer feel complete.
Every factual knowledge claim must include one or more exact citation markers such as [S1][S2].
Use only evidence IDs that are supplied in this request. Conversation context helps resolve references but is not evidence.
Reply in the user's language unless the user asks for another language.
Return Markdown only."""


def build_answer_prompt(
    question: str,
    context: ConversationContext,
    evidence: list[Evidence],
    correction_errors: list[str] | None = None,
    response_language: str | None = None,
    image_count: int = 0,
) -> tuple[str, str]:
    system = SYSTEM_PROMPT
    if response_language and response_language.casefold() in {"zh", "zh-cn", "chinese", "中文"}:
        system += (
            "\nRegardless of the source language, reply in clear Simplified Chinese. "
            "Preserve necessary English product names, technical terms, code, and exact citation markers. "
            "Translate or explain English evidence in Chinese instead of merely copying it."
        )
    if correction_errors:
        allowed = ", ".join(f"[{item.evidence_id}]" for item in evidence) or "none"
        system += (
            "\nYour previous draft failed citation validation. Correct every error. "
            f"Allowed citation markers: {allowed}. Errors: {'; '.join(correction_errors)}"
        )
    context_text = context.as_prompt()
    user_parts = [f"Question:\n{question}"]
    if image_count:
        user_parts.append(
            f"User-provided images: {image_count}. Inspect them carefully, including text, objects, "
            "charts, layout, spatial relationships, and uncertainty. Do not pretend unreadable details are clear."
        )
    if response_language:
        user_parts.append(f"Required response language: {response_language}")
    if context_text:
        user_parts.append(context_text)
    user_parts.append("Evidence:\n" + (EvidenceBuilder.context(evidence) or "(none)"))
    return system, "\n\n".join(user_parts)
