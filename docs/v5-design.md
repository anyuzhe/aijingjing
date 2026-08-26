# V5 knowledge QA and citation design

> Historical design record. The grounding and citation principles remain active, while the current desktop application adds first-party multimodal ingestion, knowledge spaces, source reading, and DeepSeek/Kimi providers. See [README](../README.md).

## Pipeline

```text
Question
  -> QuestionAnalyzer
  -> ContextualQueryRewriter (summary + recent context)
  -> V4 KnowledgeRetriever (hybrid + rerank)
  -> EvidenceBuilder ([S1]...[S12])
  -> AnswerProvider
  -> CitationValidator
  -> KnowledgeAnswer + persistence
```

The original question remains unchanged for answering. The rewritten query is used only for retrieval and is recorded in `retrieval_info`.

## Grounding boundary

An external answer provider receives only the question, bounded conversation context, and selected Evidence blocks. It does not receive the source document, database, archive, or Obsidian vault. The system prompt requires an explicit insufficient-evidence response and citations for factual claims.

Citation validation checks that every marker is in the current Evidence set. Knowledge citations must resolve to a real `(document_id, chunk_id)` pair in SQLite. Web citations must resolve to a URL supplied by the active `WebSearchProvider`. A failed answer is regenerated once with the validation errors and allowed IDs; a second failure is rejected and is not saved as an Answer.

## Conversation memory

All messages stay local. Prompt context is bounded to six recent messages by default. Older messages are compacted into a rolling summary with a persisted `summary_through_ordinal`, so already summarized messages are not repeatedly appended.

## Storage

Schema V3 adds separate `conversations`, `messages`, `answers`, `answer_evidence`, and `citations` tables. Citation locator fields are copied into the citation row. If an indexed document is later deleted, document/chunk foreign keys become null while the historical source snapshot remains available.

## Modes

`knowledge` is the default. `knowledge+web` calls the configured `WebSearchProvider` only when its `available` property is true. With no provider, the engine continues in knowledge-only mode and records the fallback in `retrieval_info`.
