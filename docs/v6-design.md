# V6 Knowledge Workbench design

> Historical design record for the optional loopback browser workbench. AI静静 2.0.1 uses the independent PySide6 desktop application and does not require Codex or Obsidian. See [README](../README.md) for the current product architecture.

## Architecture

V6 adds a local web surface to the existing Python package. It does not introduce a second search or QA backend.

```text
Browser Workbench
  |-- GET /api/bootstrap --------> SQLite navigation metadata
  |-- POST /api/search ----------> V4 KnowledgeRetriever
  |-- POST /api/ask/stream ------> V5 KnowledgeQAEngine
  |-- GET /api/conversations/:id > ConversationRepository
  |-- GET /api/source/content ---> trusted SourceReference + byte ranges
  |-- POST /api/skills/pick-files > native local file picker
  |-- POST /api/skills/invoke/stream > allowlisted local Codex Skill
  `-- POST /api/obsidian/save ---> persisted Answer -> atomic Markdown note
```

The HTTP server binds to loopback by default. Durable knowledge, conversations, answers, and citations remain in the existing local SQLite database. Browser storage holds only the current conversation pointer.

## User surface

- Left: All Knowledge, Collections, Folders, Tags, Media Type, Recent Documents, and Conversation History.
- Center: separate Search and Ask AI modes, streamed answer rendering, Markdown/code/table support, citations, history, Knowledge/Web/Deep controls, and Model Auto.
- Right: exact Evidence text, media type, page/slide/timestamp/section, local or online source actions, media playback, and Obsidian deep links.

Scope filters flow through `SearchFilters` to both BM25 and vector retrieval. A recent document selection uses its real `document_id`; folders match indexed local or Obsidian parent paths.

Workbench requests set `response_language=zh-CN`. The answer prompt requires Simplified Chinese synthesis regardless of evidence language while preserving technical terms and exact citation markers. With no explicit QA provider override, the Workbench uses the authenticated local Codex CLI in a read-only, no-approval, ephemeral run; ordinary CLI QA retains the offline extractive fallback.

The model catalog is server-owned and allowlisted. Built-in Codex choices include Auto, GPT-5.6 Luna/Terra/Sol, GPT-5.5, and GPT-5.4; the local extractive provider is always available. Additional OpenAI-compatible model IDs are exposed only when explicitly configured through `KNOWLEDGE_QA_MODELS` or the local multi-provider `providers.json`. A provider may set `temperature` to `null` when its models require the sampling parameter to be omitted. Provider credentials never enter bootstrap responses. Auto uses Luna for normal questions and Terra for deep analysis.

## Source and Obsidian safety

Local source bytes are served only after resolving a stored `chunk_id`; the browser cannot submit an arbitrary filesystem path. Audio and video endpoints support HTTP byte ranges. Native opening also resolves the path from the indexed SourceReference.

Saving an answer accepts a persisted `answer_id`, not arbitrary Markdown. The writer reloads the original question, answer, citations, and related documents from SQLite, writes into a fixed vault subdirectory with an atomic replace, and returns an `obsidian://` URI. The action is disabled when no existing vault is configured.

The Skill bridge accepts only `knowledge-ingestor`, passes prompts through standard input rather than a shell, validates selected files, runs in a workspace-write sandbox, and grants additional write roots only from the Skill's configured archive and Obsidian targets. Selecting and running the Skill is a separate, confirmed action; ordinary Search and Ask requests never start Codex or perform ingestion. Skill completion reports are stored as typed conversation messages.

The Obsidian sync closes the persistence-to-retrieval gap. It scans Markdown at Workbench startup, after a successful Skill run, or through `POST /api/obsidian/sync`; derives a stable source ID from vault and relative path; preserves `obsidian_path`; imports frontmatter tags and the top-level vault directory as facets; and delegates chunk, embedding, and FTS updates to `IndexingService`. Sync-owned notes that disappear from a complete scan are deleted from the retrieval index. A partial filesystem scan never triggers deletion. Hidden directories, `_assets`, and generated `10_Knowledge/AI Answers` are excluded.

## Local-only decision

This UI is intentionally not deployed to a public host. It depends on a private local SQLite index, local source files, native media opening, and an Obsidian vault. Publishing it without an authenticated private tunnel would either break those capabilities or expose personal knowledge paths.
