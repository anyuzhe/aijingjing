<div align="center">
  <img src="packaging/AI-Jingjing.png" width="128" alt="AI Jingjing Logo">
  <h1>AI Knowledge Base · AI Jingjing</h1>
  <p><strong>A local-first, multimodal, source-grounded personal knowledge base for desktop</strong></p>
  <p><a href="README.md">简体中文</a> · English</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/UI-PySide6-41CD52" alt="PySide6">
    <img src="https://img.shields.io/badge/Storage-SQLite%20FTS5-0F80CC" alt="SQLite FTS5">
    <img src="https://img.shields.io/badge/Test-68%20passed-2F855A" alt="68 tests passed">
    <img src="https://img.shields.io/badge/Version-2.0.5-4C8FBF" alt="Version 2.0.5">
  </p>
</div>

AI Jingjing turns PDFs, PowerPoint decks, Word files, images, audio, video, web pages, and Markdown into one searchable local knowledge base with multi-turn Q&A and precise source traceability.

It is a standalone desktop application. End users do not need Codex, Obsidian, Python, or a separate FFmpeg installation. Obsidian is an optional sync/export target. Cloud models such as DeepSeek and Kimi are contacted only when the user explicitly requests an AI answer, knowledge synthesis, or visual understanding.

## Why AI Jingjing

A file manager preserves sources but cannot reason across them. A generic chat interface may answer questions but often loses the original evidence. AI Jingjing keeps all three layers together:

```text
Original sources and reproducible archives
                  ↓
Located chunks and local semantic indexes
                  ↓
Cited answers, notes, annotations, and knowledge artifacts
```

Every supported answer can lead back to an exact PDF page, PowerPoint slide, media timestamp, source file, or web snapshot.

## Highlights

### 1. Unified multimodal ingestion

- Select or drag multiple files in one batch.
- Ingest Markdown, text, Word, PDF, PPTX, images, audio, video, and web pages.
- Understand PDFs and presentations page by page instead of extracting text only.
- Produce timestamped speech transcripts and optional video keyframes.
- Group related PPT/PDF/audio/video files into one `Source Package`.
- Archive originals, web snapshots, transcripts, retained assets, and parse manifests.
- Run the first-party Python `IngestionService` directly, with no local Codex CLI dependency.

### 2. Strict ingestion quality gate

Before a source is indexed, the application checks:

- whether the real document body or media stream was obtained;
- whether the result contains only a title, description, cover, or platform metadata;
- PDF and presentation page coverage;
- whether audio/video produced real speech or visual evidence;
- checksums, extracted content size, and parser warnings.

A restricted video page that exposes only a cover and description is rejected instead of being stored as fake video knowledge.

### 3. Local semantic retrieval

- SQLite FTS5 full-text retrieval;
- local multilingual semantic embeddings;
- vector and BM25 candidate recall;
- Reciprocal Rank Fusion and local reranking;
- automatic final relevance sorting with clearly unrelated candidates removed;
- filters for spaces, tags, media types, folders, and exact documents;
- visible fused, semantic, and keyword-hit diagnostics.

Search does not call an LLM. The first semantic search downloads an approximately 240 MB ONNX model; subsequent searches can run offline.

### 4. Multi-turn, citation-grounded Q&A

- Continue asking follow-up questions in the same conversation.
- Paste screenshots directly into the composer, drag images into it, or select up to four images with the attachment button.
- Preview and remove attachments before sending; a vision model jointly understands text, images, and retrieved evidence.
- Refer to the previous turn's image in a follow-up without attaching it again.
- Rewrite retrieval queries using recent messages and a rolling summary.
- Require every cited claim to use evidence from the current retrieval result.
- Validate citation IDs against real local document and chunk IDs.
- Click an inline citation to open the source reader at the matching page or timestamp.
- Return an explicit insufficient-evidence response instead of inventing facts.

Answer providers include DeepSeek, Kimi, and a fully offline extractive evidence model. Once DeepSeek is configured, the image-capable experimental `deepseek-v4-flash-vision-exp` model is selected by default. Retrieval itself remains local and model-independent.

### 5. Source reading and knowledge management

- Read PDF pages, images, extracted text, and media timelines inside the app.
- Inspect every parsed chunk and its locator.
- Attach notes or learning cards to a specific evidence chunk.
- Rename, disable, re-enable, reparse, or remove a source.
- Organize documents using knowledge spaces and tags.
- Detect duplicate content fingerprints.
- Preserve archived originals when a searchable record is removed.

### 6. Automation and knowledge workshop

- Watch one or more folders and ingest changes incrementally.
- Reparse changed files only.
- Disable missing sources instead of silently destroying knowledge.
- Optionally sync from Obsidian or export AI Jingjing notes to a vault.
- Generate reports, cross-source comparisons, timelines, quizzes, flashcards, and mind-map outlines from selected evidence.

### 7. Local data security

- Keep documents, chunks, conversations, answers, evidence, and citations in local SQLite.
- Store API keys in macOS Keychain or the platform credential store when available.
- Explicitly exclude API keys from application backups.
- Provide full backups, pre-restore safety snapshots, integrity checks, and FTS repair.
- Accept update manifests over HTTPS only.
- Exclude personal databases, model caches, archives, credentials, and build outputs from Git.

## Supported inputs

| Type | Extensions / form | Processing | Traceability |
|---|---|---|---|
| Markdown / text | `.md` `.txt` `.csv` `.json` `.yaml` | headings, paragraphs, code, tables | heading path, text range |
| Word | `.docx` | paragraphs, headings, tables | section and source file |
| PDF | `.pdf` | per-page text, scanned-page OCR, page images | page number |
| PowerPoint | `.pptx` | slide text, notes, images, structure | slide number |
| Images | `.png` `.jpg` `.webp` `.tiff`, etc. | OCR and optional vision analysis | original image and retained asset |
| Audio | `.mp3` `.m4a` `.wav` `.flac`, etc. | Whisper transcription | start/end timestamp |
| Video | `.mp4` `.mov` `.mkv` `.webm`, etc. | FFmpeg, Whisper, keyframes | timeline and keyframe |
| Web/media URL | `https://...` | article extraction, snapshot, or authentic media download | URL, snapshot, timestamp |
| Weixin public article | `mp.weixin.qq.com/s/...` | dedicated title/body extraction and challenge-page blocking | article URL, body snapshot |

## Architecture

```text
PySide6 Desktop UI
        │
        ├── IngestionService
        │   ├── Document / PDF / PPT / image parsers
        │   ├── OCR / Whisper / FFmpeg
        │   ├── Authenticity and completeness quality gate
        │   └── Reproducible Source Package archive
        │
        ├── KnowledgeDatabase (SQLite)
        │   ├── Documents / Chunks / Source References
        │   ├── FTS5 / Embeddings
        │   └── Conversations / Evidence / Citations
        │
        ├── KnowledgeRetriever
        │   └── Semantic Vector + BM25 + RRF + Rerank
        │
        └── KnowledgeQAEngine
            ├── Contextual query rewriting
            ├── DeepSeek / Kimi / Local Extractive
            └── Evidence construction and citation validation
```

## Quick start

Python 3.11 or newer is required.

```bash
git clone git@github.com:anyuzhe/aijingjing.git
cd aijingjing

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[full]'

ai-jingjing
```

For UI and semantic-search development without all multimedia components:

```bash
pip install -e '.[desktop,semantic]'
```

### Configure answer providers

Open **Settings → Models & Privacy** and enter a DeepSeek or Kimi API key. Keys are never returned by status APIs, written to SQLite, included in diagnostics, or copied into backups.

Without an API key, ingestion, local search, and the offline extractive answer provider remain available.

## CLI

```bash
# Start the desktop application
knowledge desktop

# Batch multimodal ingestion
knowledge ingest report.pdf slides.pptx recording.m4a

# Local hybrid search
knowledge search "frontline learning loop in FDE"

# Grounded Q&A with citations
knowledge ask "What is FDE?"

# Status, diagnostics, and index rebuild
knowledge index-status
knowledge doctor
knowledge reindex
```

Search and Q&A support scope options including `--collection`, `--tag`, `--media-type`, `--folder`, and `--document-id`.

## Local data layout

```text
AI-Jingjing/
├── archive/       Originals, web snapshots, and Source Packages
├── notes/         Source Notes, saved answers, workshop artifacts
├── assets/        PDF/PPT pages, images, and video keyframes
├── transcripts/   Audio/video transcripts
├── cache/models/  Local semantic-model cache
├── backups/       Validated backups
├── trash/         Recoverable material
├── knowledge.db   Index, conversations, answers, evidence, citations
├── providers.json Provider metadata; migrated keys are not stored in plaintext
└── settings.json  Application settings
```

- macOS: `~/Library/Application Support/AI-Jingjing`
- Windows: `%LOCALAPPDATA%\AI-Jingjing`
- Linux: `${XDG_DATA_HOME:-~/.local/share}/AI-Jingjing`

Use `AI_JINGJING_DATA_DIR` or the `--data-dir` option to select another directory.

## Privacy boundary

| Operation | Cloud call | Data sent |
|---|---:|---|
| Chunking, SQLite, FTS5, local semantic search | No | None |
| Local extractive answer | No | None |
| DeepSeek/Kimi answer | Yes | question, bounded conversation context, retrieved evidence chunks, and images explicitly attached by the user |
| DeepSeek knowledge synthesis | Yes | bounded extracted text from the current import |
| DeepSeek Vision/Kimi visual analysis | Yes | a limited set of images selected by the ingestion policy |

Model providers never receive the entire database, archive directory, or Obsidian vault.

## Tests

```bash
pip install pytest
pytest -q
```

Version 2.0.5 includes 68 automated tests covering chunking, migration, model-catalog migration, the DeepSeek Vision default, clipboard-image handling, multimodal request payloads, image follow-ups, relevance sorting and false-hit filtering, multi-turn desktop conversations, citations, Weixin article extraction and challenge-page blocking, quality gates, synchronization, backup/restore, and product behavior.

## Build desktop packages

macOS:

```bash
./scripts/build_desktop.sh
```

Windows PowerShell:

```powershell
.\scripts\build_windows.ps1
```

Public macOS distribution requires an Apple Developer ID. `packaging/sign_and_notarize.sh` implements the signing and notarization workflow.

## Repository layout

```text
src/media_knowledge/
├── desktop/       PySide6 application, controller, diagnostics, updates
├── ingestion/     Multimodal extraction, vision, quality gate, archive
├── chunking/      Media-aware chunking
├── embedding/     Local semantic and compatible embeddings
├── retrieval/     Vector, FTS5, fusion, reranking
├── qa/            Multi-turn Q&A, evidence, prompts, citations
├── storage/       SQLite, vectors, conversations
└── sync/          Obsidian and watched-folder synchronization
```

## Boundaries

- Login-protected, DRM-protected, or anti-bot media pages are ingested only after the authentic body or media stream has been obtained.
- AI synthesis never replaces original evidence; important pages and originals stay archived.
- The project does not bypass access control. Import only material you are authorized to process.
- No open-source license is included yet. Usage and redistribution remain subject to a future declaration by the repository owner.

## Contributing

Issues and pull requests are welcome. Changes to retrieval, quality gates, citations, or database migrations should include regression coverage and keep `pytest -q` green.
