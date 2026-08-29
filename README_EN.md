<div align="center">
  <img src="packaging/AI-Jingjing.png" width="128" alt="AI Jingjing Logo">
  <h1>AI Knowledge Base · AI Jingjing</h1>
  <p><strong>A local-first, multimodal, source-grounded personal knowledge base for desktop</strong></p>
  <p><a href="README.md">简体中文</a> · English</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/UI-PySide6-41CD52" alt="PySide6">
    <img src="https://img.shields.io/badge/Storage-SQLite%20FTS5-0F80CC" alt="SQLite FTS5">
    <img src="https://img.shields.io/badge/Test-205%20passed-2F855A" alt="205 tests passed">
    <img src="https://img.shields.io/badge/Version-2.2.0-4C8FBF" alt="Version 2.2.0">
  </p>
</div>

AI Jingjing turns PDFs, PowerPoint decks, Word files, images, audio, video, web pages, and Markdown into one searchable local knowledge base with multi-turn Q&A, precise source traceability, and maintainable knowledge lifecycles.

See [CHANGELOG.md](CHANGELOG.md) for release changes.

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
- The official Apple Silicon bundle includes MLX/Metal acceleration; NVIDIA systems prefer CUDA; other environments use the bundled CPU int8 runtime and display the actual route explicitly. Available models are `tiny / base / small / medium / large-v3`, with `large-v3` providing the highest accuracy.
- Preserve JSON, Markdown, TXT, SRT, and VTT transcripts together with engine, device, language, integrity metrics, and fallback reasons.
- Ingest public YouTube, Bilibili, Douyin, Xiaohongshu, and X links using public subtitles first, then only chunk-monitored HTTP/HTTPS or native DASH media. Live, unfinished replay, and HLS transports that could fall back to an unbounded FFmpeg downloader are rejected before download with guidance to save an authorized local copy first.
- Group related PPT/PDF/audio/video files into one `Source Package`.
- Archive originals, web snapshots, transcripts, retained assets, and parse manifests.
- Run the first-party Python `IngestionService` directly, with no local Codex CLI dependency.

### 2. Strict ingestion quality gate

Before a source is indexed, the application checks:

- whether the real document body or media stream was obtained;
- whether the result contains only a title, description, cover, or platform metadata;
- PDF and presentation page coverage;
- whether audio/video produced real speech or visual evidence;
- OCR line coordinates, mean/minimum confidence, low-confidence lines, and complex-layout fallback reasons;
- PP-StructureV3 Markdown tables, formulas, layout geometry, and page reading order;
- reversed, out-of-range, abnormally overlapping, empty, or poorly covered transcript segments;
- checksums, extracted content size, and parser warnings.

A restricted video page that exposes only a cover and description is rejected instead of being stored as fake video knowledge.

### 3. Governed knowledge lifecycle

- Map every imported document to a `source` item and every workshop artifact to an `output` item.
- Support six formal types: source, topic, entity, analysis, decision, and output.
- Support draft, current, needs-review, stale, and archived lifecycle states.
- Track unreviewed, indexed, summarized, compiled, and low-value maturity levels.
- Model `supports`, `extends`, `contradicts`, `supersedes`, and `opens` relations with bidirectional queries.
- Promote an answer into durable knowledge with one action, write an owned Markdown note, and connect its real source evidence. AI-generated items default to `needs-review`.
- Run health checks for orphaned items, missing provenance, empty bodies, staleness, inconsistent tags, alias collisions, and uncompiled high-value sources, with actionable recovery guidance.
- Atomically preserve a tombstone, Markdown note, aliases, tags, and graph edges before deletion; restore the original item ID and every still-valid relation from **Knowledge → Trash** without deleting source material.

### 4. Local semantic retrieval

- SQLite FTS5 full-text retrieval;
- local multilingual semantic embeddings;
- vector and BM25 candidate recall;
- Reciprocal Rank Fusion and local reranking;
- automatic final relevance sorting with clearly unrelated candidates removed;
- filters for spaces, tags, media types, folders, and exact documents;
- visible fused, semantic, and keyword-hit diagnostics.

Search does not call an LLM. The first semantic search downloads an approximately 240 MB ONNX model; subsequent searches can run offline.

### 5. Multi-turn, citation-grounded Q&A

- Continue asking follow-up questions in the same conversation.
- Persist, search, reopen, rename, delete, and export conversations as Markdown.
- Stream cloud-model answers as they are generated, with stop-and-keep-partial-output behavior.
- Copy, regenerate, and store local helpful/needs-improvement feedback for every answer.
- Paste screenshots directly into the composer, drag images into it, or select up to four images with the attachment button.
- Preview and remove attachments before sending; a vision model jointly understands text, images, and retrieved evidence.
- Refer to the previous turn's image in a follow-up without attaching it again.
- Rewrite retrieval queries using recent messages and a rolling summary.
- Require every cited claim to use evidence from the current retrieval result.
- Validate citation IDs against real local document and chunk IDs.
- Click an inline citation to open the source reader at the matching page or timestamp.
- Return an explicit insufficient-evidence response instead of inventing facts.

Context selection is adaptive: focused retrieval for ordinary questions, full context for a selected small document, and hierarchical sampling for long courses or reports. The evidence panel reports well-supported, partially supported, limited, image-only, or insufficient evidence and explains citation coverage, evidence utilization, source diversity, and the reasons behind the label. These are inspectable support metrics—not a fabricated probability that an answer is correct.

Retrieved text is always serialized as untrusted evidence data. Instruction-like content is flagged, and the answer policy explicitly forbids following commands, role overrides, prompt-exfiltration requests, or tool instructions found inside sources.

Answer providers include DeepSeek, Kimi, and a fully offline extractive evidence model. Once DeepSeek is configured, the image-capable experimental `deepseek-v4-flash-vision-exp` model is selected by default. Retrieval itself remains local and model-independent.

### 6. Source reading and knowledge management

- Read PDF pages, images, extracted text, and media timelines inside the app.
- Inspect every parsed chunk and its locator.
- Attach notes or learning cards to a specific evidence chunk.
- Rename, disable, re-enable, reparse, or remove a source.
- Organize documents using knowledge spaces and tags.
- Detect duplicate content fingerprints.
- Preserve archived originals when a searchable record is removed.

### 7. Automation and knowledge workshop

- Watch one or more folders and ingest changes incrementally.
- Persist every import batch and per-file stage, progress, error, and structured result.
- Recover interrupted work as resumable tasks after an unexpected exit and retry failed items.
- Reparse changed files only.
- Disable missing sources instead of silently destroying knowledge.
- Optionally sync from Obsidian or export AI Jingjing notes to a vault.
- Generate reports, cross-source comparisons, timelines, quizzes, flashcards, and mind-map outlines from selected evidence.

### 8. Local data security

- Keep documents, chunks, conversations, answers, evidence, and citations in local SQLite.
- Store API keys in macOS Keychain or the platform credential store when available.
- Explicitly exclude API keys from application backups.
- V2 full backups cover the database, settings, notes, original archives, retained assets, and transcripts, with per-file size and SHA-256 metadata.
- Validate paths, compression ratios, size limits, hashes, SQLite integrity, and settings before creating a safety snapshot and atomically restoring data.
- Accept only HTTPS update manifests and redirects; download packages to a temporary file and open them only after the manifest SHA-256 matches.
- Exclude personal databases, model caches, archives, credentials, and build outputs from Git.
- Detect keys, tokens, private keys, email addresses, phone numbers, absolute user paths, sensitive filenames, and image EXIF/GPS locally; image-text OCR is optional.
- Return only redacted paths, categories, and line numbers—never the matched secret or full user path.
- Build share copies from an empty-by-default selection. Governed notes, workshop outputs, and sources enter only after explicit opt-in; Source Notes and saved answers remain excluded by default because they may retain local locators.
- Permanently exclude databases, provider settings, keyrings, caches, conversations, backups, and trash from share copies; scan both before and after copying and generate a per-file SHA-256 manifest.
- Stop publication whenever a blocker or uninspected surface remains; there is no expert bypass. PDFs and modern Office packages still receive deep local inspection for a redacted risk report, but original PDF, Office/ODF, image, audio, and video containers are never copied verbatim into a safe share. The current publisher accepts only Markdown/text-like files that pass strict whole-file validation, closing unreachable-object, compressed-tail, and hidden-metadata channels. Raw containers will be reopened only after page/pixel reconstruction sanitization exists. The application creates a local copy only and never uploads or sends it.

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
| Public video platform | YouTube, Bilibili, Douyin, Xiaohongshu, X | public subtitles first; public-media transcription as fallback | original URL, subtitle, timeline |

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
        │   ├── Conversations / Evidence / Citations
        │   └── Governed Knowledge / Relations / Lifecycle
        │
        ├── KnowledgeRetriever
        │   ├── Semantic Vector + BM25 + RRF + Rerank
        │   └── Focused / Full Context / Hierarchical
        │
        └── KnowledgeQAEngine
            ├── Contextual query rewriting
            ├── DeepSeek / Kimi / Local Extractive
            ├── Untrusted-evidence boundary and prompt-injection defense
            └── Evidence construction, citation validation, quality explanation
```

A local safety layer provides redacted privacy scanning, explicit-selection share copies, a second scan of copied bytes, and SHA-256 verification. It contains no network-delivery capability.

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

Optional accelerators:

```bash
# MLX Whisper on Apple Silicon
pip install -e '.[apple-media]'

# PP-StructureV3 for tables, formulas, and multi-column scans
# First install PaddlePaddle 3.0+ for your platform: https://www.paddlepaddle.org.cn/install/quick
pip install -e '.[layout-ocr]'
```

The `full` extra includes the public-platform connector and a standalone faster-whisper runtime; source builds can add MLX with `.[apple-media]`. Official macOS Apple Silicon bundles include MLX automatically, so users do not need to install it separately. PaddleOCR remains an optional professional-layout component. `layout-ocr` includes PP-StructureV3's document-parser dependencies, while PaddlePaddle itself must match the operating system and CPU/CUDA platform. Once installed, the application automatically prefers these components, and diagnostics verify PaddleOCR, PaddleX, and PaddlePaddle together instead of reporting a shell-only installation as ready. Missing capabilities are reported in **Help → System Diagnostics** and never silently presented as success.

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

# Evaluate retrieval and citations with a local golden dataset
knowledge eval docs/golden-evaluation.example.json --top-k 10
```

Search and Q&A support scope options including `--collection`, `--tag`, `--media-type`, `--folder`, and `--document-id`.

## Local data layout

```text
AI-Jingjing/
├── archive/       Originals, web snapshots, and Source Packages
├── notes/         Source Notes, governed knowledge, saved answers, workshop artifacts
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
| Public-platform subtitle/media fetch | Yes | the public URL explicitly submitted by the user; no browser cookies, netrc, or proxy |
| Local privacy scan / safe share copy | No | none; the copy is written only to the user-selected local folder |

Model providers never receive the entire database, archive directory, or Obsidian vault.

## Tests

```bash
pip install pytest
pytest -q
```

Version 2.2.0 includes 205 automated tests plus 40 subtests covering chunking, atomic indexing, database migrations, governed knowledge, recoverable trash, graph relations and lifecycle, health checks, structured OCR and confidence gates, MLX/CUDA/CPU transcription routing and immediate cancellation, staged subtitle/derived artifacts, public-download streaming limits, HLS/live/protocol boundaries, dual raw-evidence/parse-version archive digests, content-addressed evidence and rollback, missing-evidence repair, persistent cleanup retries, model selection, real streaming output, clipboard images and multimodal requests, adaptive context, evidence quality, prompt-injection defenses, persistent conversations, resumable ingestion, citations, fail-closed PDF/Office/image privacy inspection and safe sharing, mutually exclusive database-wide operations, backup restore, secure updates, and desktop behavior.

The repository also includes a local golden-set evaluation framework for Hit Rate@K, MRR, Citation Precision, and Citation Coverage. Replace the placeholder document and chunk IDs in the example with IDs from your own library, then run `knowledge eval`; add `--retrieval-only` to evaluate retrieval without answer generation.

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

GitHub Actions runs tests on Linux, macOS, and Windows and produces unsigned macOS/Windows build artifacts. Before a public release, follow `packaging/RELEASE_SECURITY.md` for platform signing, notarization, SHA-256 manifest generation, and release verification.

## Repository layout

```text
src/media_knowledge/
├── desktop/       PySide6 application, controller, diagnostics, privacy sharing, updates
├── ingestion/     Multimodal extraction, OCR, transcription, public video, quality gate, archive
├── chunking/      Media-aware chunking
├── embedding/     Local semantic and compatible embeddings
├── retrieval/     Vector, FTS5, fusion, reranking
├── qa/            Multi-turn Q&A, evidence, prompts, citations
├── storage/       SQLite, governance, relations, vectors, conversations
└── sync/          Obsidian and watched-folder synchronization
```

## Boundaries

- Login-protected, DRM-protected, or anti-bot media pages are ingested only after the authentic body or media stream has been obtained.
- The public-platform connector uses an exact hostname allowlist and never reads cookies, browser sessions, netrc, or proxies, or bypasses login, region, or permission controls.
- AI synthesis never replaces original evidence; important pages and originals stay archived.
- The project does not bypass access control. Import only material you are authorized to process.
- No open-source license is included yet. Usage and redistribution remain subject to a future declaration by the repository owner.

## Contributing

Issues and pull requests are welcome. Changes to retrieval, quality gates, citations, or database migrations should include regression coverage and keep `pytest -q` green.
