# V4 baseline audit and integration design

> Historical design record. The current AI静静 2.0.1 desktop architecture uses FastEmbed multilingual semantic embeddings by default and a first-party ingestion service. See [README](../README.md) for current behavior.

## What was actually present

The workspace contained no application repository, source files, database, or migrations. The existing personal `knowledge-ingestor` is a Codex Skill outside this workspace. It provides:

- multimodal acquisition instructions;
- Unified Content Bundle (UCB) schema and validation;
- deterministic archive writing;
- managed-block Obsidian writing.

It does not provide a `Document` model, persistent chunk store, embeddings, full-text index, vector index, retrieval API, or test suite. V4 therefore consumes UCB 1.0 as its upstream boundary and does not modify the Skill, archive, or vault.

## Current module relationship

```text
Existing knowledge-ingestor (V1-V3 workflow)
  extractor / visual understanding
                |
                v
       UCB 1.0 JSON boundary
                |
                v
documents.adapters -> KnowledgeDocument / SourceReference
                |
                v
      MediaAwareChunker
       | page  | slide | timestamp | headings
                |
                v
       IndexingService
       |          |
       v          v
SQLite tables   EmbeddingProvider
  + FTS5          |
       |          v
       |       VectorStore
       +-----+----+
             v
     KnowledgeRetriever
  vector top40 + BM25 top40
             |
             v
       RRF top30 -> Rerank -> top_k
             |
             v
 SearchResult + exact SourceReference
```

## Data model and safety

SQLite owns only V4 metadata and derived indexes. Original files and Obsidian notes remain in their existing locations. `documents`, `chunks`, `source_references`, and `embeddings` are separate core tables; collections and tags use mapping tables. Foreign keys use `ON DELETE CASCADE` for source references, chunks, embeddings, tags, and collections. FTS rows are removed explicitly in the same delete transaction.

Document deduplication uses an upstream checksum when present, then normalized UCB content hash. Incremental updates match stable media-aware `chunk_key` values and only re-embed new or content-changed chunks. Unchanged chunks retain their embedding.

The default embedding provider is local feature hashing. Only the configured OpenAI-compatible embedding endpoint receives chunk text, and only when explicitly selected through environment variables.

## V4 files added

```text
src/media_knowledge/
  chunking/
  documents/
  embedding/
  indexing/
  rerank/
  retrieval/
  storage/
  cli.py
  config.py
  models.py
  runtime.py
```

No V1-V3 file needs modification. V5 can consume `KnowledgeRetriever.search_knowledge` and its structured `SearchResult` without changing the storage schema.
