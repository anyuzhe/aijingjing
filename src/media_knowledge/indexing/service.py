from __future__ import annotations

import json
from dataclasses import replace

from ..chunking import MediaAwareChunker
from ..embedding import EmbeddingProvider
from ..models import IndexReport, KnowledgeChunk, KnowledgeDocument, sha256_text, utcnow_iso
from ..storage import KnowledgeDatabase, SQLiteVectorStore, VectorStore


class IndexingService:
    def __init__(
        self,
        database: KnowledgeDatabase,
        embedding_provider: EmbeddingProvider,
        *,
        chunker: MediaAwareChunker | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.database = database
        self.embedding_provider = embedding_provider
        self.chunker = chunker or MediaAwareChunker()
        self.vector_store = vector_store or SQLiteVectorStore(
            database,
            provider=embedding_provider.name,
            model=embedding_provider.model,
        )

    def index_document(self, document: KnowledgeDocument) -> IndexReport:
        content_hash = document.content_hash()
        existing = self.database.get_document_by_source_id(document.source_id)
        if (
            existing
            and existing["content_hash"] == content_hash
            and self._document_state_matches(existing, document)
        ):
            existing_chunks = self.database.get_chunks(existing["id"])
            return IndexReport(
                document_id=existing["id"],
                source_id=document.source_id,
                status="unchanged",
                unchanged_chunks=len(existing_chunks),
            )

        if existing is None:
            duplicate = None
            if document.source.checksum:
                duplicate = self.database.get_document_by_checksum(document.source.checksum)
            duplicate = duplicate or self.database.get_document_by_content_hash(content_hash)
            if duplicate:
                return IndexReport(
                    document_id=duplicate["id"],
                    source_id=document.source_id,
                    status="duplicate",
                    unchanged_chunks=len(self.database.get_chunks(duplicate["id"])),
                    duplicate_of=duplicate["source_id"],
                )

        document_id = existing["id"] if existing else (document.document_id or f"doc-{sha256_text(document.source_id)[:20]}")
        document.document_id = document_id
        document.source = replace(document.source, document_id=document_id)
        document.updated_at = utcnow_iso()
        chunks = self.chunker.chunk(document)
        if not chunks:
            raise ValueError(f"document {document.source_id!r} has no indexable content")
        previous = self.database.get_chunks(document_id) if existing else {}
        next_keys = {chunk.chunk_key for chunk in chunks}
        removed = [row["id"] for key, row in previous.items() if key not in next_keys]
        to_embed: list[KnowledgeChunk] = []
        created = updated = unchanged = 0
        for chunk in chunks:
            old = previous.get(chunk.chunk_key)
            if old and old["content_hash"] == chunk.content_hash:
                chunk.embedding_status = old["embedding_status"]
                chunk.created_at = old["created_at"]
                unchanged += 1
            else:
                chunk.embedding_status = "pending"
                to_embed.append(chunk)
                if old:
                    updated += 1
                else:
                    created += 1

        # Embeddings are external work and may fail.  Complete that work before
        # mutating SQLite so a provider outage cannot leave an indexed document
        # pointing at evidence that the ingestion layer subsequently rolls back.
        vectors = (
            self.embedding_provider.embed([chunk.content for chunk in to_embed])
            if to_embed
            else []
        )
        if len(vectors) != len(to_embed):
            raise RuntimeError("embedding provider returned a mismatched vector count")

        # The document row, facets, chunks, source references, FTS rows and
        # embeddings form one logical index version.  Keeping every write inside
        # the same transaction means a storage failure restores the complete old
        # version (or leaves no trace at all for a new document).
        with self.database.connection:
            self.database.upsert_document(document, content_hash)
            self.database.replace_facets(document_id, document.collections, document.tags)
            self.database.delete_chunks(removed)
            for chunk in chunks:
                self.database.upsert_chunk(chunk, document.title)
            for chunk, vector in zip(to_embed, vectors):
                self.vector_store.upsert(
                    chunk.id,
                    vector,
                    provider=self.embedding_provider.name,
                    model=self.embedding_provider.model,
                    content_hash=chunk.content_hash,
                )
            self.database.sync_document_knowledge_state(document_id)

        return IndexReport(
            document_id=document_id,
            source_id=document.source_id,
            status="created" if existing is None else "updated",
            created_chunks=created,
            updated_chunks=updated,
            unchanged_chunks=unchanged,
            deleted_chunks=len(removed),
            embedded_chunks=len(to_embed),
        )

    def persist_document_without_search_index(self, document: KnowledgeDocument) -> IndexReport:
        """Persist source metadata while guaranteeing that no search index remains.

        This is used for Transcript V2 runs whose quality status is REVIEW or
        FAIL.  It deliberately avoids both chunking and embedding work and also
        removes an older searchable version when the same source is re-ingested.
        """

        content_hash = document.content_hash()
        existing = self.database.get_document_by_source_id(document.source_id)
        if existing is None:
            duplicate = None
            if document.source.checksum:
                duplicate = self.database.get_document_by_checksum(document.source.checksum)
            duplicate = duplicate or self.database.get_document_by_content_hash(content_hash)
            if duplicate:
                return IndexReport(
                    document_id=duplicate["id"],
                    source_id=document.source_id,
                    status="duplicate",
                    unchanged_chunks=len(self.database.get_chunks(duplicate["id"])),
                    duplicate_of=duplicate["source_id"],
                )

        document_id = existing["id"] if existing else (
            document.document_id or f"doc-{sha256_text(document.source_id)[:20]}"
        )
        document.document_id = document_id
        document.source = replace(document.source, document_id=document_id)
        document.updated_at = utcnow_iso()
        previous = self.database.get_chunks(document_id) if existing else {}
        unchanged = bool(
            existing
            and not previous
            and not bool(existing["enabled"])
            and self._document_state_matches(existing, document)
        )
        with self.database.connection:
            self.database.upsert_document(document, content_hash)
            self.database.replace_facets(document_id, document.collections, document.tags)
            self.database.delete_chunks(row["id"] for row in previous.values())
            self.database.connection.execute(
                "UPDATE documents SET enabled=0, updated_at=? WHERE id=?",
                (document.updated_at, document_id),
            )
            self.database.sync_document_knowledge_state(document_id)

        return IndexReport(
            document_id=document_id,
            source_id=document.source_id,
            status="created" if existing is None else ("unchanged" if unchanged else "updated"),
            deleted_chunks=len(previous),
        )

    def remove_document_search_index(self, document_id: str) -> int:
        """Remove all FTS/vector-backed chunks while retaining source facts."""

        chunks = self.database.get_chunks(document_id)
        with self.database.connection:
            removed = self.database.delete_chunks(row["id"] for row in chunks.values())
            self.database.sync_document_knowledge_state(document_id)
            return removed

    def _document_state_matches(self, existing, document: KnowledgeDocument) -> bool:
        """Return true only when content and every persisted locator/facet agree."""

        source = document.source

        def normalized(value: object) -> object:
            return None if value in {None, ""} else value

        scalar_fields = {
            "title": document.title,
            "media_type": document.media_type,
            "checksum": source.checksum,
            "original_uri": source.original_uri,
            "local_path": source.local_path,
            "obsidian_path": source.obsidian_path,
            "metadata_json": json.dumps(
                document.metadata,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        if any(
            normalized(existing[key]) != normalized(value)
            for key, value in scalar_fields.items()
        ):
            return False
        facets = self.database.document_facets(str(existing["id"]))
        return (
            sorted(set(facets["collections"]))
            == sorted({value.strip() for value in document.collections if value.strip()})
            and sorted(set(facets["tags"]))
            == sorted({value.strip() for value in document.tags if value.strip()})
        )

    def reindex(self) -> dict[str, int]:
        rows = self.database.all_chunks()
        vectors = self.embedding_provider.embed([row["content"] for row in rows]) if rows else []
        if len(vectors) != len(rows):
            raise RuntimeError("embedding provider returned a mismatched vector count")
        with self.database.connection:
            for row, vector in zip(rows, vectors):
                self.vector_store.upsert(
                    row["id"],
                    vector,
                    provider=self.embedding_provider.name,
                    model=self.embedding_provider.model,
                    content_hash=row["content_hash"],
                )
        fts_rows = self.database.rebuild_fts()
        return {"embedded_chunks": len(rows), "fts_rows": fts_rows}

    def ensure_embedding_profile(self) -> dict[str, int] | None:
        if not self.database.embeddings_need_reindex(
            self.embedding_provider.name,
            self.embedding_provider.model,
        ):
            return None
        return self.reindex()

    def delete_document(self, document_id: str) -> bool:
        return self.database.delete_document(document_id)
