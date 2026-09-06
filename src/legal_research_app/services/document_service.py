"""DocumentService: ingest -> parse -> chunk -> embed -> persist.

One of the section-57 API-first services. This is the only place that
orchestrates the ingestion pipeline; the CLI and (later) UI both call this,
never re-implement it (master prompt section 56: don't implement core
business logic twice).

Failure handling (master prompt section 45-47): a failed ingestion marks the
document 'error' with a message and removes any partially-written chunks --
it never leaves a document silently stuck in 'processing', and never leaves
a half-indexed document contributing incomplete retrieval results.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from supabase import Client

from legal_research_app.config import TaskName
from legal_research_app.ingestion.chunking import chunk_blocks
from legal_research_app.ingestion.parsing import UnsupportedDocumentError, parse_pdf
from legal_research_app.logging_setup import get_logger
from legal_research_app.providers.registry import ProviderRegistry

logger = get_logger("document_service")

DOCUMENTS_BUCKET = "documents"


class DocumentIngestionError(RuntimeError):
    pass


def _safe_storage_key_component(name: str) -> str:
    """Supabase Storage (S3-compatible) rejects object keys containing many
    real-world filename characters -- confirmed empirically (an em-dash in a
    real filename caused a 400 InvalidKey error). Only the STORAGE KEY is
    sanitized; the human-readable `filename` stored in the documents table
    (and shown in the UI) is never altered."""
    stem, suffix = Path(name).stem, Path(name).suffix
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "file"
    safe_suffix = re.sub(r"[^A-Za-z0-9.]+", "", suffix)
    return f"{safe_stem}{safe_suffix}"


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    status: str
    chunk_count: int
    page_count: int | None
    error_message: str | None = None


class DocumentService:
    def __init__(self, db: Client, registry: ProviderRegistry, *, storage_db: Client | None = None) -> None:
        self._db = db
        self._registry = registry
        # Supabase Storage has its own policy layer, separate from table RLS;
        # no bucket policies exist for the `authenticated` role (verified
        # empirically -- a real per-user JWT client gets a 403 on upload), so
        # storage access stays a privileged/service-role operation even when
        # `db` itself is a per-user client used for everything else.
        self._storage_db = storage_db or db

    def _ensure_bucket(self) -> None:
        try:
            self._storage_db.storage.create_bucket(DOCUMENTS_BUCKET, options={"public": False})
        except Exception as exc:
            if "already exists" not in str(exc).lower() and "duplicate" not in str(exc).lower():
                raise

    def ingest_pdf(
        self,
        *,
        org_id: str,
        file_path: str | Path,
        filename: str | None = None,
        project_id: str | None = None,
        kb_category_id: str | None = None,
        uploaded_by: str | None = None,
    ) -> IngestResult:
        if (project_id is None) == (kb_category_id is None):
            raise ValueError("Exactly one of project_id or kb_category_id must be set.")

        file_path = Path(file_path)
        filename = filename or file_path.name
        safe_filename = _safe_storage_key_component(filename)
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        self._ensure_bucket()

        document = (
            self._db.table("documents")
            .insert(
                {
                    "org_id": org_id,
                    "project_id": project_id,
                    "kb_category_id": kb_category_id,
                    "filename": filename,
                    "storage_path": f"{org_id}/pending/{safe_filename}",
                    "mime_type": "application/pdf",
                    "file_hash": file_hash,
                    "status": "pending",
                    "uploaded_by": uploaded_by,
                }
            )
            .execute()
            .data[0]
        )
        document_id = document["id"]
        storage_path = f"{org_id}/{document_id}/{safe_filename}"

        try:
            self._db.table("documents").update(
                {"status": "processing", "storage_path": storage_path}
            ).eq("id", document_id).execute()

            self._storage_db.storage.from_(DOCUMENTS_BUCKET).upload(
                storage_path, file_path.read_bytes(), {"content-type": "application/pdf"}
            )

            blocks = parse_pdf(file_path)
            chunks = chunk_blocks(blocks)
            if not chunks:
                raise DocumentIngestionError(
                    "No extractable text was found in this document. "
                    "If it is a scanned document, OCR support is not yet enabled for this ingestion path."
                )

            embedding_provider, embedding_model = self._registry.embedding_for(TaskName.EMBEDDINGS)
            vectors = embedding_provider.embed([c.content for c in chunks], embedding_model)

            page_numbers = [b.page_number for b in blocks if b.page_number is not None]
            page_count = max(page_numbers) if page_numbers else None

            # Clear any prior chunks for this document (re-ingestion safety) before
            # writing the new set, so a retry never leaves duplicate/stale chunks.
            self._db.table("document_chunks").delete().eq("document_id", document_id).execute()

            rows = [
                {
                    "document_id": document_id,
                    "org_id": org_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "heading": chunk.heading,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                    "token_count": chunk.token_count,
                    "embedding": vector,
                    "embedding_model": embedding_model,
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            self._db.table("document_chunks").insert(rows).execute()

            self._db.table("documents").update(
                {"status": "indexed", "page_count": page_count, "error_message": None}
            ).eq("id", document_id).execute()

            logger.info("Ingested document %s: %d chunks, %s pages", document_id, len(chunks), page_count)
            return IngestResult(
                document_id=document_id, status="indexed", chunk_count=len(chunks), page_count=page_count
            )

        except (UnsupportedDocumentError, DocumentIngestionError, Exception) as exc:
            # Never leave a document silently stuck 'processing', and never leave
            # partially-written chunks behind for a failed run.
            self._db.table("document_chunks").delete().eq("document_id", document_id).execute()
            error_message = str(exc)
            self._db.table("documents").update(
                {"status": "error", "error_message": error_message}
            ).eq("id", document_id).execute()
            logger.error("Ingestion failed for document %s: %s", document_id, error_message)
            return IngestResult(
                document_id=document_id, status="error", chunk_count=0, page_count=None, error_message=error_message
            )
