"""Real end-to-end ingestion: synthetic PDF -> Docling -> chunk -> BGE-M3
embeddings -> Supabase storage + Postgres. No mocks. Skips if Supabase isn't
running. First run downloads BGE-M3 weights (~2GB) and will be slow.
"""

from __future__ import annotations

from legal_research_app.config import Settings
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.services.document_service import DocumentService


def _parse_pgvector(value) -> list[float]:
    """PostgREST returns `vector` columns as their text representation
    ("[0.1,0.2,...]"), not a native JSON array -- confirmed against the real
    local instance. Application code that needs the values in Python (rare --
    similarity search should happen in SQL) must parse this explicitly."""
    if isinstance(value, list):
        return value
    return [float(x) for x in value.strip("[]").split(",")]


def test_ingest_pdf_end_to_end(supabase_client, test_org):
    project = (
        supabase_client.table("projects")
        .insert({"org_id": test_org, "name": "Smith v. Jones"})
        .execute()
        .data[0]
    )

    settings = Settings(active_profile="hybrid_openai_default", _env_file=None)
    registry = ProviderRegistry(settings)
    service = DocumentService(supabase_client, registry)

    result = service.ingest_pdf(
        org_id=test_org,
        file_path="tests/fixtures/synthetic_legal_doc.pdf",
        filename="synthetic_legal_doc.pdf",
        project_id=project["id"],
    )

    assert result.status == "indexed", result.error_message
    assert result.chunk_count > 0
    assert result.page_count == 2

    doc_row = (
        supabase_client.table("documents").select("*").eq("id", result.document_id).execute().data[0]
    )
    assert doc_row["status"] == "indexed"
    assert doc_row["page_count"] == 2

    chunks = (
        supabase_client.table("document_chunks")
        .select("content, heading, page_number, embedding, embedding_model")
        .eq("document_id", result.document_id)
        .order("chunk_index")
        .execute()
        .data
    )
    assert len(chunks) == result.chunk_count
    assert all(c["embedding_model"] == "BAAI/bge-m3" for c in chunks)
    for c in chunks:
        c["embedding"] = _parse_pgvector(c["embedding"])
    assert all(len(c["embedding"]) == 1024 for c in chunks)  # matches the schema's vector(1024)

    # Lexical search (full-text, BM25-equivalent) finds it by exact terminology.
    lexical_hits = (
        supabase_client.table("document_chunks")
        .select("id, content")
        .eq("document_id", result.document_id)
        .text_search("content_tsv", "infringement")
        .execute()
        .data
    )
    assert len(lexical_hits) > 0

    # Dense retrieval: embed a semantically-related query and confirm the
    # nearest chunk by cosine distance is topically relevant (not asserting
    # exact chunk identity -- just that similarity search returns *something*
    # sensible from real vectors, not fabricated).
    embedding_provider, embedding_model = registry.embedding_for()
    query_vector = embedding_provider.embed(
        ["What license did Acme Corp grant regarding the patent?"], embedding_model
    )[0]
    # No retrieval-layer RPC exists yet (Phase 3 builds real fused retrieval) --
    # this just confirms the stored vectors are real and topically meaningful
    # by computing cosine distance directly against the query embedding.
    import math

    def cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return 1 - dot / (norm_a * norm_b)

    distances = [(c["content"], cosine_distance(query_vector, c["embedding"])) for c in chunks]
    distances.sort(key=lambda pair: pair[1])
    best_content, best_distance = distances[0]
    assert best_distance < 0.7  # meaningfully similar, not a random match
    assert "license" in best_content.lower() or "patent" in best_content.lower()


def test_ingest_pdf_with_special_characters_in_filename(supabase_client, test_org):
    """Regression test: a real filename with an em-dash and ampersand caused
    a 400 InvalidKey error from Supabase Storage (the raw filename was used
    directly as the storage object key). The display filename must be
    preserved exactly even though the storage key is sanitized."""
    project = supabase_client.table("projects").insert({"org_id": test_org, "name": "P1"}).execute().data[0]
    settings = Settings(active_profile="hybrid_openai_default", _env_file=None)
    registry = ProviderRegistry(settings)
    service = DocumentService(supabase_client, registry)

    tricky_name = "NDPS Act Commercial & Small Quantity Chart (All Drugs) — 2026 - Bhatt & Joshi Associates.pdf"
    result = service.ingest_pdf(
        org_id=test_org,
        file_path="tests/fixtures/synthetic_legal_doc.pdf",
        filename=tricky_name,
        project_id=project["id"],
    )

    assert result.status == "indexed", result.error_message
    doc_row = supabase_client.table("documents").select("filename, storage_path").eq("id", result.document_id).execute().data[0]
    assert doc_row["filename"] == tricky_name  # display name preserved exactly
    assert "—" not in doc_row["storage_path"]  # storage key sanitized
