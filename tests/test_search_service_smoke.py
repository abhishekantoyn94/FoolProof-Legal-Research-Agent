"""Real end-to-end retrieval: ingest the synthetic fixture into two separate
projects, then verify dense, lexical (phrase/boolean/exclusion), hybrid-fused,
and cross-project scope filtering all behave correctly against live Supabase
+ live BGE-M3. No mocks.
"""

from __future__ import annotations

from legal_research_app.config import Settings
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.retrieval.types import RetrievalScope
from legal_research_app.services.document_service import DocumentService
from legal_research_app.services.search_service import SearchService


def _ingest(db, registry, org_id, project_id):
    service = DocumentService(db, registry)
    result = service.ingest_pdf(
        org_id=org_id,
        file_path="tests/fixtures/synthetic_legal_doc.pdf",
        filename="synthetic_legal_doc.pdf",
        project_id=project_id,
    )
    assert result.status == "indexed", result.error_message
    return result


def test_retrieval_end_to_end(supabase_client, test_org):
    settings = Settings(active_profile="hybrid_openai_default", _env_file=None)
    registry = ProviderRegistry(settings)

    project_a = (
        supabase_client.table("projects").insert({"org_id": test_org, "name": "Project A"}).execute().data[0]
    )
    project_b = (
        supabase_client.table("projects").insert({"org_id": test_org, "name": "Project B"}).execute().data[0]
    )
    _ingest(supabase_client, registry, test_org, project_a["id"])

    search = SearchService(supabase_client, registry)
    scope_a = RetrievalScope(org_id=test_org, project_ids=[project_a["id"]])
    scope_b = RetrievalScope(org_id=test_org, project_ids=[project_b["id"]])

    # --- Dense retrieval: semantically related query, no exact term overlap ---
    dense_hits = search.dense_search("What did the two companies agree to resolve their dispute?", scope_a)
    assert len(dense_hits) > 0
    assert all(h.matched_by == ["dense"] for h in dense_hits)

    # --- Lexical: exact phrase ---
    phrase_hits = search.lexical_search('"patent infringement"', scope_a)
    assert any("infringement" in h.content.lower() for h in phrase_hits)

    # --- Lexical: boolean OR ---
    or_hits = search.lexical_search("arbitration OR settlement", scope_a)
    assert len(or_hits) > 0

    # --- Lexical: exclusion ---
    all_patent_hits = {h.chunk_id for h in search.lexical_search("patent", scope_a)}
    excluding_arbitration = {h.chunk_id for h in search.lexical_search("patent -arbitration", scope_a)}
    assert excluding_arbitration.issubset(all_patent_hits)
    assert len(excluding_arbitration) < len(all_patent_hits)

    # --- Hybrid fusion: a query with both exact-term and semantic signal ---
    hybrid_hits = search.hybrid_search("patent license agreement between the companies", scope_a)
    assert len(hybrid_hits) > 0
    assert hybrid_hits[0].score >= hybrid_hits[-1].score  # sorted descending

    # --- Scope isolation: Project B has no documents, must return nothing ---
    assert search.dense_search("patent infringement", scope_b) == []
    assert search.lexical_search("patent", scope_b) == []
