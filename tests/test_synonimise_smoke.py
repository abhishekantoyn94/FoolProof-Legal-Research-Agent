"""Real end-to-end Synonimise: a live local Ollama model expands a query that
does NOT literally appear in the synthetic fixture, and the expanded query
(fed through build_websearch_query into real lexical search) finds the
relevant chunk anyway -- demonstrating actual recall improvement from
terminology expansion, not just that the plumbing runs.
"""

from __future__ import annotations

import pytest

from legal_research_app.config import Settings, TaskName
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.retrieval.query_building import build_websearch_query
from legal_research_app.retrieval.types import RetrievalScope
from legal_research_app.services.document_service import DocumentService
from legal_research_app.services.search_service import SearchService


def test_synonimise_expands_query_and_improves_lexical_recall(supabase_client, test_org):
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)

    query_expansion_provider, model = registry.query_expansion_for(TaskName.QUERY_EXPANSION)
    health = query_expansion_provider.health()
    if not health.available:
        pytest.skip(f"Ollama not reachable: {health.detail}")

    project = (
        supabase_client.table("projects").insert({"org_id": test_org, "name": "Recall Test"}).execute().data[0]
    )
    DocumentService(supabase_client, registry).ingest_pdf(
        org_id=test_org,
        file_path="tests/fixtures/synthetic_legal_doc.pdf",
        filename="synthetic_legal_doc.pdf",
        project_id=project["id"],
    )
    scope = RetrievalScope(org_id=test_org, project_ids=[project["id"]])
    search = SearchService(supabase_client, registry)

    # This exact wording does not appear anywhere in the fixture (which uses
    # "infringement", "settlement", "license", "arbitration") -- a plain
    # literal search for it should find nothing.
    query = "IP rights breach dispute"
    baseline_hits = search.lexical_search(query, scope)

    expansion = query_expansion_provider.expand(query, model, jurisdiction=None)
    assert len(expansion.expansions) > 0, "expected the local model to produce at least one expansion"

    expanded_query = build_websearch_query(expansion)
    expanded_hits = search.lexical_search(expanded_query, scope)

    assert len(expanded_hits) > len(baseline_hits), (
        f"expected expansion to surface hits the literal query missed. "
        f"expansions were: {[e.text for e in expansion.expansions]}"
    )
