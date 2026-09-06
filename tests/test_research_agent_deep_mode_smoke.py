"""Verifies the one architectural guarantee that must hold regardless of LLM
quality: DEEP mode runs at least MODE_MIN_ROUNDS[DEEP] search rounds, because
research depth is a deterministic product decision (master prompt section 33),
not something left to the model's own judgment.
"""

from __future__ import annotations

import pytest

from legal_research_app.agent.research_agent import ResearchAgent
from legal_research_app.agent.types import MODE_MIN_ROUNDS, ResearchMode
from legal_research_app.config import Settings
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.services.document_service import DocumentService
from legal_research_app.services.search_service import SearchService


def test_deep_mode_runs_at_least_the_mode_minimum_rounds(supabase_client, test_org):
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)

    health = registry.health_report().get("ollama")
    if not health or not health.available:
        pytest.skip(f"Ollama not reachable: {health.detail if health else 'no health info'}")

    project = supabase_client.table("projects").insert({"org_id": test_org, "name": "Deep Mode Test"}).execute().data[0]
    DocumentService(supabase_client, registry).ingest_pdf(
        org_id=test_org,
        file_path="tests/fixtures/synthetic_legal_doc.pdf",
        filename="synthetic_legal_doc.pdf",
        project_id=project["id"],
    )

    agent = ResearchAgent(supabase_client, registry, SearchService(supabase_client, registry))
    session_id = agent.start(
        org_id=test_org,
        question="What are the terms of the settlement?",
        project_id=project["id"],
        research_mode=ResearchMode.DEEP,
    )

    session = supabase_client.table("research_sessions").select("*").eq("id", session_id).execute().data[0]
    assert session["status"] == "completed"

    search_round_count = sum(1 for e in session["operational_trace"] if e["step"] == "search_completed")
    assert search_round_count >= MODE_MIN_ROUNDS[ResearchMode.DEEP], (
        f"DEEP mode must run at least {MODE_MIN_ROUNDS[ResearchMode.DEEP]} search rounds "
        f"regardless of what the LLM's gap analysis decides; only {search_round_count} ran."
    )
