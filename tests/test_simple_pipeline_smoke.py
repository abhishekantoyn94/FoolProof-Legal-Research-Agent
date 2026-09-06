"""Real end-to-end test of the new SIMPLE research mode: semantic search +
keyword expansion + keyword search -> fused chunks -> one OpenAI call that
answers and cross-checks itself. First real test of the OpenAI path in this
codebase (everything else so far used local Ollama to stay free/reproducible) --
this one costs a few cents in real API calls and requires OPENAI_API_KEY.
"""

from __future__ import annotations

import pytest

from legal_research_app.config import Settings
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.services.document_service import DocumentService
from legal_research_app.services.research_service import ResearchService
from legal_research_app.services.search_service import SearchService
from legal_research_app.agent.research_agent import ResearchAgent


def test_simple_mode_end_to_end_with_openai(supabase_client, test_org):
    settings = Settings(active_profile="hybrid_openai_default")
    registry = ProviderRegistry(settings)

    health = registry.health_report().get("openai")
    if not health or not health.available:
        pytest.skip(f"OpenAI not configured: {health.detail if health else 'no health info'}")

    project = (
        supabase_client.table("projects").insert({"org_id": test_org, "name": "Simple Mode Test"}).execute().data[0]
    )
    DocumentService(supabase_client, registry).ingest_pdf(
        org_id=test_org,
        file_path="tests/fixtures/synthetic_legal_doc.pdf",
        filename="synthetic_legal_doc.pdf",
        project_id=project["id"],
    )

    agent = ResearchAgent(supabase_client, registry, SearchService(supabase_client, registry))
    session_id = agent.run_simple(
        org_id=test_org,
        question="What was the total settlement payment and how is it structured?",
        project_id=project["id"],
    )

    session = supabase_client.table("research_sessions").select("*").eq("id", session_id).execute().data[0]
    assert session["status"] == "completed"
    assert session["research_mode"] == "simple"

    trace_steps = [e["step"] for e in session["operational_trace"]]
    assert trace_steps == ["queries_generated", "search_completed", "cross_check_completed", "final_answer_ready"]

    answer = session["final_answer"]
    assert answer is not None
    assert "500,000" in answer["executive_answer"] or "500000" in answer["executive_answer"].replace(",", "")

    # Anti-fabrication: every cited evidence chunk_id must really exist.
    evidence_chunk_ids = [e["chunk_id"] for e in answer["evidence"]]
    real_chunks = (
        supabase_client.table("document_chunks").select("id").in_("id", evidence_chunk_ids).execute().data
    )
    assert {c["id"] for c in real_chunks} == set(evidence_chunk_ids)

    print("\n--- Simple mode result ---")
    print(answer["executive_answer"])
    print("Confidence:", answer["confidence"])


def test_simple_mode_uses_real_openai_model_not_local(supabase_client, test_org):
    """Confirms the response actually came from OpenAI, not a silently
    substituted provider -- checked via a real distinguishing behavior rather
    than trusting config alone."""
    settings = Settings(active_profile="hybrid_openai_default")
    registry = ProviderRegistry(settings)
    health = registry.health_report().get("openai")
    if not health or not health.available:
        pytest.skip("OpenAI not configured")

    from legal_research_app.config import TaskName
    from legal_research_app.providers.base import Message

    provider, model = registry.llm_for(TaskName.EVIDENCE_ANALYZER)
    assert provider.name == "openai"
    assert model == "gpt-4o-mini"
    response = provider.complete(messages=[Message(role="user", content="Say the word: verified")], model=model, max_tokens=5)
    assert response.provider == "openai"
    assert "verified" in response.content.lower()
