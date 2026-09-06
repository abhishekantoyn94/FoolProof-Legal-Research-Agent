"""Real end-to-end research agent run: two synthetic documents (one amending/
contradicting the other) ingested into a project, a real question researched
via the live local Ollama model, real Supabase persistence throughout.

Assertions focus on STRUCTURAL correctness that's mechanically verifiable
regardless of a small local model's output quality: every state transition
happened in order, every cited evidence chunk_id genuinely exists in the
database (zero tolerance for fabrication), the final answer has the required
shape, and confidence is a real computed value. Whether the small model's own
prose is insightful is not something this test can or should assert.
"""

from __future__ import annotations

import pytest

from legal_research_app.agent.research_agent import ResearchAgent
from legal_research_app.agent.types import ResearchMode
from legal_research_app.config import Settings
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.services.document_service import DocumentService
from legal_research_app.services.search_service import SearchService


def _ingest_both_fixtures(db, registry, org_id, project_id):
    service = DocumentService(db, registry)
    for filename in ("synthetic_legal_doc.pdf", "synthetic_legal_doc_amendment.pdf"):
        result = service.ingest_pdf(
            org_id=org_id, file_path=f"tests/fixtures/{filename}", filename=filename, project_id=project_id
        )
        assert result.status == "indexed", result.error_message


def test_research_agent_quick_mode_end_to_end(supabase_client, test_org):
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)

    health = registry.health_report().get("ollama")
    if not health or not health.available:
        pytest.skip(f"Ollama not reachable: {health.detail if health else 'no health info'}")

    project = (
        supabase_client.table("projects").insert({"org_id": test_org, "name": "Contradiction Test"}).execute().data[0]
    )
    _ingest_both_fixtures(supabase_client, registry, test_org, project["id"])

    agent = ResearchAgent(supabase_client, registry, SearchService(supabase_client, registry))
    session_id = agent.start(
        org_id=test_org,
        question="What settlement payment amount was agreed, and does any document dispute it?",
        project_id=project["id"],
        jurisdiction=None,
        research_mode=ResearchMode.QUICK,
    )

    session = supabase_client.table("research_sessions").select("*").eq("id", session_id).execute().data[0]

    assert session["status"] == "completed", session.get("state_data")
    assert session["agent_state"] == "completed"

    trace_steps = [event["step"] for event in session["operational_trace"]]
    expected_prefix = ["plan_created", "queries_generated", "search_completed", "synthesis_completed", "challenge_completed"]
    assert trace_steps[: len(expected_prefix)] == expected_prefix
    assert trace_steps[-1] == "final_synthesis_generated"
    assert "verification_completed" in trace_steps

    final_answer = session["final_answer"]
    assert final_answer is not None
    for key in ("executive_answer", "key_findings", "evidence", "contradictions", "gaps", "research_coverage", "confidence"):
        assert key in final_answer

    assert final_answer["confidence"]["level"] in ("high", "medium", "low")
    assert final_answer["confidence"]["rationale"]

    # Anti-fabrication: every evidence item cited anywhere in the final answer
    # must correspond to a real chunk that actually exists in this org's data.
    evidence_chunk_ids = [e["chunk_id"] for e in final_answer["evidence"]]
    if evidence_chunk_ids:
        real_chunks = (
            supabase_client.table("document_chunks")
            .select("id, org_id")
            .in_("id", evidence_chunk_ids)
            .execute()
            .data
        )
        real_ids = {c["id"] for c in real_chunks}
        assert set(evidence_chunk_ids) == real_ids, "final answer cited a chunk_id that does not exist -- fabrication"
        assert all(c["org_id"] == test_org for c in real_chunks)

        # Every finding's evidence_indices must resolve to indices that exist
        # in the evidence list (structural integrity of the index mapping).
        for finding in final_answer["key_findings"]:
            assert all(0 <= i < len(final_answer["evidence"]) for i in finding["evidence_indices"])

    print("\n--- Research agent QUICK mode result (informational, not asserted) ---")
    print("Executive answer:", final_answer["executive_answer"])
    print("Findings:", [f["statement"] for f in final_answer["key_findings"]])
    print("Contradictions:", [c["explanation"] for c in final_answer["contradictions"]])
    print("Confidence:", final_answer["confidence"])
