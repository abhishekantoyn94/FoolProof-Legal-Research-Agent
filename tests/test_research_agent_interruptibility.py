"""Interruptibility (master prompt section 36), tested against the real local
Supabase for persistence but WITHOUT waiting on any real LLM call: a stop
request set before the very first step must pause before doing any work at
all -- this is the bug the manual code read caught (see README)."""

from __future__ import annotations

from legal_research_app.agent.research_agent import ResearchAgent
from legal_research_app.agent.types import AgentState, ResearchMode, ResearchPlan
from legal_research_app.config import Settings
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.services.search_service import SearchService


class _ExplodingSearchService(SearchService):
    """Stands in for a real SearchService -- if the agent ever tries to
    actually search, this fails the test loudly instead of hanging on a
    real LLM/DB call the test doesn't want to pay for."""

    def __init__(self):
        pass

    def dense_search(self, *args, **kwargs):
        raise AssertionError("should not have reached search -- stop was not honored")

    def lexical_search(self, *args, **kwargs):
        raise AssertionError("should not have reached search -- stop was not honored")


def test_stop_requested_before_first_step_pauses_without_doing_any_work(supabase_client, test_org):
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)
    agent = ResearchAgent(supabase_client, registry, _ExplodingSearchService())

    row = (
        supabase_client.table("research_sessions")
        .insert(
            {
                "org_id": test_org,
                "question": "irrelevant -- should never be researched",
                "research_mode": ResearchMode.STANDARD.value,
                "privacy_mode": "local_only",
                "provider_profile": "local_only",
                "status": "pending",
                "agent_state": AgentState.IDLE.value,
                "stop_requested": True,  # set before any work has started
                "state_data": {
                    "org_id": test_org,
                    "project_id": None,
                    "kb_category_ids": [],
                    "jurisdiction": None,
                    "research_mode": ResearchMode.STANDARD.value,
                },
            }
        )
        .execute()
        .data[0]
    )

    agent.resume(row["id"])  # must not call the planner LLM or search at all

    session = supabase_client.table("research_sessions").select("*").eq("id", row["id"]).execute().data[0]
    assert session["status"] == "paused"
    assert session["agent_state"] == "idle"
    assert session["stop_requested"] is False  # consumed, not left dangling
    assert session["operational_trace"] == []  # confirms zero steps ran


def test_resuming_a_paused_session_continues_from_where_it_stopped(supabase_client, test_org):
    """A session paused mid-loop (after planning, before search) resumes into
    GENERATE_QUERIES without re-running the planning step -- proves state_data
    (the plan) survives the pause/resume round-trip, not just the state label."""
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)
    agent = ResearchAgent(supabase_client, registry, _ExplodingSearchService())

    plan = ResearchPlan(subquestions=["Q1"], search_strategies=["semantic"], stopping_conditions=["coverage"])
    row = (
        supabase_client.table("research_sessions")
        .insert(
            {
                "org_id": test_org,
                "question": "irrelevant",
                "research_mode": ResearchMode.STANDARD.value,
                "privacy_mode": "local_only",
                "provider_profile": "local_only",
                "status": "paused",
                "agent_state": AgentState.GENERATE_QUERIES.value,
                "stop_requested": True,  # pause again immediately, before GENERATE_QUERIES runs
                "operational_trace": [{"step": "plan_created", "detail": "1 subquestion(s) identified", "at": "x"}],
                "state_data": {
                    "org_id": test_org,
                    "project_id": None,
                    "kb_category_ids": [],
                    "jurisdiction": None,
                    "research_mode": ResearchMode.STANDARD.value,
                    "plan": {
                        "subquestions": plan.subquestions,
                        "search_strategies": plan.search_strategies,
                        "stopping_conditions": plan.stopping_conditions,
                    },
                    "next_source_queries": ["irrelevant", "Q1"],
                },
            }
        )
        .execute()
        .data[0]
    )

    agent.resume(row["id"])

    session = supabase_client.table("research_sessions").select("*").eq("id", row["id"]).execute().data[0]
    assert session["status"] == "paused"
    assert session["agent_state"] == "generate_queries"  # did not skip ahead or restart
    # The plan from before the pause is still there -- it was not silently lost or redone.
    assert session["state_data"]["plan"]["subquestions"] == ["Q1"]
    assert len(session["operational_trace"]) == 1  # no new steps ran (still paused at the checkpoint)
