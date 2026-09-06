"""ResearchService: the entry point a CLI/UI calls -- never re-implement the
agent loop elsewhere (master prompt section 56)."""

from __future__ import annotations

from supabase import Client

from legal_research_app.agent.research_agent import ResearchAgent
from legal_research_app.agent.types import ResearchMode
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.services.search_service import SearchService


class ResearchService:
    def __init__(self, db: Client, registry: ProviderRegistry) -> None:
        self._agent = ResearchAgent(db, registry, SearchService(db, registry))

    def start_research(
        self,
        *,
        org_id: str,
        question: str,
        project_id: str | None = None,
        kb_category_ids: list[str] | None = None,
        jurisdiction: str | None = None,
        research_mode: ResearchMode = ResearchMode.STANDARD,
        created_by: str | None = None,
    ) -> str:
        return self._agent.start(
            org_id=org_id,
            question=question,
            project_id=project_id,
            kb_category_ids=kb_category_ids,
            jurisdiction=jurisdiction,
            research_mode=research_mode,
            created_by=created_by,
        )

    def run_simple_query(
        self,
        *,
        org_id: str,
        question: str,
        project_id: str | None = None,
        kb_category_ids: list[str] | None = None,
        jurisdiction: str | None = None,
        created_by: str | None = None,
    ) -> str:
        return self._agent.run_simple(
            org_id=org_id,
            question=question,
            project_id=project_id,
            kb_category_ids=kb_category_ids,
            jurisdiction=jurisdiction,
            created_by=created_by,
        )

    def resume(self, session_id: str) -> None:
        self._agent.resume(session_id)

    def request_stop(self, session_id: str) -> None:
        self._agent.request_stop(session_id)
