"""Unit tests for the VERIFIED-mode web search step, using a fake
WebSearchProvider (no real Tavily calls, no network) so these are fast and
deterministic. Real end-to-end verification against the live Tavily API is a
separate smoke test, skipped without a real TAVILY_API_KEY.
"""

from __future__ import annotations

from legal_research_app.agent.research_agent import (
    WEB_CONTENT_EXCERPT_CHARS,
    WEB_MAX_QUERIES_PER_ROUND,
    ResearchAgent,
    _WorkingMemory,
)
from legal_research_app.agent.types import ResearchMode
from legal_research_app.providers.base import ProviderHealth, WebSearchProvider, WebSearchResult


class FakeWebSearchProvider(WebSearchProvider):
    name = "fake"

    def __init__(self, results_by_query: dict[str, list[WebSearchResult]]) -> None:
        self._results_by_query = results_by_query
        self.queries_received: list[str] = []

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def search(self, query: str, *, max_results: int = 8) -> list[WebSearchResult]:
        self.queries_received.append(query)
        return self._results_by_query.get(query, [])[:max_results]


class FakeRegistry:
    """Stands in for ProviderRegistry -- only web_search() is exercised here."""

    def __init__(self, web_provider: WebSearchProvider) -> None:
        self._web_provider = web_provider

    def web_search(self) -> WebSearchProvider:
        return self._web_provider


def _memory(mode: ResearchMode = ResearchMode.VERIFIED) -> _WorkingMemory:
    return _WorkingMemory(org_id="org", project_id="proj", kb_category_ids=[], jurisdiction=None, research_mode=mode)


def _result(url: str, authority: str = "primary", content: str | None = "some content") -> WebSearchResult:
    return WebSearchResult(url=url, title=f"Title for {url}", snippet="snippet", content=content, source_authority=authority)


def test_web_search_adds_evidence_tagged_with_source_authority():
    provider = FakeWebSearchProvider({"question": [_result("https://indiankanoon.org/doc/1", "primary")]})
    agent = ResearchAgent(db=None, registry=FakeRegistry(provider), search=None)
    memory = _memory()

    added = agent._step_web_search(memory, [("question", "expanded question")])

    assert added == 1
    assert len(memory.evidence) == 1
    ev = memory.evidence[0]
    assert ev.source_authority == "primary"
    assert ev.source_url == "https://indiankanoon.org/doc/1"
    assert ev.matched_by == ["web"]


def test_web_search_dedupes_urls_across_calls():
    provider = FakeWebSearchProvider({"q": [_result("https://indiankanoon.org/doc/1")]})
    agent = ResearchAgent(db=None, registry=FakeRegistry(provider), search=None)
    memory = _memory()

    agent._step_web_search(memory, [("q", "q")])
    added_second_time = agent._step_web_search(memory, [("q", "q")])

    assert added_second_time == 0  # same URL already seen
    assert len(memory.evidence) == 1


def test_web_search_caps_queries_per_round():
    queries = [f"query{i}" for i in range(WEB_MAX_QUERIES_PER_ROUND + 5)]
    provider = FakeWebSearchProvider({q: [_result(f"https://indiankanoon.org/{q}")] for q in queries})
    agent = ResearchAgent(db=None, registry=FakeRegistry(provider), search=None)
    memory = _memory()

    agent._step_web_search(memory, [(q, q) for q in queries])

    assert len(provider.queries_received) == WEB_MAX_QUERIES_PER_ROUND


def test_web_search_truncates_long_content():
    long_content = "x" * (WEB_CONTENT_EXCERPT_CHARS * 2)
    provider = FakeWebSearchProvider({"q": [_result("https://indiankanoon.org/1", content=long_content)]})
    agent = ResearchAgent(db=None, registry=FakeRegistry(provider), search=None)
    memory = _memory()

    agent._step_web_search(memory, [("q", "q")])

    assert len(memory.evidence[0].content) == WEB_CONTENT_EXCERPT_CHARS


def test_web_search_falls_back_to_snippet_when_no_full_content():
    provider = FakeWebSearchProvider({"q": [_result("https://indiankanoon.org/1", content=None)]})
    agent = ResearchAgent(db=None, registry=FakeRegistry(provider), search=None)
    memory = _memory()

    agent._step_web_search(memory, [("q", "q")])

    assert memory.evidence[0].content == "snippet"


def test_web_search_skips_results_with_empty_content_and_snippet():
    empty_result = WebSearchResult(url="https://indiankanoon.org/1", title="t", snippet="", content=None, source_authority="primary")
    provider = FakeWebSearchProvider({"q": [empty_result]})
    agent = ResearchAgent(db=None, registry=FakeRegistry(provider), search=None)
    memory = _memory()

    added = agent._step_web_search(memory, [("q", "q")])

    assert added == 0
    assert memory.evidence == []


