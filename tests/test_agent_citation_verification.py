"""Unit tests for VERIFIED mode's citation-candidate verification step, using
fakes (no real LLM/web calls). This is the direct regression test for the real
incident that motivated it: an LLM naming one real case and one fabricated one,
and the engine must confirm the real one with real evidence while explicitly
flagging the fabricated one as unconfirmed -- never silently trusting either.
"""

from __future__ import annotations

import json

from legal_research_app.agent.research_agent import ResearchAgent, _WorkingMemory
from legal_research_app.agent.types import ResearchMode
from legal_research_app.providers.base import LLMResponse, Message, ProviderHealth, WebSearchProvider, WebSearchResult


class FakeLLM:
    name = "fake"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def complete(self, *, messages: list[Message], model, temperature=0.0, max_tokens=None) -> LLMResponse:
        return LLMResponse(content=self._response_text, provider=self.name, model=model)


class FakeWebSearch(WebSearchProvider):
    name = "fake"

    def __init__(self, results_by_query: dict[str, list[WebSearchResult]]) -> None:
        self._results_by_query = results_by_query

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def search(self, query: str, *, max_results: int = 8) -> list[WebSearchResult]:
        return self._results_by_query.get(query, [])


class FakeRegistry:
    def __init__(self, llm: FakeLLM, web: FakeWebSearch) -> None:
        self._llm = llm
        self._web = web

    def llm_for(self, task):
        return self._llm, "fake-model"

    def web_search(self):
        return self._web


def _memory() -> _WorkingMemory:
    return _WorkingMemory(org_id="org", project_id="proj", kb_category_ids=[], jurisdiction="India", research_mode=ResearchMode.VERIFIED)


def test_real_case_confirmed_and_fabricated_case_flagged():
    """The exact scenario that happened: one real candidate, one fabricated."""
    llm_response = json.dumps(
        {
            "candidates": [
                {"case_name": "Anil Kumar Dash v. State of Orissa", "court": "Orissa HC", "year": "2015", "why_relevant": "on point"},
                {"case_name": "Raju Boruah v. State of Assam", "court": "Gauhati HC", "year": "2020", "why_relevant": "on point"},
            ]
        }
    )
    web = FakeWebSearch(
        {
            '"Anil Kumar Dash v. State of Orissa"': [
                WebSearchResult(
                    url="https://indiankanoon.org/doc/176834622",
                    title="Anil Kumar Dash vs State Of Orissa on 22 September, 2015",
                    snippet="...",
                    content="the said quantity is lesser than commercial quantity",
                    source_authority="primary",
                )
            ],
            '"Raju Boruah v. State of Assam"': [],  # nothing found -- exactly what really happened
        }
    )
    agent = ResearchAgent(db=None, registry=FakeRegistry(FakeLLM(llm_response), web), search=None)
    memory = _memory()

    added = agent._step_citation_verification(memory, question="does section 37 apply to 20kg of ganja")

    assert added == 1
    assert len(memory.evidence) == 1
    confirmed = memory.evidence[0]
    assert "Anil Kumar Dash" in confirmed.document_filename
    assert confirmed.source_url == "https://indiankanoon.org/doc/176834622"
    assert confirmed.matched_by == ["citation_verified"]

    assert memory.unconfirmed_citations == ["Raju Boruah v. State of Assam"]


def test_no_candidates_recalled_is_a_normal_empty_result():
    agent = ResearchAgent(db=None, registry=FakeRegistry(FakeLLM('{"candidates": []}'), FakeWebSearch({})), search=None)
    memory = _memory()
    added = agent._step_citation_verification(memory, question="some question")
    assert added == 0
    assert memory.evidence == []
    assert memory.unconfirmed_citations == []


def test_a_search_result_that_does_not_actually_match_the_title_is_not_confirmed():
    """Guards against a false positive: a search returning SOME result for the
    query text is not the same as confirming that specific case exists."""
    llm_response = json.dumps({"candidates": [{"case_name": "Totally Fictional Case v. Nobody", "court": "", "year": "", "why_relevant": ""}]})
    web = FakeWebSearch(
        {
            '"Totally Fictional Case v. Nobody"': [
                WebSearchResult(url="https://indiankanoon.org/doc/1", title="Unrelated vs Different Party", snippet="s", content="c", source_authority="primary")
            ]
        }
    )
    agent = ResearchAgent(db=None, registry=FakeRegistry(FakeLLM(llm_response), web), search=None)
    memory = _memory()

    added = agent._step_citation_verification(memory, question="q")

    assert added == 0
    assert memory.unconfirmed_citations == ["Totally Fictional Case v. Nobody"]


def test_unconfirmed_citations_appear_explicitly_in_final_answer_gaps():
    memory = _memory()
    memory.unconfirmed_citations = ["Raju Boruah v. State of Assam"]
    agent = ResearchAgent(
        db=None,
        registry=FakeRegistry(FakeLLM('{"answer": "n/a"}'), FakeWebSearch({})),
        search=None,
    )
    # _step_final calls the LLM for the executive summary text; give it a real completion.
    agent._registry = FakeRegistry(FakeLLM("An executive summary."), FakeWebSearch({}))
    final = agent._step_final(memory, question="does section 37 apply?")
    assert any("Raju Boruah" in gap and "UNCONFIRMED" in gap for gap in final.gaps)
