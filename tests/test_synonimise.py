"""Unit tests for Synonimise's parsing/prompting and query-building logic,
using a fake LLMProvider so these run instantly with no network/model calls."""

from __future__ import annotations

import json

from legal_research_app.providers.base import (
    ExpandedQuery,
    LLMProvider,
    LLMResponse,
    Message,
    ProviderHealth,
    QueryExpansionResult,
)
from legal_research_app.providers.synonimise_provider import SynonimiseProvider, _build_prompt, _parse_response
from legal_research_app.retrieval.query_building import build_websearch_query


class FakeLLMProvider(LLMProvider):
    name = "fake"

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.last_messages: list[Message] | None = None

    def health(self) -> ProviderHealth:
        return ProviderHealth(available=True)

    def complete(self, *, messages, model, temperature=0.0, max_tokens=None) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(content=self._response_text, provider=self.name, model=model)


def test_prompt_includes_jurisdiction_when_given():
    prompt = _build_prompt("patent infringement", "US")
    assert "US" in prompt
    assert "patent infringement" in prompt


def test_prompt_notes_no_jurisdiction_when_absent():
    prompt = _build_prompt("patent infringement", None)
    assert "No specific jurisdiction was given" in prompt


def test_parse_response_happy_path_json():
    raw = json.dumps([{"text": "IP infringement", "kind": "phrase"}, {"text": "IPR", "kind": "term"}])
    expansions = _parse_response(raw)
    assert expansions == [
        ExpandedQuery(text="IP infringement", kind="phrase"),
        ExpandedQuery(text="IPR", kind="term"),
    ]


def test_parse_response_strips_markdown_code_fence():
    raw = "```json\n" + json.dumps([{"text": "patent violation", "kind": "phrase"}]) + "\n```"
    expansions = _parse_response(raw)
    assert expansions == [ExpandedQuery(text="patent violation", kind="phrase")]


def test_parse_response_extracts_json_array_from_surrounding_prose():
    raw = 'Here are the terms:\n' + json.dumps([{"text": "infringement", "kind": "term"}]) + "\nHope that helps!"
    expansions = _parse_response(raw)
    assert expansions == [ExpandedQuery(text="infringement", kind="term")]


def test_parse_response_falls_back_to_numbered_lines_when_not_json():
    raw = "1. patent infringement\n2. IP violation\n3. infringement of patent rights"
    expansions = _parse_response(raw)
    assert [e.text for e in expansions] == [
        "patent infringement",
        "IP violation",
        "infringement of patent rights",
    ]
    assert all(e.kind == "term" for e in expansions)


def test_synonimise_provider_expand_end_to_end_with_fake_llm():
    fake = FakeLLMProvider(json.dumps([{"text": "patent violation", "kind": "phrase"}]))
    provider = SynonimiseProvider(fake)

    result = provider.expand("patent infringement", model="fake-model", jurisdiction="US")

    assert result.original_query == "patent infringement"
    assert result.expansions == [ExpandedQuery(text="patent violation", kind="phrase")]
    assert "US" in fake.last_messages[0].content


def test_build_websearch_query_quotes_phrases_and_dedupes():
    result = QueryExpansionResult(
        original_query="patent infringement",
        expansions=[
            ExpandedQuery(text="patent violation", kind="phrase"),
            ExpandedQuery(text="IPR", kind="term"),
            ExpandedQuery(text="Patent Infringement", kind="phrase"),  # dup of original, different case
        ],
    )
    query = build_websearch_query(result)
    terms = query.split(" or ")

    assert '"patent infringement"' in terms  # original, as a phrase
    assert '"patent violation"' in terms  # expansion, as a phrase
    assert "IPR" in terms  # single-word term, unquoted
    # Multi-word expansions are also decomposed into individual words, since an
    # LLM expansion is a paraphrase, not a verbatim excerpt (see query_building.py).
    assert "patent" in terms
    assert "violation" in terms
    # Case-insensitive dedup: the original query's own decomposition already
    # contributes "infringement" (lowercase), so "Patent Infringement"'s later
    # decomposition is a full duplicate and contributes nothing new.
    assert terms.count('"patent infringement"') == 1
    assert terms.count("patent") == 1
    assert "infringement" in terms
    assert terms.count("infringement") == 1


def test_build_websearch_query_decomposes_multiword_expansions_into_words():
    """The concrete failure mode this guards against: an LLM paraphrase like
    "IP rights infringement" won't phrase- or AND-match a document that only
    says "patent infringement" -- but the decomposed word "infringement" will."""
    result = QueryExpansionResult(
        original_query="IP rights breach dispute",
        expansions=[ExpandedQuery(text="IP rights infringement", kind="phrase")],
    )
    query = build_websearch_query(result)
    terms = query.split(" or ")
    assert "infringement" in terms
    assert "rights" in terms
    # "IP" is <=2 chars and filtered out as insignificant, matching _significant_words.
    assert "IP" not in terms


def test_build_websearch_query_passes_through_boolean_expression_unquoted():
    result = QueryExpansionResult(
        original_query="infringement",
        expansions=[ExpandedQuery(text="patent -settlement", kind="boolean_expression")],
    )
    query = build_websearch_query(result, include_original=False)
    assert query == "patent -settlement"


def test_build_websearch_query_respects_max_terms():
    result = QueryExpansionResult(
        original_query="q",
        expansions=[ExpandedQuery(text=f"term{i}", kind="term") for i in range(10)],
    )
    query = build_websearch_query(result, max_terms=3)
    assert len(query.split(" or ")) == 3


def test_build_websearch_query_decomposes_the_original_question_too():
    """Real production failure this guards against: a question naming a
    specific entity ("...arrested with exact 20kg of Ganja...") didn't match a
    terse table row ("Ganja | 1000 gm | 20 kg") because the whole question was
    only ever added as one long implicit-AND phrase, while Synonimise's own
    expansions paraphrased the entity name away (e.g. into "cannabis").
    Decomposing the question itself guarantees "Ganja" survives as its own
    OR-term regardless of what the expansion step chose to generate."""
    result = QueryExpansionResult(
        original_query="If a person is arrested with exact 20kg of Ganja, does section 37 apply?",
        expansions=[ExpandedQuery(text="cannabis possession threshold", kind="phrase")],
    )
    query = build_websearch_query(result)
    terms = query.split(" or ")
    assert "Ganja" in terms
    assert "arrested" in terms
    assert "section" in terms
