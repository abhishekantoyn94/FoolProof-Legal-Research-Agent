"""Synonimise: LLM-driven legal terminology/query expansion.

Rebuilt from Legal-AI-Agents-Local-'s `tools.py:synonimize()` as a
provider-agnostic QueryExpansionProvider -- it wraps whatever LLMProvider the
active profile assigns to the query_expansion task (master prompt section 16),
rather than being tied to Ollama specifically.

Two things were deliberately dropped from the original: the Indian-Kanoon-only
framing (jurisdiction is now a parameter, not baked in), and the nonstandard
`ORR`/`ANDD`/`NOTT` operators -- this system's lexical search runs on
Postgres's `websearch_to_tsquery`, so expansions are built to feed that
directly (see retrieval/query_building.py), using its real `OR`/`-` syntax.
"""

from __future__ import annotations

import json
import re

from legal_research_app.logging_setup import get_logger
from legal_research_app.providers.base import (
    ExpandedQuery,
    LLMProvider,
    Message,
    ProviderHealth,
    QueryExpansionProvider,
    QueryExpansionResult,
)

logger = get_logger("synonimise")

_MAX_EXPANSIONS = 18


def _build_prompt(query: str, jurisdiction: str | None) -> str:
    jurisdiction_line = (
        f"Focus on terminology as used in {jurisdiction} legal practice."
        if jurisdiction
        else "No specific jurisdiction was given -- use general, cross-jurisdiction legal terminology "
        "and note any jurisdiction-specific variants explicitly in the text if relevant."
    )
    return f"""You expand a legal research query into alternative search terms to reduce the
chance of missing relevant documents due to terminology differences. You do NOT
answer the legal question, give a legal conclusion, or invent case names,
statute numbers, or citations that were not given to you.

Query: "{query}"

{jurisdiction_line}

Generate up to {_MAX_EXPANSIONS} alternative search terms/phrases covering:
- synonyms, formal/legal terminology, common abbreviations or acronyms
- broader terms, narrower terms, and related concepts a researcher might have used instead
- IMPORTANT -- also include a few terms targeting the *general legal principle or
  definition* the question turns on, separate from its specific facts. A document
  stating that principle may use a completely different worked example (a different
  substance, party, or number) than the question asks about, and won't share any of
  the question's specific entity names or figures. For example, for "does 20kg of
  Ganja count as a commercial quantity", also generate terms like "definition of
  commercial quantity", "meaning of commercial quantity", "quantity greater than
  specified amount", "interpretation of commercial quantity threshold" -- not just
  terms about Ganja specifically.

Respond with ONLY a JSON array (no other text, no markdown fences) of objects:
[{{"text": "...", "kind": "term"}}, {{"text": "a multi-word phrase", "kind": "phrase"}}]

"kind" is "term" for a single word, "phrase" for a multi-word exact phrase, or
"boolean_expression" only if you are combining terms with OR/AND/NOT yourself
(rare -- prefer separate entries instead)."""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _expansions_from_json(data: object) -> list[ExpandedQuery] | None:
    if not isinstance(data, list):
        return None
    expansions = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        kind = item.get("kind", "term")
        if kind not in ("term", "phrase", "boolean_expression"):
            kind = "term"
        expansions.append(ExpandedQuery(text=text, kind=kind))
    return expansions or None


def _parse_response(content: str) -> list[ExpandedQuery]:
    text = _strip_code_fence(content)

    try:
        parsed = _expansions_from_json(json.loads(text))
        if parsed is not None:
            return parsed
    except json.JSONDecodeError:
        pass

    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = _expansions_from_json(json.loads(text[start : end + 1]))
            if parsed is not None:
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("Synonimise: LLM response was not valid JSON; falling back to line-based parsing.")
    expansions = []
    for line in text.splitlines():
        cleaned = re.sub(r"^[\s\-*•]*\d*[.)]?\s*", "", line).strip().strip('"')
        if cleaned:
            expansions.append(ExpandedQuery(text=cleaned, kind="term"))
    return expansions


class SynonimiseProvider(QueryExpansionProvider):
    name = "synonimise"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def health(self) -> ProviderHealth:
        return self._llm.health()

    def expand(self, query: str, model: str, *, jurisdiction: str | None = None) -> QueryExpansionResult:
        response = self._llm.complete(
            messages=[Message(role="user", content=_build_prompt(query, jurisdiction))],
            model=model,
            temperature=0.3,
            max_tokens=1000,
        )
        expansions = _parse_response(response.content)[:_MAX_EXPANSIONS]
        return QueryExpansionResult(original_query=query, expansions=expansions)
