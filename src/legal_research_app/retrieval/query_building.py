"""Bridges Synonimise output into the `websearch_to_tsquery` syntax that
SearchService.lexical_search already speaks (see the Phase 3 retrieval
migration).

Multi-word expansions are OR-joined both as a quoted phrase AND decomposed
into their individual significant words. This looks redundant but isn't:
an LLM-generated expansion is a paraphrase, not a verbatim excerpt from the
target corpus (confirmed empirically -- see test_synonimise_smoke.py), so
requiring the whole phrase to match verbatim (or even co-occur, unquoted)
routinely finds nothing even when the underlying concept is well covered.
Decomposing into individual words is what actually recovers matches like a
document that says "patent infringement" when the expansion was "IP rights
infringement". Master prompt section 11 explicitly frames the goal as
recall ("reduce the probability of missing relevant evidence"), not tidiness --
downstream fusion, reranking, and evidence review are what bring precision
back, not this stage.

The ORIGINAL question gets the same decomposition, not just expansions --
confirmed as a real gap in production use: a question naming a specific
entity ("...arrested with exact 20kg of Ganja...") wasn't matching a terse
table row ("Ganja | 1000 gm | 20 kg") because the whole question was only
ever added as one long implicit-AND phrase (matching nothing) while
Synonimise's own expansions paraphrased the drug name away entirely (e.g.
into "cannabis", "narcotic substance"). Decomposing the question itself
guarantees an entity actually typed by the user survives as its own OR-term,
independent of whether the expansion step happened to preserve it.
"""

from __future__ import annotations

import re

from legal_research_app.providers.base import ExpandedQuery, QueryExpansionResult

_STOPWORDS = frozenset(
    {"a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "with", "by",
     "is", "are", "that", "this", "at", "as", "its", "be", "from"}
)


def significant_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9']+", text) if len(w) > 2 and w.lower() not in _STOPWORDS]


def _quoted_or_bare(text: str) -> str:
    return f'"{text}"' if " " in text else text


def build_websearch_query(
    result: QueryExpansionResult, *, include_original: bool = True, max_terms: int | None = None
) -> str:
    terms: list[str] = []

    if include_original:
        original = result.original_query.strip()
        terms.append(_quoted_or_bare(original))
        if " " in original:
            terms.extend(significant_words(original))

    for expansion in result.expansions:
        text = expansion.text.strip()
        if expansion.kind == "boolean_expression":
            terms.append(text)  # already in target syntax, pass through as-is
            continue
        terms.append(_quoted_or_bare(text))
        if " " in text:
            terms.extend(significant_words(text))

    if max_terms is not None:
        terms = terms[:max_terms]

    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            deduped.append(term)

    return " or ".join(deduped)
