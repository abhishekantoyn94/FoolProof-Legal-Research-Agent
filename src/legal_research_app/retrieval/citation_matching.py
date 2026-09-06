"""Matches an LLM-recalled case name against a real search result's title, to
decide whether a candidate citation is actually confirmed to exist.

Deterministic word-overlap, not another LLM call: the whole point of this
step is to NOT trust model judgment for the one place where trusting it
wrongly (declaring a fabricated case "confirmed") would be worst. Case name
formatting varies a lot ("v." vs "vs", added dates, "The State of X" vs
"State of X"), so this can't be exact string equality -- it requires most of
the candidate's own distinctive (party-name) words to appear in the result title.
"""

from __future__ import annotations

from legal_research_app.retrieval.query_building import significant_words

MATCH_THRESHOLD = 0.6


def titles_match(candidate_case_name: str, result_title: str) -> bool:
    candidate_words = {w.lower() for w in significant_words(candidate_case_name)}
    if not candidate_words:
        return False
    title_words = {w.lower() for w in significant_words(result_title)}
    overlap = candidate_words & title_words
    return (len(overlap) / len(candidate_words)) >= MATCH_THRESHOLD
