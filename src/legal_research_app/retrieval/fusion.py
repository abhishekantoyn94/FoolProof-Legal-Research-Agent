"""Reciprocal Rank Fusion (RRF): combines independently-ranked result lists
(dense cosine similarity, lexical ts_rank) into one ranking, using only each
result's rank position rather than its raw score. This sidesteps the problem
of dense and lexical scores living on incomparable scales (master prompt
section 22: "do not blindly concatenate", "a principled ranking/fusion strategy").
"""

from __future__ import annotations

from dataclasses import replace

from legal_research_app.retrieval.types import SearchHit

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]], *, k: int = DEFAULT_RRF_K
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchHit] = {}
    matched_by: dict[str, set[str]] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
            matched_by.setdefault(hit.chunk_id, set()).update(hit.matched_by)

    fused = [
        replace(first_seen[chunk_id], score=score, matched_by=sorted(matched_by[chunk_id]))
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused
