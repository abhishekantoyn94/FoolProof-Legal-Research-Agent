"""SearchService: dense + lexical retrieval, fused. Master prompt section 22.

Boolean/exact-phrase support is not a bespoke parser -- lexical_search passes
the query straight to Postgres's `websearch_to_tsquery` (see the retrieval
migration), which already understands quoted phrases, implicit AND, "OR", and
a "-" exclusion prefix. Synonimise (Phase 4) will emit queries in that syntax.
"""

from __future__ import annotations

from supabase import Client

from legal_research_app.config import TaskName
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from legal_research_app.retrieval.neighbor_term_extraction import NeighborTerm, extract_salient_neighbor_terms
from legal_research_app.retrieval.types import RetrievalScope, SearchHit


def _scope_params(scope: RetrievalScope) -> dict:
    return {
        "p_org_id": scope.org_id,
        "p_project_ids": scope.project_ids,
        "p_kb_category_ids": scope.kb_category_ids,
        "p_document_ids": scope.document_ids,
    }


class SearchService:
    def __init__(self, db: Client, registry: ProviderRegistry) -> None:
        self._db = db
        self._registry = registry

    def dense_search(self, query_text: str, scope: RetrievalScope, *, match_count: int = 20) -> list[SearchHit]:
        provider, model = self._registry.embedding_for(TaskName.EMBEDDINGS)
        query_vector = provider.embed([query_text], model)[0]
        rows = (
            self._db.rpc(
                "match_document_chunks_dense",
                {**_scope_params(scope), "p_query_embedding": query_vector, "p_match_count": match_count},
            )
            .execute()
            .data
        )
        return [
            SearchHit(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                content=row["content"],
                heading=row["heading"],
                section=row["section"],
                page_number=row["page_number"],
                score=1.0 - row["distance"],  # cosine distance -> similarity
                matched_by=["dense"],
            )
            for row in rows
        ]

    def lexical_search(self, query_text: str, scope: RetrievalScope, *, match_count: int = 20) -> list[SearchHit]:
        rows = (
            self._db.rpc(
                "match_document_chunks_lexical",
                {**_scope_params(scope), "p_query": query_text, "p_match_count": match_count},
            )
            .execute()
            .data
        )
        return [
            SearchHit(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                content=row["content"],
                heading=row["heading"],
                section=row["section"],
                page_number=row["page_number"],
                score=row["rank"],
                matched_by=["lexical"],
            )
            for row in rows
        ]

    def hybrid_search(
        self,
        query_text: str,
        scope: RetrievalScope,
        *,
        match_count: int = 10,
        candidate_pool: int = 30,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[SearchHit]:
        """Runs dense + lexical independently over a larger candidate pool,
        then fuses by rank (RRF) and truncates to match_count. A query with no
        exact terminology overlap still surfaces dense-only hits, and vice versa --
        neither signal alone silently starves the other."""
        dense_hits = self.dense_search(query_text, scope, match_count=candidate_pool)
        lexical_hits = self.lexical_search(query_text, scope, match_count=candidate_pool)
        fused = reciprocal_rank_fusion([dense_hits, lexical_hits], k=rrf_k)
        return fused[:match_count]

    def suggest_terms(
        self, query_text: str, scope: RetrievalScope, *, neighbor_count: int = 20, top_n: int = 15
    ) -> list[NeighborTerm]:
        """Prototype: real corpus wording pulled from the query's actual
        nearest-neighbor chunks, not LLM-generated synonyms -- see
        retrieval/neighbor_term_extraction.py for why. Reuses dense_search;
        no new embeddings or infrastructure needed."""
        neighbors = self.dense_search(query_text, scope, match_count=neighbor_count)
        return extract_salient_neighbor_terms([h.content for h in neighbors], top_n=top_n)
