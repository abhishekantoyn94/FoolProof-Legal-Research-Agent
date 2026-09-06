from legal_research_app.retrieval.fusion import reciprocal_rank_fusion
from legal_research_app.retrieval.types import SearchHit


def _hit(chunk_id: str, matched_by: list[str]) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc-1",
        content=f"content for {chunk_id}",
        heading=None,
        section=None,
        page_number=None,
        score=0.0,
        matched_by=matched_by,
    )


def test_item_in_both_lists_ranks_above_item_in_only_one():
    dense = [_hit("a", ["dense"]), _hit("b", ["dense"])]
    lexical = [_hit("a", ["lexical"]), _hit("c", ["lexical"])]

    fused = reciprocal_rank_fusion([dense, lexical])

    assert fused[0].chunk_id == "a"  # appears in both -> highest fused score
    assert sorted(fused[0].matched_by) == ["dense", "lexical"]
    remaining_ids = {hit.chunk_id for hit in fused[1:]}
    assert remaining_ids == {"b", "c"}


def test_first_place_beats_later_place_within_one_list():
    dense = [_hit("a", ["dense"]), _hit("b", ["dense"]), _hit("c", ["dense"])]
    fused = reciprocal_rank_fusion([dense])
    assert [h.chunk_id for h in fused] == ["a", "b", "c"]


def test_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_empty_list_alongside_populated_one():
    dense = [_hit("a", ["dense"])]
    fused = reciprocal_rank_fusion([dense, []])
    assert [h.chunk_id for h in fused] == ["a"]
