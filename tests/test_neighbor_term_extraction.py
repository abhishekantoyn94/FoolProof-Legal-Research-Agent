from legal_research_app.retrieval.neighbor_term_extraction import extract_salient_neighbor_terms


def test_term_recurring_across_multiple_chunks_is_surfaced():
    chunks = [
        "The commercial quantity of ganja is specified by notification.",
        "For ganja, the commercial quantity threshold applies to bail.",
        "This case does not mention that substance at all.",
    ]
    results = extract_salient_neighbor_terms(chunks, min_chunk_count=2)
    texts = {r.text for r in results}
    assert "commercial quantity" in texts

    # Single words are excluded by default (see DEFAULT_NGRAM_SIZES docstring:
    # generic domain words swamp specific phrases on real legal corpora), but
    # the underlying recurrence mechanism still works for them when requested.
    results_with_unigrams = extract_salient_neighbor_terms(chunks, ngram_sizes=(1, 2, 3), min_chunk_count=2)
    assert "ganja" in {r.text for r in results_with_unigrams}


def test_term_appearing_in_only_one_chunk_is_excluded_by_default():
    chunks = ["A term seen only once here.", "Completely unrelated content."]
    results = extract_salient_neighbor_terms(chunks, min_chunk_count=2)
    assert results == []


def test_repetition_within_a_single_chunk_does_not_inflate_count():
    """The whole point is cross-chunk recurrence, not raw frequency -- a term
    repeated 10 times in one chunk must not outrank a term seen once each in
    two genuinely different chunks."""
    chunks = [
        "ganja ganja ganja ganja ganja ganja ganja ganja ganja ganja",
        "a completely different topic with no overlap",
    ]
    results = extract_salient_neighbor_terms(chunks, ngram_sizes=(1, 2, 3), min_chunk_count=1)
    ganja_result = next(r for r in results if r.text == "ganja")
    assert ganja_result.chunk_count == 1  # appeared in 1 chunk, not 10


def test_phrase_boundaries_trim_leading_and_trailing_stopwords():
    chunks = [
        "the commercial quantity of the specified drug",
        "commercial quantity of a specified drug is defined",
    ]
    results = extract_salient_neighbor_terms(chunks, min_chunk_count=2)
    texts = {r.text for r in results}
    assert "commercial quantity" in texts
    assert "of the" not in texts
    assert "the specified" not in texts


def test_results_ranked_by_chunk_count_then_phrase_length():
    chunks = [
        "commercial quantity applies here",
        "commercial quantity applies here too",
        "commercial quantity is defined",
    ]
    results = extract_salient_neighbor_terms(chunks, min_chunk_count=2, top_n=5)
    assert results[0].text == "commercial quantity"
    assert results[0].chunk_count == 3


def test_empty_input_returns_empty_list():
    assert extract_salient_neighbor_terms([]) == []
