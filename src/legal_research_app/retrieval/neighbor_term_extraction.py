"""Prototype: surface real corpus wording from a query's nearest-neighbor
chunks, as suggested alternate search terms.

This is deliberately NOT an LLM call and NOT a generic word-embedding lookup.
Two reasons:

1. A sentence-embedding model like BGE-M3 is trained to place similar
   *passages* close together, not to act like a thesaurus for individual
   words -- nearest neighbors of a short phrase in that space tend to be
   topically related sentences, not clean synonym lists. Asking it to do
   word-level analogy work it wasn't trained for produces noisy output.

2. We don't need a separate "vector database of all words" to try this idea
   at all: every chunk is already embedded in pgvector, and dense_search
   already does the nearest-neighbor lookup. The genuinely new piece is
   small -- pull out the terms that actually recur across the real top-K
   matches for a query, rather than trusting a generative model to invent
   plausible-sounding synonyms. A term is only surfaced here if it is real
   corpus text found in more than one independent match, which is a strong,
   cheap anti-fabrication property this technique gets for free.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "and", "or", "in", "on", "for", "to", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
        "those", "at", "as", "its", "from", "it", "he", "she", "they", "which",
        "who", "whom", "shall", "will", "not", "no", "such", "any", "all",
    }
)

DEFAULT_NGRAM_SIZES = (2, 3)
"""Single words are deliberately excluded by default: in a real legal corpus
common domain words ("bail", "accused", "Act", "offence") recur across nearly
every chunk and swamp the genuinely specific phrases. Bigrams/trigrams are
where this technique actually pays off -- confirmed against the real NDPS
corpus, where 1-grams surfaced only generic noise and 2-3 grams surfaced
"commercial quantity", "Section 37", "Psychotropic Substances Act", etc."""
DEFAULT_MIN_CHUNK_COUNT = 2  # must recur across at least this many independent matches


@dataclass(frozen=True)
class NeighborTerm:
    text: str
    chunk_count: int  # how many of the nearest-neighbor chunks this term appeared in


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 1]


def _ngrams_in_chunk(text: str, ngram_sizes: tuple[int, ...]) -> set[str]:
    """All n-grams in one chunk, deduped within that chunk (so a term
    repeated many times in one long chunk doesn't outweigh a term that
    appears once each in several different chunks -- the whole point is
    cross-chunk recurrence, not raw term frequency)."""
    words = _tokenize(text)
    found: set[str] = set()
    for n in ngram_sizes:
        for i in range(len(words) - n + 1):
            span = words[i : i + n]
            if span[0].lower() in _STOPWORDS or span[-1].lower() in _STOPWORDS:
                continue  # e.g. drop "of the", "the specified" -- keep phrase boundaries meaningful
            if all(w.lower() in _STOPWORDS for w in span):
                continue
            found.add(" ".join(span))
    return found


def extract_salient_neighbor_terms(
    chunk_contents: list[str],
    *,
    ngram_sizes: tuple[int, ...] = DEFAULT_NGRAM_SIZES,
    min_chunk_count: int = DEFAULT_MIN_CHUNK_COUNT,
    top_n: int = 15,
) -> list[NeighborTerm]:
    """chunk_contents is the raw text of a query's nearest-neighbor chunks
    (e.g. from SearchService.dense_search). Returns real corpus phrases that
    recur across multiple of those chunks, ranked by how many chunks they
    appear in (ties broken toward longer, more specific phrases)."""
    chunk_counts: Counter[str] = Counter()
    for content in chunk_contents:
        for term in _ngrams_in_chunk(content, ngram_sizes):
            chunk_counts[term] += 1

    candidates = [(term, count) for term, count in chunk_counts.items() if count >= min_chunk_count]
    candidates.sort(key=lambda tc: (-tc[1], -len(tc[0])))
    return [NeighborTerm(text=term, chunk_count=count) for term, count in candidates[:top_n]]
