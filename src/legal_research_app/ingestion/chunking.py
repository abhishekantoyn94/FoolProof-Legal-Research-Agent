"""Structure-aware chunking (master prompt section 24).

Groups parsed blocks by the heading they fall under, and keeps each section as
one chunk whenever it fits within the size budget -- only sub-splitting
sections that are actually too large, rather than blindly slicing every N
characters regardless of structure. Every chunk retains its heading and page
number for provenance.
"""

from __future__ import annotations

from dataclasses import dataclass

from legal_research_app.ingestion.parsing import ParsedBlock

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    content: str
    heading: str | None
    section: str | None
    page_number: int | None
    token_count: int


def _approx_token_count(text: str) -> int:
    """Rough, provider-agnostic estimate (~4 chars/token). Good enough for
    budgeting; exact counts are provider-specific and not needed here."""
    return max(1, len(text) // 4)


def _group_by_heading(blocks: list[ParsedBlock]) -> list[tuple[str | None, list[ParsedBlock]]]:
    sections: list[tuple[str | None, list[ParsedBlock]]] = []
    current_heading: str | None = None
    current_group: list[ParsedBlock] = []

    for block in blocks:
        if block.kind == "heading":
            if current_group:
                sections.append((current_heading, current_group))
            current_heading = block.text
            current_group = []
        else:
            current_group.append(block)

    if current_group:
        sections.append((current_heading, current_group))
    return sections


def _split_oversized_blocks(group: list[ParsedBlock], max_chars: int) -> list[ParsedBlock]:
    """A single block (e.g. one very long paragraph) larger than max_chars is
    hard-split on whitespace boundaries so no downstream chunk ever exceeds budget."""
    result: list[ParsedBlock] = []
    for block in group:
        if len(block.text) <= max_chars:
            result.append(block)
            continue
        words = block.text.split(" ")
        buffer = ""
        for word in words:
            candidate = f"{buffer} {word}".strip()
            if len(candidate) > max_chars and buffer:
                result.append(ParsedBlock(kind=block.kind, text=buffer, page_number=block.page_number))
                buffer = word
            else:
                buffer = candidate
        if buffer:
            result.append(ParsedBlock(kind=block.kind, text=buffer, page_number=block.page_number))
    return result


def _pack_section_into_chunks(
    group: list[ParsedBlock], max_chars: int, overlap_chars: int
) -> list[tuple[str, int | None]]:
    """Greedily packs blocks into chunks up to max_chars, carrying a small
    tail-overlap forward for context continuity across chunk boundaries."""
    group = _split_oversized_blocks(group, max_chars)
    packed: list[tuple[str, int | None]] = []
    buffer_parts: list[str] = []
    buffer_len = 0
    buffer_first_page: int | None = None
    overlap_prefix = ""

    def flush() -> None:
        nonlocal buffer_parts, buffer_len, buffer_first_page, overlap_prefix
        if not buffer_parts:
            return
        text = "\n\n".join(buffer_parts)
        packed.append((text, buffer_first_page))
        overlap_prefix = text[-overlap_chars:] if overlap_chars > 0 else ""
        buffer_parts = []
        buffer_len = 0
        buffer_first_page = None

    for block in group:
        if buffer_first_page is None:
            buffer_first_page = block.page_number
        addition_len = len(block.text) + (len(overlap_prefix) if not buffer_parts else 0)
        if buffer_len + addition_len > max_chars and buffer_parts:
            flush()
            buffer_first_page = block.page_number
        if not buffer_parts and overlap_prefix:
            buffer_parts.append(overlap_prefix)
            buffer_len += len(overlap_prefix)
        buffer_parts.append(block.text)
        buffer_len += len(block.text)

    flush()
    return packed


def chunk_blocks(
    blocks: list[ParsedBlock],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for heading, group in _group_by_heading(blocks):
        non_empty_group = [b for b in group if b.text.strip()]
        if not non_empty_group:
            continue
        for text, page in _pack_section_into_chunks(non_empty_group, max_chars, overlap_chars):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_index=idx,
                    content=text,
                    heading=heading,
                    section=heading,
                    page_number=page,
                    token_count=_approx_token_count(text),
                )
            )
            idx += 1
    return chunks
