"""Unit tests for parsing + chunking (no network, no DB, no model loading)."""

from legal_research_app.ingestion.chunking import chunk_blocks
from legal_research_app.ingestion.parsing import ParsedBlock, UnsupportedDocumentError, parse_pdf


def test_parse_pdf_rejects_unsupported_extension(tmp_path):
    fake_file = tmp_path / "notes.txt"
    fake_file.write_text("hello")
    try:
        parse_pdf(fake_file)
        assert False, "expected UnsupportedDocumentError"
    except UnsupportedDocumentError as exc:
        assert ".txt" in str(exc)


def test_parse_pdf_missing_file_raises_file_not_found(tmp_path):
    try:
        parse_pdf(tmp_path / "does_not_exist.pdf")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_chunk_blocks_keeps_short_sections_whole():
    blocks = [
        ParsedBlock(kind="heading", text="1. Background", page_number=1),
        ParsedBlock(kind="text", text="Short paragraph under background.", page_number=1),
        ParsedBlock(kind="heading", text="2. Terms", page_number=1),
        ParsedBlock(kind="text", text="Short paragraph under terms.", page_number=2),
    ]
    chunks = chunk_blocks(blocks, max_chars=1000)
    assert len(chunks) == 2
    assert chunks[0].heading == "1. Background"
    assert chunks[0].content == "Short paragraph under background."
    assert chunks[0].page_number == 1
    assert chunks[1].heading == "2. Terms"
    assert chunks[1].page_number == 2


def test_chunk_blocks_splits_oversized_section_with_overlap():
    long_text = "Clause. " * 400  # ~3200 chars, well over a small max_chars
    blocks = [
        ParsedBlock(kind="heading", text="1. Long Section", page_number=1),
        ParsedBlock(kind="text", text=long_text, page_number=1),
    ]
    chunks = chunk_blocks(blocks, max_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.content) <= 500 + 50  # budget + carried overlap prefix
        assert c.heading == "1. Long Section"
    # Overlap: end of one chunk should reappear at the start of the next.
    assert chunks[1].content.startswith(chunks[0].content[-50:])


def test_chunk_blocks_handles_table_blocks():
    blocks = [
        ParsedBlock(kind="heading", text="Payment Schedule", page_number=1),
        ParsedBlock(kind="text", text="See table below.", page_number=1),
        ParsedBlock(kind="table", text="| A | B |\n|---|---|\n| 1 | 2 |", page_number=1),
    ]
    chunks = chunk_blocks(blocks, max_chars=1000)
    assert len(chunks) == 1
    assert "| A | B |" in chunks[0].content


def test_chunk_blocks_empty_input_returns_no_chunks():
    assert chunk_blocks([]) == []


def test_parse_synthetic_pdf_preserves_headings_pages_and_table():
    """Real Docling parse of the checked-in synthetic (non-confidential) fixture."""
    blocks = parse_pdf("tests/fixtures/synthetic_legal_doc.pdf")

    headings = [b.text for b in blocks if b.kind == "heading"]
    assert "1. Background" in headings
    assert "2.1 Payment Schedule" in headings
    assert "3. Governing Law" in headings

    pages = {b.page_number for b in blocks}
    assert 1 in pages and 2 in pages  # multi-page provenance preserved

    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    assert "125,000" in tables[0].text

    chunks = chunk_blocks(blocks)
    governing_law_chunk = next(c for c in chunks if c.heading == "3. Governing Law")
    assert governing_law_chunk.page_number == 2
