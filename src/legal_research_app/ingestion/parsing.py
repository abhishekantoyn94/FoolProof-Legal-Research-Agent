"""Docling-based structure-preserving PDF parsing (master prompt section 23).

Unlike LocalDocQA (flattens to plain markdown text) and ProDocQA (flattens to
plain text via pypdf), this keeps per-block page numbers and heading/table
structure so it survives into chunk provenance.

OCR is off by default -- most legal documents are born-digital and OCR both
slows ingestion significantly and downloads ~30MB of model weights on first
use. Pass `ocr=True` for scanned documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel

HEADING_LABELS = frozenset({DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE})
TEXT_LABELS = frozenset({DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM})


class UnsupportedDocumentError(ValueError):
    """Raised for a file format Docling can't parse. Master prompt section 26:
    never silently pretend to support a format the installed parser stack can't."""


@dataclass(frozen=True)
class ParsedBlock:
    kind: str  # "heading" | "text" | "table"
    text: str
    page_number: int | None


def _build_converter(*, ocr: bool) -> DocumentConverter:
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = ocr
    pdf_opts.do_table_structure = True
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)})


def parse_pdf(path: str | Path, *, ocr: bool = False) -> list[ParsedBlock]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise UnsupportedDocumentError(
            f"'{path.suffix}' is not supported yet. Currently supported: .pdf"
        )

    converter = _build_converter(ocr=ocr)
    result = converter.convert(str(path))
    doc = result.document

    blocks: list[ParsedBlock] = []
    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        prov = getattr(item, "prov", None)
        page_number = prov[0].page_no if prov else None

        if label in HEADING_LABELS:
            text = getattr(item, "text", "") or ""
            if text.strip():
                blocks.append(ParsedBlock(kind="heading", text=text.strip(), page_number=page_number))
        elif label in TEXT_LABELS:
            text = getattr(item, "text", "") or ""
            if text.strip():
                blocks.append(ParsedBlock(kind="text", text=text.strip(), page_number=page_number))
        elif label == DocItemLabel.TABLE:
            markdown = item.export_to_markdown(doc)
            if markdown.strip():
                blocks.append(ParsedBlock(kind="table", text=markdown.strip(), page_number=page_number))
        # Other labels (footers, headers, pictures, captions, etc.) are
        # intentionally skipped for now -- not relevant to research retrieval.

    return blocks
