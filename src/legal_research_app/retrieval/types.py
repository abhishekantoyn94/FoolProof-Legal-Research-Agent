from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalScope:
    """What a search is allowed to touch. `org_id` is always required (tenant
    isolation); the rest are optional narrowing filters. None/empty means
    "no restriction on this dimension" -- e.g. a research session spanning one
    Project plus several Knowledge Base categories sets both lists."""

    org_id: str
    project_ids: list[str] | None = None
    kb_category_ids: list[str] | None = None
    document_ids: list[str] | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    content: str
    heading: str | None
    section: str | None
    page_number: int | None
    score: float
    matched_by: list[str] = field(default_factory=list)  # "dense" and/or "lexical"
