"""Domain types for the research agent. Master prompt sections 12, 35, 60-61.

Findings/evidence/contradictions are kept here as typed Python objects and
persisted as jsonb on `research_sessions` (not normalized into their own
tables yet) -- deferred in Phase 1 until this exact shape existed; it exists
now, but a normalized schema is still not needed until multi-session querying
across findings is an actual requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ResearchMode(StrEnum):
    SIMPLE = "simple"  # single-pass: search + keyword expansion + one cross-check call
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    VERIFIED = "verified"  # DEEP + web search against primary legal sources, source-authority-aware


# Deterministic min/max search rounds per mode (master prompt section 33) --
# not left to LLM judgment alone. QUICK never loops at all ("fast retrieval and
# synthesis"); STANDARD allows a little iteration; DEEP is the full iterative
# loop. VERIFIED gets DEEP's depth plus room for a dedicated verification round
# against web-sourced primary law. Within [min, max), the LLM's gap analysis
# decides whether to continue.
MODE_MIN_ROUNDS: dict[ResearchMode, int] = {
    ResearchMode.SIMPLE: 1,  # unused by the simple pipeline (it doesn't loop), defensive default only
    ResearchMode.QUICK: 1,
    ResearchMode.STANDARD: 2,
    ResearchMode.DEEP: 3,
    ResearchMode.VERIFIED: 3,
}
MODE_MAX_ROUNDS: dict[ResearchMode, int] = {
    ResearchMode.SIMPLE: 1,
    ResearchMode.QUICK: 1,
    ResearchMode.STANDARD: 3,
    ResearchMode.DEEP: 5,
    ResearchMode.VERIFIED: 6,
}
NOVELTY_THRESHOLD = 0.2  # below this fraction of new unique chunks, stop early


class AgentState(StrEnum):
    IDLE = "idle"
    UNDERSTAND_AND_PLAN = "understand_and_plan"
    GENERATE_QUERIES = "generate_queries"
    SEARCH = "search"  # includes evidence extraction: dedup + stable index assignment
    SYNTHESIZE = "synthesize"
    CHALLENGE = "challenge"
    GAP_ANALYSIS = "gap_analysis"
    VERIFY = "verify"
    FINAL = "final"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass(frozen=True)
class CitationCandidate:
    """A case name the LLM recalled as possibly relevant -- unverified by
    construction. Never treated as evidence until independently confirmed to
    exist via a targeted search (agent/research_agent.py's citation
    verification step); see README for why this exists (a fabricated case
    citation from an LLM is not a hypothetical risk -- it happened during
    real testing of this project)."""

    case_name: str
    court: str = ""
    year: str = ""
    why_relevant: str = ""


@dataclass
class ResearchPlan:
    subquestions: list[str] = field(default_factory=list)
    search_strategies: list[str] = field(default_factory=list)
    stopping_conditions: list[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    """One retrieved+deduped chunk, given a stable per-session index so LLM
    steps can cite it by number instead of by chunk_id (anti-fabrication:
    the model can only pick from what actually exists, see agent/prompts.py).

    `source_authority` distinguishes the user's own uploaded documents from
    web evidence (ResearchMode.VERIFIED only), and web evidence itself from
    a primary source (bare statute/notification/judgment) vs a secondary one
    (commentary) -- see retrieval/source_authority.py. Findings resting only
    on a secondary source must be weighed accordingly (agent/prompts.py)."""

    index: int
    chunk_id: str
    document_id: str
    document_filename: str
    content: str
    heading: str | None
    page_number: int | None
    score: float
    matched_by: list[str]
    source_authority: str = "user_document"  # "user_document" | "primary" | "secondary"
    source_url: str | None = None  # set only for web evidence


@dataclass
class Finding:
    statement: str
    subquestion_index: int | None
    evidence_indices: list[int]


@dataclass
class Contradiction:
    finding_index: int
    conflicting_evidence_indices: list[int]
    explanation: str


@dataclass
class ConfidenceAssessment:
    level: str  # "high" | "medium" | "low"
    rationale: list[str]


@dataclass
class FinalAnswer:
    """Matches the master prompt section 60 answer format."""

    executive_answer: str
    key_findings: list[Finding]
    evidence: list[EvidenceItem]
    contradictions: list[Contradiction]
    gaps: list[str]
    research_coverage: list[str]
    confidence: ConfidenceAssessment
