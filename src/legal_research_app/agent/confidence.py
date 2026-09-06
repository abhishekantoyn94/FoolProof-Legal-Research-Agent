"""Confidence scoring from observable signals only (master prompt section 61):
evidence quantity, source diversity, retrieval-method agreement, contradiction
status, and subquestion coverage. Never "ask the LLM how confident it is".
"""

from __future__ import annotations

from legal_research_app.agent.types import ConfidenceAssessment, Contradiction, EvidenceItem, Finding


def finding_evidence_strength(finding: Finding, evidence_by_index: dict[int, EvidenceItem]) -> str:
    cited = [evidence_by_index[i] for i in finding.evidence_indices if i in evidence_by_index]
    if not cited:
        return "low"

    points = min(len(cited), 3)  # up to 3 points for evidence quantity
    if len({e.document_id for e in cited}) > 1:
        points += 2  # corroborated by more than one source document
    if any(len(e.matched_by) > 1 for e in cited):
        points += 1  # confirmed by both dense and lexical retrieval

    if points >= 5:
        return "high"
    if points >= 2:
        return "medium"
    return "low"


def assess_confidence(
    *,
    findings: list[Finding],
    evidence_by_index: dict[int, EvidenceItem],
    contradictions: list[Contradiction],
    subquestion_count: int,
) -> ConfidenceAssessment:
    if not findings:
        return ConfidenceAssessment(
            level="low",
            rationale=["No findings were established from the retrieved evidence within this session."],
        )

    strengths = [finding_evidence_strength(f, evidence_by_index) for f in findings]
    high_count = strengths.count("high")
    low_count = strengths.count("low")
    rationale = [
        f"{len(findings)} finding(s) established: {high_count} with strong (multi-source) evidence, "
        f"{low_count} with weak (single-source or sparse) evidence."
    ]

    covered = {f.subquestion_index for f in findings if f.subquestion_index is not None}
    coverage_fraction = (len(covered) / subquestion_count) if subquestion_count else 1.0
    if subquestion_count:
        rationale.append(f"{len(covered)} of {subquestion_count} subquestions have at least one supporting finding.")

    contradicted_indices = {c.finding_index for c in contradictions}
    if contradicted_indices:
        rationale.append(
            f"{len(contradicted_indices)} finding(s) have unresolved contradicting evidence -- see Contradictions."
        )

    if contradicted_indices or low_count > high_count or coverage_fraction < 0.5:
        level = "low"
    elif high_count == len(findings) and coverage_fraction >= 0.8:
        level = "high"
    else:
        level = "medium"

    return ConfidenceAssessment(level=level, rationale=rationale)
