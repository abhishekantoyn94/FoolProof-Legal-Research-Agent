from legal_research_app.agent.confidence import assess_confidence, finding_evidence_strength
from legal_research_app.agent.types import Contradiction, EvidenceItem, Finding


def _evidence(index: int, document_id: str, matched_by: list[str]) -> EvidenceItem:
    return EvidenceItem(
        index=index,
        chunk_id=f"chunk-{index}",
        document_id=document_id,
        document_filename="f.pdf",
        content="c",
        heading=None,
        page_number=1,
        score=0.9,
        matched_by=matched_by,
    )


def test_finding_with_no_evidence_is_low_strength():
    finding = Finding(statement="x", subquestion_index=0, evidence_indices=[])
    assert finding_evidence_strength(finding, {}) == "low"


def test_finding_with_single_source_single_signal_is_low_or_medium():
    evidence_by_index = {0: _evidence(0, "doc-1", ["dense"])}
    finding = Finding(statement="x", subquestion_index=0, evidence_indices=[0])
    assert finding_evidence_strength(finding, evidence_by_index) in ("low", "medium")


def test_finding_with_multiple_sources_and_dual_retrieval_is_high():
    evidence_by_index = {
        0: _evidence(0, "doc-1", ["dense", "lexical"]),
        1: _evidence(1, "doc-2", ["dense"]),
        2: _evidence(2, "doc-3", ["lexical"]),
    }
    finding = Finding(statement="x", subquestion_index=0, evidence_indices=[0, 1, 2])
    assert finding_evidence_strength(finding, evidence_by_index) == "high"


def test_no_findings_yields_low_confidence():
    result = assess_confidence(findings=[], evidence_by_index={}, contradictions=[], subquestion_count=2)
    assert result.level == "low"
    assert result.rationale


def test_full_coverage_strong_evidence_no_contradictions_is_high_confidence():
    evidence_by_index = {
        0: _evidence(0, "doc-1", ["dense", "lexical"]),
        1: _evidence(1, "doc-2", ["dense"]),
    }
    findings = [Finding(statement="x", subquestion_index=0, evidence_indices=[0, 1])]
    result = assess_confidence(
        findings=findings, evidence_by_index=evidence_by_index, contradictions=[], subquestion_count=1
    )
    assert result.level == "high"


def test_unresolved_contradiction_forces_low_confidence_even_with_strong_evidence():
    evidence_by_index = {
        0: _evidence(0, "doc-1", ["dense", "lexical"]),
        1: _evidence(1, "doc-2", ["dense"]),
    }
    findings = [Finding(statement="x", subquestion_index=0, evidence_indices=[0, 1])]
    contradictions = [Contradiction(finding_index=0, conflicting_evidence_indices=[1], explanation="conflict")]
    result = assess_confidence(
        findings=findings, evidence_by_index=evidence_by_index, contradictions=contradictions, subquestion_count=1
    )
    assert result.level == "low"
    assert any("contradict" in line.lower() for line in result.rationale)


def test_low_coverage_forces_low_confidence():
    evidence_by_index = {0: _evidence(0, "doc-1", ["dense", "lexical"])}
    findings = [Finding(statement="x", subquestion_index=0, evidence_indices=[0])]
    result = assess_confidence(
        findings=findings, evidence_by_index=evidence_by_index, contradictions=[], subquestion_count=4
    )
    assert result.level == "low"
