"""Unit tests for prompt parsing + the anti-fabrication index validation --
the part of the agent that must never trust the LLM blindly."""

from __future__ import annotations

import json

from legal_research_app.agent.prompts import (
    build_final_summary_prompt,
    parse_challenge_response,
    parse_plan_response,
    parse_synthesis_response,
)
from legal_research_app.agent.types import EvidenceItem, Finding


def _evidence(index: int) -> EvidenceItem:
    return EvidenceItem(
        index=index,
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index}",
        document_filename="f.pdf",
        content="content",
        heading=None,
        page_number=1,
        score=0.9,
        matched_by=["dense"],
    )


def test_parse_plan_response_happy_path():
    raw = json.dumps(
        {
            "subquestions": ["Q1", "Q2"],
            "search_strategies": ["semantic", "boolean"],
            "stopping_conditions": ["coverage reached"],
        }
    )
    plan = parse_plan_response(raw)
    assert plan.subquestions == ["Q1", "Q2"]
    assert plan.search_strategies == ["semantic", "boolean"]


def test_parse_plan_response_malformed_returns_empty_plan_not_crash():
    plan = parse_plan_response("not json at all")
    assert plan.subquestions == []


def test_parse_synthesis_drops_out_of_range_evidence_indices():
    """The core anti-fabrication guarantee: an LLM claiming evidence index 99
    (which doesn't exist) must never end up cited in a real finding."""
    raw = json.dumps(
        {
            "findings": [
                {"statement": "A fact.", "subquestion_index": 0, "evidence_indices": [0, 99]},
            ],
            "uncovered_subquestion_indices": [],
        }
    )
    findings, uncovered = parse_synthesis_response(raw, evidence_count=2, subquestion_count=1)
    assert len(findings) == 1
    assert findings[0].evidence_indices == [0]  # 99 silently dropped, not trusted


def test_parse_synthesis_drops_finding_with_zero_valid_evidence():
    """A finding citing ONLY fabricated indices must not survive at all --
    a finding with no real evidence is not a finding (section 31)."""
    raw = json.dumps({"findings": [{"statement": "Fabricated.", "evidence_indices": [50]}]})
    findings, _ = parse_synthesis_response(raw, evidence_count=2, subquestion_count=1)
    assert findings == []


def test_parse_synthesis_malformed_response_reports_all_subquestions_uncovered():
    findings, uncovered = parse_synthesis_response("garbage", evidence_count=3, subquestion_count=2)
    assert findings == []
    assert uncovered == [0, 1]


def test_parse_challenge_drops_invalid_finding_and_evidence_indices():
    raw = json.dumps(
        {
            "contradictions": [
                {"finding_index": 0, "conflicting_evidence_indices": [1], "explanation": "real conflict"},
                {"finding_index": 5, "conflicting_evidence_indices": [0], "explanation": "fabricated finding ref"},
                {"finding_index": 0, "conflicting_evidence_indices": [99], "explanation": "fabricated evidence ref"},
            ],
            "gaps": ["missing coverage of X"],
            "additional_search_terms": ["term1"],
        }
    )
    contradictions, gaps, terms = parse_challenge_response(raw, evidence_count=2, finding_count=1)
    assert len(contradictions) == 1
    assert contradictions[0].finding_index == 0
    assert contradictions[0].conflicting_evidence_indices == [1]
    assert gaps == ["missing coverage of X"]
    assert terms == ["term1"]


def test_build_final_summary_prompt_includes_all_sections():
    prompt = build_final_summary_prompt(
        "What license was granted?",
        findings=[Finding(statement="A license was granted.", subquestion_index=0, evidence_indices=[0])],
        contradictions=[],
        gaps=["no gap"],
    )
    assert "A license was granted." in prompt
    assert "(none identified)" in prompt  # contradictions section, empty
