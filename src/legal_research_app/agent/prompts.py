"""Prompt construction + response validation for each LLM-calling agent step.

Anti-fabrication (master prompt section 31): the model is never allowed to
write a chunk_id, page number, or filename itself. It only ever picks integer
indices from a numbered evidence list this module renders; every parse
function here validates those indices against the real range and silently
drops (with a logged warning) anything out of range rather than trusting it.
"""

from __future__ import annotations

from legal_research_app.agent.types import CitationCandidate, Contradiction, EvidenceItem, Finding, ResearchPlan
from legal_research_app.logging_setup import get_logger
from legal_research_app.providers.json_parsing import LLMJSONParseError, parse_json_object

logger = get_logger("agent.prompts")

# Real failure mode observed in production use: a court order excerpt stated a
# general statutory-interpretation principle ("commercial quantity" means a
# quantity *greater than* the notified threshold) using Methamphetamine/50g as
# its worked example. Asked about Ganja/20kg, the model treated the excerpt as
# scoped to Methamphetamine only and refused to apply the same principle to a
# different substance/quantity, even though the excerpt was retrieved and
# shown to it -- a reasoning gap, not a retrieval gap. This instruction is
# reused in every prompt that asks the model to ground an answer/finding in
# evidence, so the fix isn't confined to just one mode.
LEGAL_PRINCIPLE_APPLICATION_INSTRUCTION = (
    "If an excerpt states a general legal principle, statutory definition, or rule of "
    "interpretation, apply that principle to the facts of the question even when the "
    "excerpt's own worked example involves a different substance, quantity, party, or "
    "case than the question asks about. Do not treat a general principle as limited to "
    "the specific example used to illustrate it.\n\n"
    "Pay strict attention to comparison operators in numeric thresholds: words like "
    "'greater than', 'more than', 'exceeds', or 'above' are EXCLUSIVE of the boundary "
    "value itself, while 'at least', 'not less than', or 'or more' are INCLUSIVE of it. "
    "Before concluding whether a specific quantity in the question meets a threshold, "
    "explicitly check whether the question's number is above, exactly equal to, or below "
    "the threshold, and which comparison word the excerpt actually uses -- do not assume "
    "a quantity classification without doing this comparison. An amount exactly equal to "
    "a threshold described with 'greater than'/'exceeds'/'above' does NOT meet that "
    "threshold.\n\n"
    "Worked example of this exact reasoning (unrelated to any real question, do not "
    "reuse its facts): a rule states 'a surcharge applies to any amount greater than "
    "$50,000'. A table elsewhere lists '$50,000' as the surcharge threshold for a "
    "certain category. Question: does the surcharge apply to exactly $50,000? Correct "
    "reasoning: the rule requires an amount greater than $50,000; $50,000 itself is "
    "equal to, not greater than, $50,000; therefore the surcharge does NOT apply to "
    "exactly $50,000, only to amounts above it. A wrong answer would say 'yes, it "
    "equals/exceeds the threshold' without noticing 'greater than' excludes the exact "
    "value. Apply this same careful equal-vs-greater-than check to the actual question."
)


# --------------------------------------------------------------------------
# UNDERSTAND + PLAN
# --------------------------------------------------------------------------

def build_plan_prompt(question: str, jurisdiction: str | None, research_mode: str) -> str:
    jurisdiction_line = f"Jurisdiction: {jurisdiction}." if jurisdiction else "No jurisdiction was specified."
    return f"""You are planning a legal research task over an indexed document corpus.
You do not answer the question yet -- you only plan how to research it.

Question: "{question}"
{jurisdiction_line}
Research mode: {research_mode}

Break the question into 2-5 concrete subquestions, list the search strategies
that should be used (e.g. "semantic search for concept X", "exact phrase
search for defined terms", "search for contradicting evidence"), and list
stopping conditions (signals that enough evidence has been gathered).

Respond with ONLY a JSON object, no other text:
{{"subquestions": ["...", "..."], "search_strategies": ["...", "..."], "stopping_conditions": ["...", "..."]}}"""


def parse_plan_response(content: str) -> ResearchPlan:
    try:
        data = parse_json_object(content, context="research plan")
    except LLMJSONParseError:
        return ResearchPlan(subquestions=[], search_strategies=[], stopping_conditions=[])
    return ResearchPlan(
        subquestions=[str(s).strip() for s in data.get("subquestions", []) if str(s).strip()],
        search_strategies=[str(s).strip() for s in data.get("search_strategies", []) if str(s).strip()],
        stopping_conditions=[str(s).strip() for s in data.get("stopping_conditions", []) if str(s).strip()],
    )


# --------------------------------------------------------------------------
# VERIFIED mode: candidate case-citation recall + independent verification
#
# Real incident this exists to prevent: asked to help verify an NDPS Act
# question, ChatGPT named two case citations. One ("Anil Kumar Dash v. State
# of Orissa", Orissa HC, 2015) was real and directly on point -- confirmed by
# fetching the actual judgment text. The other ("Raju Boruah v. State of
# Assam", 2020) could not be found anywhere, including a search restricted to
# indiankanoon.org specifically -- it appears fabricated. An LLM's case-law
# recall is a genuinely useful lead generator (it named the real one), but
# only if every lead is independently verified before being trusted, since it
# also names ones that don't exist with equal confidence.
# --------------------------------------------------------------------------

def build_citation_candidates_prompt(question: str, jurisdiction: str | None) -> str:
    jurisdiction_line = f"Jurisdiction: {jurisdiction}." if jurisdiction else "No jurisdiction specified."
    return f"""You are recalling specific case law that MIGHT be relevant to a legal
research question, from your own training. This is a memory-recall step, not
a final answer -- every case you name will be independently verified against
real legal databases before being used for anything. Do NOT invent a
plausible-sounding case if you don't actually recall a real one; it is far
better to name fewer cases, or none, than to guess. For each case you
genuinely recall, give the case name as precisely as you can (the party names
matter most for verification), the court and approximate year if known, and
why it might be relevant.

Question: "{question}"
{jurisdiction_line}

Respond with ONLY a JSON object, no other text:
{{"candidates": [{{"case_name": "Party A v. Party B", "court": "...", "year": "...", "why_relevant": "..."}}]}}
If you do not confidently recall any specific real case, respond with {{"candidates": []}} -- this is a normal and expected answer, not a failure."""


def parse_citation_candidates_response(content: str) -> list[CitationCandidate]:
    try:
        data = parse_json_object(content, context="citation candidates")
    except LLMJSONParseError:
        return []
    candidates = []
    for item in data.get("candidates", []):
        if not isinstance(item, dict):
            continue
        case_name = str(item.get("case_name", "")).strip()
        if not case_name:
            continue
        candidates.append(
            CitationCandidate(
                case_name=case_name,
                court=str(item.get("court", "")).strip(),
                year=str(item.get("year", "")).strip(),
                why_relevant=str(item.get("why_relevant", "")).strip(),
            )
        )
    return candidates


# --------------------------------------------------------------------------
# SIMPLE mode: one-shot answer + cross-check, no multi-round loop
# --------------------------------------------------------------------------

def build_simple_answer_prompt(question: str, evidence: list[EvidenceItem]) -> str:
    return f"""Answer the question using ONLY the numbered excerpts below. Do not use
outside knowledge and do not invent facts, sources, or page numbers not present
in these excerpts. This applies even to facts you are confident about from your own
training (e.g. specific statutory thresholds, quantities, or figures) -- if a specific
number or threshold the question needs is not stated in the excerpts below, say
explicitly that the excerpts do not state it, rather than supplying it from memory.

{LEGAL_PRINCIPLE_APPLICATION_INSTRUCTION}

After drafting your answer, cross-check it against the excerpts: for anything
in your answer that is NOT clearly supported by at least one excerpt (including
by applying a general principle from an excerpt as described above), remove it
or explicitly flag it as unsupported rather than stating it as fact.

Question: "{question}"

Evidence excerpts:
{render_evidence_list(evidence)}

Respond with ONLY a JSON object, no other text:
{{"answer": "...", "supporting_evidence_indices": [1, 3], "unsupported_or_uncertain": "note anything the excerpts don't fully support, or empty string if none"}}"""


def parse_simple_answer_response(content: str, *, evidence_count: int) -> tuple[str, list[int], str]:
    try:
        data = parse_json_object(content, context="simple answer")
    except LLMJSONParseError:
        return content.strip(), [], ""

    answer = str(data.get("answer", "")).strip()
    raw_indices = data.get("supporting_evidence_indices", [])
    valid_indices = [i for i in raw_indices if isinstance(i, int) and 0 <= i < evidence_count]
    dropped = [i for i in raw_indices if i not in valid_indices]
    if dropped:
        logger.warning("Dropped fabricated/out-of-range evidence indices from simple answer: %s", dropped)
    unsupported = str(data.get("unsupported_or_uncertain", "")).strip()
    return answer, valid_indices, unsupported


# --------------------------------------------------------------------------
# SYNTHESIZE (evidence -> findings)
# --------------------------------------------------------------------------

_SOURCE_AUTHORITY_LABELS = {
    "user_document": "YOUR DOCUMENT",
    "primary": "PRIMARY SOURCE (statute/notification/judgment)",
    "secondary": "SECONDARY SOURCE (commentary, not authoritative)",
}


def render_evidence_list(evidence: list[EvidenceItem]) -> str:
    lines = []
    for e in evidence:
        page = f", p.{e.page_number}" if e.page_number is not None else ""
        heading = f", heading: {e.heading!r}" if e.heading else ""
        authority = _SOURCE_AUTHORITY_LABELS.get(e.source_authority, e.source_authority)
        lines.append(f"[{e.index}] [{authority}] ({e.document_filename}{page}{heading}): {e.content}")
    return "\n\n".join(lines)


SOURCE_AUTHORITY_WEIGHTING_INSTRUCTION = (
    "Each excerpt is labeled YOUR DOCUMENT, PRIMARY SOURCE, or SECONDARY SOURCE. "
    "When excerpts conflict, a PRIMARY SOURCE (the bare statute/notification text, or a "
    "court's own judgment) always outweighs a SECONDARY SOURCE (commentary, a law-firm "
    "blog, a summary) -- a secondary source's paraphrase is not authoritative over what "
    "the primary source actually says, even if the secondary source states its claim "
    "more simply or confidently. If a finding would rest ONLY on a SECONDARY SOURCE with "
    "no PRIMARY SOURCE or YOUR DOCUMENT excerpt supporting it, state that explicitly and "
    "note it needs primary-source verification rather than presenting it as settled."
)


def build_synthesis_prompt(subquestions: list[str], evidence: list[EvidenceItem]) -> str:
    numbered_subquestions = "\n".join(f"{i}. {q}" for i, q in enumerate(subquestions))
    return f"""You extract evidence-grounded findings from retrieved document excerpts.
You must ONLY use the numbered excerpts below -- never invent a fact, page
number, or source that isn't in this list. If a subquestion has no supporting
excerpt, do not force a finding for it -- list its number as uncovered instead.

{LEGAL_PRINCIPLE_APPLICATION_INSTRUCTION}

{SOURCE_AUTHORITY_WEIGHTING_INSTRUCTION}

Subquestions:
{numbered_subquestions}

Evidence excerpts:
{render_evidence_list(evidence)}

For each finding, cite the excerpt number(s) that support it. Respond with
ONLY a JSON object, no other text:
{{"findings": [{{"statement": "...", "subquestion_index": 0, "evidence_indices": [1, 3]}}],
  "uncovered_subquestion_indices": [2]}}"""


def parse_synthesis_response(
    content: str, *, evidence_count: int, subquestion_count: int
) -> tuple[list[Finding], list[int]]:
    try:
        data = parse_json_object(content, context="synthesis")
    except LLMJSONParseError:
        return [], list(range(subquestion_count))

    findings: list[Finding] = []
    for item in data.get("findings", []):
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        raw_indices = item.get("evidence_indices", [])
        valid_indices = [i for i in raw_indices if isinstance(i, int) and 0 <= i < evidence_count]
        dropped = [i for i in raw_indices if i not in valid_indices]
        if dropped:
            logger.warning("Dropped fabricated/out-of-range evidence indices from finding: %s", dropped)
        if not valid_indices:
            continue  # a finding with zero real evidence is not a finding -- section 31
        subquestion_index = item.get("subquestion_index")
        if not (isinstance(subquestion_index, int) and 0 <= subquestion_index < subquestion_count):
            subquestion_index = None
        findings.append(
            Finding(statement=statement, subquestion_index=subquestion_index, evidence_indices=valid_indices)
        )

    uncovered = [
        i for i in data.get("uncovered_subquestion_indices", [])
        if isinstance(i, int) and 0 <= i < subquestion_count
    ]
    return findings, uncovered


# --------------------------------------------------------------------------
# CHALLENGE (contradictions + gaps)
# --------------------------------------------------------------------------

def build_challenge_prompt(findings: list[Finding], evidence: list[EvidenceItem]) -> str:
    numbered_findings = "\n".join(f"{i}. {f.statement}" for i, f in enumerate(findings))
    return f"""You critically review research findings for weaknesses. Actively look for:
contradicting evidence among the excerpts, findings supported by only one
source, missed terminology, gaps where a claim is needed but lacks evidence,
and a finding that hedges or says evidence is inconclusive even though an
excerpt states a general principle that could have been applied to reach a
definite answer (a general principle is not limited to the specific example
used to illustrate it -- if a finding missed that, report it as a gap). Also
check whether a finding rests only on a SECONDARY SOURCE where a PRIMARY
SOURCE or YOUR DOCUMENT excerpt is available or could resolve it -- if so,
report that the finding needs primary-source verification, and include a
targeted search term aimed at finding that primary source.
Do not resolve contradictions yourself -- report them.

{SOURCE_AUTHORITY_WEIGHTING_INSTRUCTION}

Findings:
{numbered_findings}

Evidence excerpts (same numbering as before):
{render_evidence_list(evidence)}

Respond with ONLY a JSON object, no other text:
{{"contradictions": [{{"finding_index": 0, "conflicting_evidence_indices": [5], "explanation": "..."}}],
  "gaps": ["description of a gap"],
  "additional_search_terms": ["term to search for next round"]}}"""


def parse_challenge_response(
    content: str, *, evidence_count: int, finding_count: int
) -> tuple[list[Contradiction], list[str], list[str]]:
    try:
        data = parse_json_object(content, context="challenge")
    except LLMJSONParseError:
        return [], [], []

    contradictions: list[Contradiction] = []
    for item in data.get("contradictions", []):
        if not isinstance(item, dict):
            continue
        finding_index = item.get("finding_index")
        if not (isinstance(finding_index, int) and 0 <= finding_index < finding_count):
            continue
        raw_indices = item.get("conflicting_evidence_indices", [])
        valid_indices = [i for i in raw_indices if isinstance(i, int) and 0 <= i < evidence_count]
        if not valid_indices:
            continue
        explanation = str(item.get("explanation", "")).strip()
        if not explanation:
            continue
        contradictions.append(
            Contradiction(finding_index=finding_index, conflicting_evidence_indices=valid_indices, explanation=explanation)
        )

    gaps = [str(g).strip() for g in data.get("gaps", []) if str(g).strip()]
    additional_terms = [str(t).strip() for t in data.get("additional_search_terms", []) if str(t).strip()]
    return contradictions, gaps, additional_terms


# --------------------------------------------------------------------------
# FINAL (executive summary prose only -- everything else assembled in code)
# --------------------------------------------------------------------------

def build_final_summary_prompt(
    question: str, findings: list[Finding], contradictions: list[Contradiction], gaps: list[str]
) -> str:
    findings_text = "\n".join(f"- {f.statement}" for f in findings) or "(none established)"
    contradictions_text = "\n".join(f"- {c.explanation}" for c in contradictions) or "(none identified)"
    gaps_text = "\n".join(f"- {g}" for g in gaps) or "(none identified)"
    return f"""Write a concise executive-summary paragraph (3-5 sentences) answering
this research question, based ONLY on the findings, contradictions, and gaps
below. Do not introduce any fact, source, or citation not already present here.
Use measured language -- never claim certainty the evidence doesn't support,
and explicitly note if the answer is incomplete or contested.

{LEGAL_PRINCIPLE_APPLICATION_INSTRUCTION}

You are writing a fresh summary from the findings text below, not just copying
it -- re-derive your own conclusion from what the findings actually say rather
than defaulting to whichever phrasing sounds most confident. If a finding
states a specific numeric threshold and its comparison operator (greater
than/at least/etc), your summary must respect that operator exactly; do not
describe a quantity as "exceeding" a threshold it is merely equal to.

Question: "{question}"

Findings:
{findings_text}

Contradictions:
{contradictions_text}

Gaps/limitations:
{gaps_text}

Respond with ONLY the paragraph text, no headers, no JSON."""
