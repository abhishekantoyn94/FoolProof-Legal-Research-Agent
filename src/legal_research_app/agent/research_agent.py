"""ResearchAgent: the state machine described in master prompt section 35.

IDLE -> UNDERSTAND_AND_PLAN -> GENERATE_QUERIES -> SEARCH -> SYNTHESIZE ->
CHALLENGE -> GAP_ANALYSIS -> (loop to GENERATE_QUERIES, or) -> VERIFY -> FINAL

State is checkpointed to `research_sessions` after every transition (agent_state
+ state_data), so a killed process or an externally-set `stop_requested` flag
both resume cleanly from the last completed step (section 36) instead of
restarting or corrupting a partial run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from supabase import Client

from legal_research_app.agent.confidence import assess_confidence
from legal_research_app.agent.control import should_continue_research
from legal_research_app.agent.prompts import (
    build_challenge_prompt,
    build_citation_candidates_prompt,
    build_final_summary_prompt,
    build_plan_prompt,
    build_simple_answer_prompt,
    build_synthesis_prompt,
    parse_challenge_response,
    parse_citation_candidates_response,
    parse_plan_response,
    parse_simple_answer_response,
    parse_synthesis_response,
)
from legal_research_app.agent.types import (
    NOVELTY_THRESHOLD,
    AgentState,
    ConfidenceAssessment,
    Contradiction,
    EvidenceItem,
    FinalAnswer,
    Finding,
    ResearchMode,
    ResearchPlan,
)
from legal_research_app.config import TaskName
from legal_research_app.logging_setup import get_logger
from legal_research_app.providers.registry import ProviderRegistry
from legal_research_app.providers.base import Message
from legal_research_app.retrieval.citation_matching import titles_match
from legal_research_app.retrieval.fusion import reciprocal_rank_fusion
from legal_research_app.retrieval.query_building import build_websearch_query
from legal_research_app.retrieval.types import RetrievalScope
from legal_research_app.services.search_service import SearchService

logger = get_logger("agent.research_agent")

EVIDENCE_POOL_CAP = 40
MATCH_COUNT_PER_QUERY = 10
WEB_MAX_QUERIES_PER_ROUND = 3  # caps API calls/cost per round, not per session
WEB_MAX_RESULTS_PER_QUERY = 5
WEB_CONTENT_EXCERPT_CHARS = 3000  # a fetched page can be very long; keep evidence citation-sized
MAX_CITATION_CANDIDATES = 5


@dataclass
class _WorkingMemory:
    org_id: str
    project_id: str | None
    kb_category_ids: list[str]
    jurisdiction: str | None
    research_mode: ResearchMode
    plan: ResearchPlan = field(default_factory=ResearchPlan)
    next_source_queries: list[str] = field(default_factory=list)
    pending_query_pairs: list[tuple[str, str]] = field(default_factory=list)
    seen_chunk_ids: set[str] = field(default_factory=set)
    seen_urls: set[str] = field(default_factory=set)  # web evidence dedup, VERIFIED mode only
    evidence: list[EvidenceItem] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    pending_additional_terms: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    unconfirmed_citations: list[str] = field(default_factory=list)  # VERIFIED mode: named but not confirmed real
    round_number: int = 0
    last_round_novelty: float = 1.0

    def to_dict(self) -> dict:
        return {
            "org_id": self.org_id,
            "project_id": self.project_id,
            "kb_category_ids": self.kb_category_ids,
            "jurisdiction": self.jurisdiction,
            "research_mode": self.research_mode.value,
            "plan": asdict(self.plan),
            "next_source_queries": self.next_source_queries,
            "pending_query_pairs": [list(pair) for pair in self.pending_query_pairs],
            "seen_chunk_ids": sorted(self.seen_chunk_ids),
            "seen_urls": sorted(self.seen_urls),
            "evidence": [asdict(e) for e in self.evidence],
            "findings": [asdict(f) for f in self.findings],
            "contradictions": [asdict(c) for c in self.contradictions],
            "gaps": self.gaps,
            "pending_additional_terms": self.pending_additional_terms,
            "verification_notes": self.verification_notes,
            "unconfirmed_citations": self.unconfirmed_citations,
            "round_number": self.round_number,
            "last_round_novelty": self.last_round_novelty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_WorkingMemory":
        return cls(
            org_id=data["org_id"],
            project_id=data.get("project_id"),
            kb_category_ids=data.get("kb_category_ids", []),
            jurisdiction=data.get("jurisdiction"),
            research_mode=ResearchMode(data["research_mode"]),
            plan=ResearchPlan(**data.get("plan", {})) if data.get("plan") else ResearchPlan(),
            next_source_queries=data.get("next_source_queries", []),
            pending_query_pairs=[tuple(pair) for pair in data.get("pending_query_pairs", [])],
            seen_chunk_ids=set(data.get("seen_chunk_ids", [])),
            seen_urls=set(data.get("seen_urls", [])),
            evidence=[EvidenceItem(**e) for e in data.get("evidence", [])],
            findings=[Finding(**f) for f in data.get("findings", [])],
            contradictions=[Contradiction(**c) for c in data.get("contradictions", [])],
            gaps=data.get("gaps", []),
            pending_additional_terms=data.get("pending_additional_terms", []),
            verification_notes=data.get("verification_notes", []),
            unconfirmed_citations=data.get("unconfirmed_citations", []),
            round_number=data.get("round_number", 0),
            last_round_novelty=data.get("last_round_novelty", 1.0),
        )


class ResearchAgent:
    def __init__(self, db: Client, registry: ProviderRegistry, search: SearchService) -> None:
        self._db = db
        self._registry = registry
        self._search = search

    # -- public API ---------------------------------------------------------

    def start(
        self,
        *,
        org_id: str,
        question: str,
        project_id: str | None = None,
        kb_category_ids: list[str] | None = None,
        jurisdiction: str | None = None,
        research_mode: ResearchMode = ResearchMode.STANDARD,
        created_by: str | None = None,
    ) -> str:
        row = (
            self._db.table("research_sessions")
            .insert(
                {
                    "org_id": org_id,
                    "project_id": project_id,
                    "created_by": created_by,
                    "question": question,
                    "jurisdiction": jurisdiction,
                    "research_mode": research_mode.value,
                    "privacy_mode": self._registry.privacy_mode.value,
                    "provider_profile": self._registry.profile_name,
                    "status": "pending",
                    "agent_state": AgentState.IDLE.value,
                    "operational_trace": [],
                    "state_data": _WorkingMemory(
                        org_id=org_id,
                        project_id=project_id,
                        kb_category_ids=kb_category_ids or [],
                        jurisdiction=jurisdiction,
                        research_mode=research_mode,
                    ).to_dict(),
                }
            )
            .execute()
            .data[0]
        )
        session_id = row["id"]
        self.resume(session_id)
        return session_id

    def request_stop(self, session_id: str) -> None:
        self._db.table("research_sessions").update({"stop_requested": True}).eq("id", session_id).execute()

    def run_simple(
        self,
        *,
        org_id: str,
        question: str,
        project_id: str | None = None,
        kb_category_ids: list[str] | None = None,
        jurisdiction: str | None = None,
        created_by: str | None = None,
        match_count: int = 15,
    ) -> str:
        """Single-pass pipeline: semantic search + keyword expansion + keyword
        search -> fused chunks -> one LLM call that answers and cross-checks
        itself against those chunks. No planning step, no multi-round loop --
        deliberately simpler than QUICK/STANDARD/DEEP, which remain available
        for when the fuller agent loop (contradiction detection, iterative
        gap-filling) is actually wanted.
        """
        row = (
            self._db.table("research_sessions")
            .insert(
                {
                    "org_id": org_id,
                    "project_id": project_id,
                    "created_by": created_by,
                    "question": question,
                    "jurisdiction": jurisdiction,
                    "research_mode": ResearchMode.SIMPLE.value,
                    "privacy_mode": self._registry.privacy_mode.value,
                    "provider_profile": self._registry.profile_name,
                    "status": "running",
                    "agent_state": AgentState.SEARCH.value,
                }
            )
            .execute()
            .data[0]
        )
        session_id = row["id"]
        trace: list[dict] = []

        scope = RetrievalScope(
            org_id=org_id,
            project_ids=[project_id] if project_id else None,
            kb_category_ids=kb_category_ids or None,
        )

        qe_provider, qe_model = self._registry.query_expansion_for(TaskName.QUERY_EXPANSION)
        expansion = qe_provider.expand(question, qe_model, jurisdiction=jurisdiction)
        expanded_query = build_websearch_query(expansion)
        trace.append(self._trace_event("queries_generated", f"{len(expansion.expansions)} keyword variant(s) generated"))

        dense_hits = self._search.dense_search(question, scope, match_count=match_count)
        lexical_hits = self._search.lexical_search(expanded_query, scope, match_count=match_count)
        fused = reciprocal_rank_fusion([dense_hits, lexical_hits])[:match_count]
        trace.append(self._trace_event("search_completed", f"{len(fused)} chunk(s) retrieved (semantic + keyword)"))

        if not fused:
            final_answer = self._final_answer_to_dict(
                FinalAnswer(
                    executive_answer="No relevant evidence was found in the indexed corpus for this question.",
                    key_findings=[],
                    evidence=[],
                    contradictions=[],
                    gaps=["No matching chunks were retrieved by either semantic or keyword search."],
                    research_coverage=[f"0 chunk(s) via semantic search + keyword expansion ({len(expansion.expansions)} variant(s))"],
                    confidence=ConfidenceAssessment(level="low", rationale=["No evidence was retrieved."]),
                )
            )
            self._db.table("research_sessions").update(
                {"status": "completed", "agent_state": AgentState.COMPLETED.value, "operational_trace": trace, "final_answer": final_answer}
            ).eq("id", session_id).execute()
            return session_id

        document_ids = sorted({h.document_id for h in fused})
        filenames = self._document_filenames(document_ids)
        evidence = [
            EvidenceItem(
                index=i,
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                document_filename=filenames.get(h.document_id, "unknown"),
                content=h.content,
                heading=h.heading,
                page_number=h.page_number,
                score=h.score,
                matched_by=h.matched_by,
            )
            for i, h in enumerate(fused)
        ]

        provider, model = self._registry.llm_for(TaskName.EVIDENCE_ANALYZER)
        response = provider.complete(
            messages=[Message(role="user", content=build_simple_answer_prompt(question, evidence))],
            model=model,
            temperature=0.1,
            max_tokens=1000,
        )
        answer_text, cited_indices, unsupported_note = parse_simple_answer_response(
            response.content, evidence_count=len(evidence)
        )
        trace.append(
            self._trace_event(
                "cross_check_completed",
                f"answer grounded in {len(cited_indices)} excerpt(s)"
                + (f"; flagged: {unsupported_note}" if unsupported_note else ""),
            )
        )

        confidence = assess_confidence(
            findings=(
                [Finding(statement=answer_text, subquestion_index=0, evidence_indices=cited_indices)]
                if cited_indices
                else []
            ),
            evidence_by_index={e.index: e for e in evidence},
            contradictions=[],
            subquestion_count=1,
        )

        final_answer = self._final_answer_to_dict(
            FinalAnswer(
                executive_answer=answer_text,
                key_findings=[],
                evidence=evidence,
                contradictions=[],
                gaps=[unsupported_note] if unsupported_note else [],
                research_coverage=[
                    f"{len(evidence)} chunk(s) via semantic search + keyword expansion ({len(expansion.expansions)} variant(s))"
                ],
                confidence=confidence,
            )
        )
        trace.append(self._trace_event("final_answer_ready", f"confidence: {confidence.level}"))

        self._db.table("research_sessions").update(
            {"status": "completed", "agent_state": AgentState.COMPLETED.value, "operational_trace": trace, "final_answer": final_answer}
        ).eq("id", session_id).execute()
        return session_id

    def resume(self, session_id: str) -> None:
        session = self._db.table("research_sessions").select("*").eq("id", session_id).execute().data[0]
        state = AgentState(session["agent_state"])
        memory = _WorkingMemory.from_dict(session["state_data"])
        trace: list[dict] = list(session["operational_trace"])
        question = session["question"]

        def stop_requested_and_clear() -> bool:
            # A single atomic check-then-clear, used identically before the
            # first (planning) step and before every loop step -- clearing the
            # flag anywhere else would make a later check see a false "not
            # requested" for a stop that hasn't actually been honored yet.
            current = (
                self._db.table("research_sessions").select("stop_requested").eq("id", session_id).execute().data[0]
            )
            if current["stop_requested"]:
                self._db.table("research_sessions").update({"stop_requested": False}).eq("id", session_id).execute()
                return True
            return False

        if state == AgentState.IDLE:
            if stop_requested_and_clear():
                self._checkpoint(session_id, state, memory, trace, status="paused")
                return
            self._checkpoint(session_id, AgentState.IDLE, memory, trace, status="planning")
            plan = self._step_understand_and_plan(question, memory.jurisdiction, memory.research_mode)
            if not plan.subquestions:
                plan.subquestions = [question]  # never leave planning with nothing to research
            memory.plan = plan
            memory.next_source_queries = [question, *plan.subquestions]
            trace.append(self._trace_event("plan_created", f"{len(plan.subquestions)} subquestion(s) identified"))
            state = AgentState.GENERATE_QUERIES
            self._checkpoint(session_id, state, memory, trace, status="running")

        while True:
            if stop_requested_and_clear():
                self._checkpoint(session_id, state, memory, trace, status="paused")
                return

            if state == AgentState.GENERATE_QUERIES:
                memory.pending_query_pairs = self._step_generate_queries(memory)
                trace.append(
                    self._trace_event(
                        "queries_generated", f"{len(memory.pending_query_pairs)} search quer(ies) prepared"
                    )
                )
                state = AgentState.SEARCH
                self._checkpoint(session_id, state, memory, trace)

            elif state == AgentState.SEARCH:
                new_count = self._step_search(memory, memory.pending_query_pairs)
                trace.append(
                    self._trace_event(
                        "search_completed",
                        f"{new_count} new evidence item(s); {len(memory.evidence)} total; "
                        f"novelty {memory.last_round_novelty:.0%}",
                    )
                )
                state = AgentState.SYNTHESIZE
                self._checkpoint(session_id, state, memory, trace)

            elif state == AgentState.SYNTHESIZE:
                uncovered = self._step_synthesize(memory)
                trace.append(
                    self._trace_event(
                        "synthesis_completed",
                        f"{len(memory.findings)} finding(s); {len(uncovered)} subquestion(s) still uncovered",
                    )
                )
                state = AgentState.CHALLENGE
                self._checkpoint(session_id, state, memory, trace)

            elif state == AgentState.CHALLENGE:
                self._step_challenge(memory)
                trace.append(
                    self._trace_event(
                        "challenge_completed",
                        f"{len(memory.contradictions)} contradiction(s), {len(memory.gaps)} gap(s) identified",
                    )
                )
                state = AgentState.GAP_ANALYSIS
                self._checkpoint(session_id, state, memory, trace)

            elif state == AgentState.GAP_ANALYSIS:
                completed_rounds = memory.round_number + 1
                continue_research = should_continue_research(
                    mode=memory.research_mode,
                    completed_rounds=completed_rounds,
                    has_gaps_or_new_terms=bool(memory.gaps or memory.pending_additional_terms),
                    last_round_novelty=memory.last_round_novelty,
                    novelty_threshold=NOVELTY_THRESHOLD,
                )
                if continue_research:
                    memory.round_number += 1
                    memory.next_source_queries = memory.pending_additional_terms or memory.gaps or [question]
                    trace.append(self._trace_event("more_research_needed", f"starting round {memory.round_number + 1}"))
                    state = AgentState.GENERATE_QUERIES
                else:
                    trace.append(self._trace_event("research_sufficient", f"stopping after {completed_rounds} round(s)"))
                    state = AgentState.VERIFY
                self._checkpoint(session_id, state, memory, trace)

            elif state == AgentState.VERIFY:
                self._step_verify(memory, question)
                trace.append(self._trace_event("verification_completed", f"{len(memory.verification_notes)} note(s)"))
                state = AgentState.FINAL
                self._checkpoint(session_id, state, memory, trace)

            elif state == AgentState.FINAL:
                final_answer = self._step_final(memory, question)
                trace.append(self._trace_event("final_synthesis_generated", f"confidence: {final_answer.confidence.level}"))
                self._checkpoint(
                    session_id,
                    AgentState.COMPLETED,
                    memory,
                    trace,
                    status="completed",
                    final_answer=self._final_answer_to_dict(final_answer),
                )
                return

    # -- state steps ----------------------------------------------------

    def _step_understand_and_plan(self, question: str, jurisdiction: str | None, mode: ResearchMode) -> ResearchPlan:
        provider, model = self._registry.llm_for(TaskName.PLANNER)
        response = provider.complete(
            messages=[Message(role="user", content=build_plan_prompt(question, jurisdiction, mode.value))],
            model=model,
            temperature=0.2,
            max_tokens=800,
        )
        return parse_plan_response(response.content)

    def _step_generate_queries(self, memory: _WorkingMemory) -> list[tuple[str, str]]:
        qe_provider, qe_model = self._registry.query_expansion_for(TaskName.QUERY_EXPANSION)
        pairs = []
        for source_query in memory.next_source_queries:
            expansion = qe_provider.expand(source_query, qe_model, jurisdiction=memory.jurisdiction)
            pairs.append((source_query, build_websearch_query(expansion)))
        return pairs

    def _step_search(self, memory: _WorkingMemory, query_pairs: list[tuple[str, str]]) -> int:
        scope = RetrievalScope(
            org_id=memory.org_id,
            project_ids=[memory.project_id] if memory.project_id else None,
            kb_category_ids=memory.kb_category_ids or None,
        )
        ranked_lists = []
        for natural_query, expanded_query in query_pairs:
            ranked_lists.append(self._search.dense_search(natural_query, scope, match_count=MATCH_COUNT_PER_QUERY))
            ranked_lists.append(self._search.lexical_search(expanded_query, scope, match_count=MATCH_COUNT_PER_QUERY))

        fused = reciprocal_rank_fusion(ranked_lists) if ranked_lists else []
        new_hits = [h for h in fused if h.chunk_id not in memory.seen_chunk_ids]
        memory.last_round_novelty = (len(new_hits) / len(fused)) if fused else 0.0

        room_left = max(EVIDENCE_POOL_CAP - len(memory.evidence), 0)
        hits_to_add = new_hits[:room_left]

        document_ids = sorted({h.document_id for h in hits_to_add})
        filenames = self._document_filenames(document_ids)

        for hit in hits_to_add:
            memory.evidence.append(
                EvidenceItem(
                    index=len(memory.evidence),
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    document_filename=filenames.get(hit.document_id, "unknown"),
                    content=hit.content,
                    heading=hit.heading,
                    page_number=hit.page_number,
                    score=hit.score,
                    matched_by=hit.matched_by,
                )
            )
            memory.seen_chunk_ids.add(hit.chunk_id)

        web_added = 0
        if memory.research_mode == ResearchMode.VERIFIED:
            web_added = self._step_web_search(memory, query_pairs)
            if memory.round_number == 0:
                # Case-name recall is a one-time lead-generation step, not
                # informed by new evidence in later rounds -- run once.
                web_added += self._step_citation_verification(memory, question=query_pairs[0][0] if query_pairs else "")

        return len(hits_to_add) + web_added

    def _step_citation_verification(self, memory: _WorkingMemory, question: str) -> int:
        """VERIFIED mode, round 0 only. Asks the LLM to recall specific case
        names that might be relevant, then independently verifies each one via
        a targeted search before it can ever become evidence -- see the module
        docstring in agent/prompts.py for the real incident (a fabricated
        citation) this exists to catch. A candidate that can't be confirmed is
        recorded in memory.unconfirmed_citations and surfaced as an explicit
        gap in the final answer; it is never silently dropped or silently trusted.
        """
        if not question:
            return 0
        provider, model = self._registry.llm_for(TaskName.QUERY_EXPANSION)
        response = provider.complete(
            messages=[Message(role="user", content=build_citation_candidates_prompt(question, memory.jurisdiction))],
            model=model,
            temperature=0.2,
            max_tokens=800,
        )
        candidates = parse_citation_candidates_response(response.content)[:MAX_CITATION_CANDIDATES]
        if not candidates:
            return 0

        web_search = self._registry.web_search()
        added = 0
        for candidate in candidates:
            try:
                results = web_search.search(f'"{candidate.case_name}"', max_results=3)
            except Exception as exc:
                logger.warning("Citation verification search failed for %r: %s", candidate.case_name, exc)
                results = []

            match = next((r for r in results if titles_match(candidate.case_name, r.title)), None)
            if match is None:
                memory.unconfirmed_citations.append(candidate.case_name)
                continue
            if match.url in memory.seen_urls:
                continue
            content = (match.content or match.snippet or "")[:WEB_CONTENT_EXCERPT_CHARS]
            if not content.strip():
                memory.unconfirmed_citations.append(candidate.case_name)
                continue
            memory.evidence.append(
                EvidenceItem(
                    index=len(memory.evidence),
                    chunk_id=match.url,
                    document_id=match.url,
                    document_filename=f"{candidate.case_name} (confirmed: {match.title})",
                    content=content,
                    heading=None,
                    page_number=None,
                    score=0.0,
                    matched_by=["citation_verified"],
                    source_authority=match.source_authority,
                    source_url=match.url,
                )
            )
            memory.seen_urls.add(match.url)
            added += 1

        return added

    def _step_web_search(self, memory: _WorkingMemory, query_pairs: list[tuple[str, str]]) -> int:
        """VERIFIED mode only: a second evidence channel searching an
        allowlisted set of primary Indian legal sources (indiankanoon.org,
        sci.gov.in, gov.in/nic.in) plus a few established secondary ones,
        each result tagged with its source authority so synthesis/challenge
        can weight a primary source over a secondary one when they conflict."""
        web_search = self._registry.web_search()
        queries = [natural for natural, _expanded in query_pairs][:WEB_MAX_QUERIES_PER_ROUND]

        added = 0
        for query in queries:
            try:
                results = web_search.search(query, max_results=WEB_MAX_RESULTS_PER_QUERY)
            except Exception as exc:
                logger.warning("Web search failed for query %r: %s", query, exc)
                continue

            room_left = max(EVIDENCE_POOL_CAP - len(memory.evidence), 0)
            for result in results:
                if room_left <= 0:
                    break
                if result.url in memory.seen_urls:
                    continue
                content = (result.content or result.snippet or "")[:WEB_CONTENT_EXCERPT_CHARS]
                if not content.strip():
                    continue
                memory.evidence.append(
                    EvidenceItem(
                        index=len(memory.evidence),
                        chunk_id=result.url,
                        document_id=result.url,
                        document_filename=result.title,
                        content=content,
                        heading=None,
                        page_number=None,
                        score=0.0,
                        matched_by=["web"],
                        source_authority=result.source_authority,
                        source_url=result.url,
                    )
                )
                memory.seen_urls.add(result.url)
                added += 1
                room_left -= 1

        return added

    def _step_synthesize(self, memory: _WorkingMemory) -> list[int]:
        if not memory.evidence:
            memory.findings = []
            return list(range(len(memory.plan.subquestions)))
        provider, model = self._registry.llm_for(TaskName.EVIDENCE_ANALYZER)
        response = provider.complete(
            messages=[Message(role="user", content=build_synthesis_prompt(memory.plan.subquestions, memory.evidence))],
            model=model,
            temperature=0.1,
            max_tokens=1500,
        )
        findings, uncovered = parse_synthesis_response(
            response.content,
            evidence_count=len(memory.evidence),
            subquestion_count=len(memory.plan.subquestions),
        )
        memory.findings = findings
        return uncovered

    def _step_challenge(self, memory: _WorkingMemory) -> None:
        if not memory.findings:
            memory.contradictions = []
            memory.gaps = ["No findings were established from the evidence retrieved so far."]
            memory.pending_additional_terms = []
            return
        provider, model = self._registry.llm_for(TaskName.EVIDENCE_ANALYZER)
        response = provider.complete(
            messages=[Message(role="user", content=build_challenge_prompt(memory.findings, memory.evidence))],
            model=model,
            temperature=0.2,
            max_tokens=1200,
        )
        contradictions, gaps, additional_terms = parse_challenge_response(
            response.content, evidence_count=len(memory.evidence), finding_count=len(memory.findings)
        )
        memory.contradictions = contradictions
        memory.gaps = gaps
        memory.pending_additional_terms = additional_terms

    def _step_verify(self, memory: _WorkingMemory, question: str) -> None:
        """Advisory only: the verifier may add caveats but can never rewrite or
        remove an already evidence-linked finding (anti-fabrication guarantee
        holds even for the dedicated verification pass)."""
        provider, model = self._registry.llm_for(TaskName.VERIFIER)
        findings_text = "\n".join(f"- {f.statement}" for f in memory.findings) or "(none)"
        prompt = (
            f'Question: "{question}"\n\nFindings to review:\n{findings_text}\n\n'
            "Note any concern about overstatement, weak support, or missing caveats. "
            "Respond with a short bullet list of concerns, or 'No concerns.' if none. "
            "Do not introduce new facts."
        )
        response = provider.complete(messages=[Message(role="user", content=prompt)], model=model, temperature=0.2, max_tokens=400)
        text = response.content.strip()
        if text and "no concerns" not in text.lower():
            memory.verification_notes = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]

    def _step_final(self, memory: _WorkingMemory, question: str) -> FinalAnswer:
        provider, model = self._registry.llm_for(TaskName.FINAL_SYNTHESIS)
        response = provider.complete(
            messages=[
                Message(
                    role="user",
                    content=build_final_summary_prompt(question, memory.findings, memory.contradictions, memory.gaps),
                )
            ],
            model=model,
            temperature=0.3,
            max_tokens=500,
        )
        evidence_by_index = {e.index: e for e in memory.evidence}
        confidence = assess_confidence(
            findings=memory.findings,
            evidence_by_index=evidence_by_index,
            contradictions=memory.contradictions,
            subquestion_count=len(memory.plan.subquestions),
        )
        coverage = [
            f"{len(memory.evidence)} evidence item(s) retrieved across {memory.round_number + 1} search round(s)",
            *memory.plan.search_strategies,
        ]
        gaps = list(memory.gaps) + [f"Verifier note: {n}" for n in memory.verification_notes]
        gaps += [
            f"UNCONFIRMED CITATION -- a case named '{name}' was suggested but could NOT be independently "
            "verified to exist. Do not cite or rely on it without confirming it yourself through a "
            "primary legal database."
            for name in memory.unconfirmed_citations
        ]
        return FinalAnswer(
            executive_answer=response.content.strip(),
            key_findings=memory.findings,
            evidence=memory.evidence,
            contradictions=memory.contradictions,
            gaps=gaps,
            research_coverage=coverage,
            confidence=confidence,
        )

    # -- helpers ----------------------------------------------------------

    def _document_filenames(self, document_ids: list[str]) -> dict[str, str]:
        if not document_ids:
            return {}
        rows = self._db.table("documents").select("id, filename").in_("id", document_ids).execute().data
        return {row["id"]: row["filename"] for row in rows}

    @staticmethod
    def _trace_event(step: str, detail: str) -> dict:
        from datetime import UTC, datetime

        return {"step": step, "detail": detail, "at": datetime.now(UTC).isoformat()}

    @staticmethod
    def _final_answer_to_dict(answer: FinalAnswer) -> dict:
        return {
            "executive_answer": answer.executive_answer,
            "key_findings": [asdict(f) for f in answer.key_findings],
            "evidence": [asdict(e) for e in answer.evidence],
            "contradictions": [asdict(c) for c in answer.contradictions],
            "gaps": answer.gaps,
            "research_coverage": answer.research_coverage,
            "confidence": asdict(answer.confidence),
        }

    def _checkpoint(
        self,
        session_id: str,
        state: AgentState,
        memory: _WorkingMemory,
        trace: list[dict],
        *,
        status: str | None = None,
        final_answer: dict | None = None,
    ) -> None:
        update = {"agent_state": state.value, "state_data": memory.to_dict(), "operational_trace": trace}
        if status is not None:
            update["status"] = status
        if final_answer is not None:
            update["final_answer"] = final_answer
        self._db.table("research_sessions").update(update).eq("id", session_id).execute()
