"""Pure stopping-criteria decision (master prompt section 12 step 13, section
33). Kept separate from ResearchAgent so the loop-control logic is testable
without mocking any LLM/DB/search machinery -- this is the one piece of the
agent that must behave deterministically regardless of model quality.
"""

from __future__ import annotations

from legal_research_app.agent.types import MODE_MAX_ROUNDS, MODE_MIN_ROUNDS, ResearchMode


def should_continue_research(
    *,
    mode: ResearchMode,
    completed_rounds: int,
    has_gaps_or_new_terms: bool,
    last_round_novelty: float,
    novelty_threshold: float,
) -> bool:
    """completed_rounds counts the round that just finished (1 after the first).

    Below the mode's minimum, always continue -- research depth is a product
    decision (QUICK/STANDARD/DEEP), not something the LLM's gap analysis gets
    to shortcut. Above the minimum, continue only while there's something left
    to chase (a reported gap or a requested follow-up term) AND the last round
    still surfaced meaningfully novel evidence -- once novelty drops below
    threshold, more rounds would just re-fetch what's already been seen.
    """
    min_rounds = MODE_MIN_ROUNDS[mode]
    max_rounds = MODE_MAX_ROUNDS[mode]

    if completed_rounds < min_rounds:
        return True
    if completed_rounds >= max_rounds:
        return False
    return has_gaps_or_new_terms and last_round_novelty >= novelty_threshold
