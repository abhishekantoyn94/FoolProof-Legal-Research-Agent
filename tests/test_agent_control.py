from legal_research_app.agent.control import should_continue_research
from legal_research_app.agent.types import ResearchMode

THRESHOLD = 0.2


def test_quick_mode_never_continues_past_its_single_baseline_round():
    assert should_continue_research(
        mode=ResearchMode.QUICK,
        completed_rounds=1,
        has_gaps_or_new_terms=True,
        last_round_novelty=1.0,
        novelty_threshold=THRESHOLD,
    ) is False


def test_standard_mode_forces_second_round_even_with_no_gaps():
    """Below the mode's minimum, the LLM's gap analysis cannot shortcut research depth."""
    assert should_continue_research(
        mode=ResearchMode.STANDARD,
        completed_rounds=1,
        has_gaps_or_new_terms=False,
        last_round_novelty=0.0,
        novelty_threshold=THRESHOLD,
    ) is True


def test_deep_mode_stops_at_max_rounds_regardless_of_remaining_gaps():
    from legal_research_app.agent.types import MODE_MAX_ROUNDS

    assert should_continue_research(
        mode=ResearchMode.DEEP,
        completed_rounds=MODE_MAX_ROUNDS[ResearchMode.DEEP],
        has_gaps_or_new_terms=True,
        last_round_novelty=1.0,
        novelty_threshold=THRESHOLD,
    ) is False


def test_beyond_minimum_stops_when_novelty_drops_below_threshold():
    from legal_research_app.agent.types import MODE_MIN_ROUNDS

    assert should_continue_research(
        mode=ResearchMode.DEEP,
        completed_rounds=MODE_MIN_ROUNDS[ResearchMode.DEEP],
        has_gaps_or_new_terms=True,
        last_round_novelty=0.05,  # below threshold: diminishing returns
        novelty_threshold=THRESHOLD,
    ) is False


def test_beyond_minimum_continues_when_gaps_remain_and_novelty_is_healthy():
    from legal_research_app.agent.types import MODE_MIN_ROUNDS

    assert should_continue_research(
        mode=ResearchMode.DEEP,
        completed_rounds=MODE_MIN_ROUNDS[ResearchMode.DEEP],
        has_gaps_or_new_terms=True,
        last_round_novelty=0.5,
        novelty_threshold=THRESHOLD,
    ) is True


def test_beyond_minimum_stops_when_no_gaps_remain_even_with_high_novelty():
    from legal_research_app.agent.types import MODE_MIN_ROUNDS

    assert should_continue_research(
        mode=ResearchMode.DEEP,
        completed_rounds=MODE_MIN_ROUNDS[ResearchMode.DEEP],
        has_gaps_or_new_terms=False,
        last_round_novelty=0.9,
        novelty_threshold=THRESHOLD,
    ) is False
