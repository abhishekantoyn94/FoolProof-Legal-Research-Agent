from legal_research_app.retrieval.citation_matching import titles_match


def test_matches_real_case_despite_formatting_differences():
    """The exact real-world case this exists for: 'v.' vs 'vs', an appended date."""
    assert titles_match(
        "Anil Kumar Dash v. State of Orissa",
        "Anil Kumar Dash vs State Of Orissa on 22 September, 2015",
    )


def test_does_not_match_an_unrelated_case():
    assert not titles_match(
        "Raju Boruah v. State of Assam",
        "Manjeet Kumar vs The State Of Assam on 6 January, 2020",
    )


def test_empty_candidate_name_never_matches():
    assert not titles_match("", "Anything vs Something on 1 January, 2020")


def test_partial_party_name_overlap_below_threshold_does_not_match():
    # Shares only "State" -- nowhere near enough to confirm it's the same case.
    assert not titles_match("Anil Kumar Dash v. State of Orissa", "Totally Different Party vs State Of Kerala")
