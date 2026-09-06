from legal_research_app.retrieval.source_authority import classify_source_authority


def test_indiankanoon_is_primary():
    assert classify_source_authority("https://indiankanoon.org/doc/128615827/") == "primary"


def test_supreme_court_site_is_primary():
    assert classify_source_authority("https://sci.gov.in/some-judgment") == "primary"


def test_any_gov_in_domain_is_primary():
    assert classify_source_authority("https://delhihighcourt.nic.in/app/showFileJudgment/x.pdf") == "primary"
    assert classify_source_authority("https://egazette.gov.in/notification/123") == "primary"


def test_law_firm_blog_is_secondary():
    assert classify_source_authority("https://bhattandjoshiassociates.com/some-article") == "secondary"


def test_www_prefix_does_not_affect_classification():
    assert classify_source_authority("https://www.indiankanoon.org/doc/1/") == "primary"


def test_unknown_domain_defaults_to_secondary():
    assert classify_source_authority("https://random-blog.example.com/post") == "secondary"
