"""Classifies a URL as a primary or secondary legal source, by domain.

Deliberately deterministic and domain-based, not LLM-judged: the whole point
is that the research agent's synthesis/challenge prompts can trust this label
unconditionally when deciding which source wins a conflict, which only holds
if classification itself doesn't depend on the same model being asked to
reason carefully about the content (the exact thing found unreliable earlier
in this project). A wrong classification here is a config problem to fix by
editing this list, not a per-answer reasoning failure to hope the model avoids.

Primary: the bare text of a statute, an official Gazette notification, or a
court's own judgment. Secondary: everything else (law firm blogs, summaries,
commentary, news coverage of a case) -- useful context, but never authoritative
over a primary source it conflicts with.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Indian legal-research domains specifically, since that's this deployment's
# primary use so far -- extend this list as new jurisdictions are added rather
# than trying to guess a universal rule.
PRIMARY_DOMAINS = frozenset(
    {
        "indiankanoon.org",  # case law + bare acts
        "sci.gov.in",  # Supreme Court of India
        "indiacode.nic.in",  # official bare acts/statutes
        "egazette.gov.in",  # official Gazette of India notifications
        "egazette.nic.in",
        "doj.gov.in",  # Department of Justice
        "legislative.gov.in",
        "livelaw.in",  # court order/judgment full-text reporting (not commentary-only)
    }
)

# Any *.gov.in or *.nic.in domain not explicitly listed above is still treated
# as primary (government/judiciary sites), since High Courts each have their
# own such domain (e.g. delhihighcourt.nic.in, hpsja.nic.in) -- listing every
# one individually isn't maintainable.
PRIMARY_SUFFIXES = (".gov.in", ".nic.in")

# Domains explicitly allowed for Verified Research web search (the allowlist
# passed to the search provider itself). Includes reputable secondary sources
# too -- being "secondary" doesn't mean untrustworthy, just non-authoritative
# when it conflicts with a primary source.
ALLOWED_SEARCH_DOMAINS: tuple[str, ...] = (
    "indiankanoon.org",
    "sci.gov.in",
    "indiacode.nic.in",
    "egazette.gov.in",
    "livelaw.in",
    "scconline.com",
    "barandbench.com",
)


def classify_source_authority(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    domain = domain.removeprefix("www.")
    if domain in PRIMARY_DOMAINS or any(domain.endswith(suffix) for suffix in PRIMARY_SUFFIXES):
        return "primary"
    return "secondary"
