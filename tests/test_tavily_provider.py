"""Unit test for the domain-allowlist enforcement in TavilyWebSearchProvider.

Real finding this guards against: Tavily's own `include_domains` parameter is
not a strict filter (confirmed empirically -- a 7-domain allowlist still
returned a result from an unlisted domain). This test mocks the HTTP response
to deterministically prove the provider's own post-filter actually drops
disallowed domains, without depending on Tavily's API behavior on any given day.
"""

from __future__ import annotations

import httpx
import pytest

from legal_research_app.providers.tavily_provider import TavilyWebSearchProvider


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_search_drops_results_outside_the_allowlist(monkeypatch):
    fake_payload = {
        "results": [
            {"url": "https://indiankanoon.org/doc/1", "title": "Allowed", "content": "c"},
            {"url": "https://supremetoday.ai/some-article", "title": "Not allowed", "content": "c"},
            {"url": "https://delhihighcourt.nic.in/judgment.pdf", "title": "gov.in suffix allowed", "content": "c"},
        ]
    }
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _FakeResponse(fake_payload))

    provider = TavilyWebSearchProvider(api_key="test-key", allowed_domains=("indiankanoon.org",))
    results = provider.search("query")

    urls = {r.url for r in results}
    assert "https://indiankanoon.org/doc/1" in urls
    assert "https://delhihighcourt.nic.in/judgment.pdf" in urls  # .nic.in implicitly allowed
    assert "https://supremetoday.ai/some-article" not in urls  # not on allowlist, dropped


def test_search_raises_clear_error_without_api_key():
    provider = TavilyWebSearchProvider(api_key=None)
    with pytest.raises(Exception, match="TAVILY_API_KEY is not set"):
        provider.search("query")
