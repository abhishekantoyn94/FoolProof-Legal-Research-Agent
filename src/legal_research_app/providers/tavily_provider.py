"""Tavily web search: the evidence channel for ResearchMode.VERIFIED.

Chosen over a generic search API because it's built for exactly this use case
(LLM/agent research) -- native domain restriction and optional full-page
content extraction in one call, rather than a search-snippets-only API that
would need a separate fetch+parse step per result.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

from legal_research_app.providers.base import ProviderError, ProviderHealth, WebSearchProvider, WebSearchResult
from legal_research_app.retrieval.source_authority import (
    ALLOWED_SEARCH_DOMAINS,
    PRIMARY_SUFFIXES,
    classify_source_authority,
)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    """Tavily's `include_domains` is not a strict filter -- confirmed
    empirically (a query with a 7-domain allowlist still returned a result
    from a domain not on that list). Never trust an external API's own
    restriction parameter to be authoritative; re-check every result here."""
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    if any(domain == d or domain.endswith("." + d) for d in allowed_domains):
        return True
    # Any government/judiciary domain is implicitly allowed even if not
    # individually listed (see source_authority.py's PRIMARY_SUFFIXES) --
    # High Courts each have their own such domain.
    return any(domain.endswith(suffix) for suffix in PRIMARY_SUFFIXES)


class TavilyWebSearchProvider(WebSearchProvider):
    name = "tavily"

    def __init__(self, api_key: str | None, *, allowed_domains: tuple[str, ...] = ALLOWED_SEARCH_DOMAINS) -> None:
        self._api_key = api_key
        self._allowed_domains = list(allowed_domains)

    def health(self) -> ProviderHealth:
        if not self._api_key:
            return ProviderHealth(available=False, detail="TAVILY_API_KEY is not set.")
        return ProviderHealth(available=True, detail=f"API key configured; {len(self._allowed_domains)} domain(s) allowlisted.")

    def search(self, query: str, *, max_results: int = 8) -> list[WebSearchResult]:
        if not self._api_key:
            raise ProviderError(
                "Web search failed: TAVILY_API_KEY is not set. "
                "Verified Research mode requires it -- set it in .env or use a different research mode."
            )
        try:
            response = httpx.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_domains": self._allowed_domains,
                    "include_raw_content": True,
                    "max_results": max_results,
                },
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Web search failed: Tavily returned {exc.response.status_code}. "
                "Check TAVILY_API_KEY, or retry -- no local document data was affected."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"Web search failed: could not reach Tavily ({exc}). Retry, or use a different research mode."
            ) from exc

        data = response.json()
        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            if not url or not _domain_allowed(url, self._allowed_domains):
                continue
            results.append(
                WebSearchResult(
                    url=url,
                    title=item.get("title", "") or url,
                    snippet=item.get("content", "") or "",
                    content=item.get("raw_content") or item.get("content") or None,
                    source_authority=classify_source_authority(url),
                )
            )
        return results
