"""Provider-agnostic interfaces. See master prompt sections 14-15.

Every task (planner, query expansion, embeddings, reranker, evidence analysis,
verification, final synthesis) is served through one of these interfaces, so the
application never depends on a specific vendor SDK outside the `providers/` package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class ProviderHealth:
    available: bool
    detail: str = ""


class ProviderError(RuntimeError):
    """Raised on provider failure. Must always explain what happened and what
    the caller can do next (master prompt section 45) -- never a bare exception.
    """


class PrivacyViolationError(RuntimeError):
    """Raised when the active privacy mode forbids the provider a task resolved to.

    This must never be caught to silently pick a different provider (master
    prompt section 18) -- it is surfaced to the user as an explicit choice.
    """


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    def complete(
        self,
        *,
        messages: list[Message],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class EmbeddingProvider(ABC):
    name: str

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    def embed(self, texts: list[str], model: str) -> list[list[float]]: ...


@dataclass(frozen=True)
class RerankedResult:
    index: int
    score: float


class RerankerProvider(ABC):
    name: str

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    def rerank(self, query: str, documents: list[str], model: str) -> list[RerankedResult]: ...


@dataclass(frozen=True)
class ExpandedQuery:
    """One structured output element of query expansion (Synonimise)."""

    text: str
    kind: str = "term"  # "term" | "boolean_expression" | "phrase"


@dataclass(frozen=True)
class QueryExpansionResult:
    original_query: str
    expansions: list[ExpandedQuery] = field(default_factory=list)


class QueryExpansionProvider(ABC):
    name: str

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    def expand(self, query: str, model: str, *, jurisdiction: str | None = None) -> QueryExpansionResult: ...


@dataclass(frozen=True)
class WebSearchResult:
    url: str
    title: str
    snippet: str
    content: str | None  # full fetched page text, if available; else just the snippet
    source_authority: str  # "primary" | "secondary" -- see retrieval/source_authority.py


class WebSearchProvider(ABC):
    """Used only by ResearchMode.VERIFIED -- a second evidence channel alongside
    local document search, restricted to a domain allowlist (master prompt's
    general caution about unrestricted external research applies doubly to a
    legal tool: an unreliable source stated confidently is worse than no source).
    """

    name: str

    @abstractmethod
    def health(self) -> ProviderHealth: ...

    @abstractmethod
    def search(self, query: str, *, max_results: int = 8) -> list[WebSearchResult]: ...
