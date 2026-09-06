"""Resolves a task (planner, verifier, etc.) to a concrete provider instance,
enforcing the active privacy mode. This is the ONLY place task -> provider
resolution happens (master prompt section 53: no scattered provider logic).

Hard rule (master prompt section 18): if privacy mode forbids the provider a
task's active profile names, this raises PrivacyViolationError. It never picks
a different provider on your behalf.
"""

from __future__ import annotations

from legal_research_app.config import (
    CLOUD_PROVIDERS,
    PrivacyMode,
    ProviderName,
    Settings,
    TaskConfig,
    TaskName,
)
from legal_research_app.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    PrivacyViolationError,
    ProviderHealth,
    QueryExpansionProvider,
    WebSearchProvider,
)
from legal_research_app.providers.gemini_provider import GeminiProvider
from legal_research_app.providers.local_embedding_provider import LocalEmbeddingProvider
from legal_research_app.providers.ollama_provider import OllamaProvider
from legal_research_app.providers.openai_provider import OpenAIProvider
from legal_research_app.providers.synonimise_provider import SynonimiseProvider
from legal_research_app.providers.tavily_provider import TavilyWebSearchProvider


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._profile = settings.profile()
        self._llm_providers: dict[ProviderName, LLMProvider] = {}
        self._embedding_providers: dict[ProviderName, EmbeddingProvider] = {}
        self._web_search_provider: WebSearchProvider | None = None

    def _llm_provider(self, provider_name: ProviderName) -> LLMProvider:
        if provider_name not in self._llm_providers:
            if provider_name == ProviderName.OPENAI:
                self._llm_providers[provider_name] = OpenAIProvider(self._settings.openai_api_key)
            elif provider_name == ProviderName.GEMINI:
                self._llm_providers[provider_name] = GeminiProvider(self._settings.gemini_api_key)
            elif provider_name == ProviderName.OLLAMA:
                self._llm_providers[provider_name] = OllamaProvider(self._settings.ollama_host)
            else:
                raise ValueError(f"No LLM provider implementation for '{provider_name.value}'.")
        return self._llm_providers[provider_name]

    def _enforce_privacy(self, task: TaskName, provider_name: ProviderName) -> None:
        if self._profile.privacy_mode == PrivacyMode.LOCAL_ONLY and provider_name in CLOUD_PROVIDERS:
            raise PrivacyViolationError(
                f"Task '{task.value}' resolved to cloud provider '{provider_name.value}', "
                f"but the active profile '{self._profile.name}' is LOCAL_ONLY. "
                "No document content or question text may leave this machine/server in this mode. "
                "Switch to a HYBRID or CLOUD_ASSISTED profile if cloud use is intended, "
                "or pick a profile whose task assignments stay local."
            )

    def _embedding_provider(self, provider_name: ProviderName) -> EmbeddingProvider:
        if provider_name != ProviderName.LOCAL:
            raise ValueError(
                f"No embedding provider implementation for '{provider_name.value}' yet -- "
                "only local (sentence-transformers) embeddings are currently supported."
            )
        if provider_name not in self._embedding_providers:
            self._embedding_providers[provider_name] = LocalEmbeddingProvider()
        return self._embedding_providers[provider_name]

    def task_config(self, task: TaskName) -> TaskConfig:
        return self._profile.task(task)

    def llm_for(self, task: TaskName) -> tuple[LLMProvider, str]:
        """Returns (provider, model) for a task, after privacy enforcement."""
        cfg = self.task_config(task)
        self._enforce_privacy(task, cfg.provider)
        return self._llm_provider(cfg.provider), cfg.model

    def embedding_for(self, task: TaskName = TaskName.EMBEDDINGS) -> tuple[EmbeddingProvider, str]:
        cfg = self.task_config(task)
        self._enforce_privacy(task, cfg.provider)
        return self._embedding_provider(cfg.provider), cfg.model

    def query_expansion_for(
        self, task: TaskName = TaskName.QUERY_EXPANSION
    ) -> tuple[QueryExpansionProvider, str]:
        """Synonimise wraps whichever LLM the task resolves to -- there is no
        separate per-vendor implementation to maintain (section 52-53)."""
        llm_provider, model = self.llm_for(task)
        return SynonimiseProvider(llm_provider), model

    def web_search(self) -> WebSearchProvider:
        """Only used by ResearchMode.VERIFIED. Web search inherently sends the
        query (and, for fetched pages, receives external content) outside this
        machine/server, so it is incompatible with LOCAL_ONLY -- enforced the
        same way as every other cloud-bound task (section 18: no silent fallback,
        this raises rather than skipping web search quietly)."""
        if self._profile.privacy_mode == PrivacyMode.LOCAL_ONLY:
            raise PrivacyViolationError(
                f"Verified Research mode requires web search, but the active profile "
                f"'{self._profile.name}' is LOCAL_ONLY. Web search sends your question "
                "to an external search API and fetches external pages -- incompatible "
                "with LOCAL_ONLY by definition. Use a HYBRID/CLOUD_ASSISTED profile, "
                "or use a different research mode."
            )
        if self._web_search_provider is None:
            self._web_search_provider = TavilyWebSearchProvider(self._settings.tavily_api_key)
        return self._web_search_provider

    def health_report(self) -> dict[str, ProviderHealth]:
        """Health of every distinct LLM provider referenced by the active profile."""
        report: dict[str, ProviderHealth] = {}
        seen: set[ProviderName] = set()
        for cfg in self._profile.tasks.values():
            if cfg.provider in (ProviderName.LOCAL,) or cfg.provider in seen:
                continue
            seen.add(cfg.provider)
            report[cfg.provider.value] = self._llm_provider(cfg.provider).health()
        return report

    @property
    def privacy_mode(self) -> PrivacyMode:
        return self._profile.privacy_mode

    @property
    def profile_name(self) -> str:
        return self._profile.name
