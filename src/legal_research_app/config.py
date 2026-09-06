"""Centralized configuration: settings, provider profiles, privacy modes.

Nothing in this module ever prints or logs a secret value. See `Settings.safe_dict()`
for the masked view used anywhere configuration needs to be displayed or logged.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrivacyMode(StrEnum):
    """Per-organization privacy setting. See master prompt section 19.

    LOCAL_ONLY: no document content or question text may reach OpenAI/Gemini.
        LLM tasks must resolve to Ollama; embeddings must resolve to local.
    HYBRID: documents/embeddings stay local; small evidence excerpts may be
        sent to a configured cloud provider for planning/synthesis/verification.
    CLOUD_ASSISTED: org has explicitly opted into broader cloud model usage.
    """

    LOCAL_ONLY = "local_only"
    HYBRID = "hybrid"
    CLOUD_ASSISTED = "cloud_assisted"


class ProviderName(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    LOCAL = "local"  # in-process, e.g. sentence-transformers for embeddings


# Providers that count as "leaves this machine/org infra" for privacy enforcement.
CLOUD_PROVIDERS = frozenset({ProviderName.OPENAI, ProviderName.GEMINI})


class TaskName(StrEnum):
    """Task-level provider assignment, per master prompt section 16."""

    PLANNER = "planner"
    QUERY_EXPANSION = "query_expansion"  # Synonimise
    EMBEDDINGS = "embeddings"
    RERANKER = "reranker"
    EVIDENCE_ANALYZER = "evidence_analyzer"
    VERIFIER = "verifier"
    FINAL_SYNTHESIS = "final_synthesis"


class TaskConfig(BaseModel):
    provider: ProviderName
    model: str


class ProviderProfile(BaseModel):
    """A named, complete assignment of provider+model to every task.

    Switching providers means switching the active profile, never scattering
    provider/model literals through application code (master prompt section 53).
    """

    name: str
    privacy_mode: PrivacyMode
    tasks: dict[TaskName, TaskConfig]

    def task(self, task: TaskName) -> TaskConfig:
        try:
            return self.tasks[task]
        except KeyError as exc:
            raise ValueError(
                f"Provider profile '{self.name}' has no configuration for task '{task.value}'."
            ) from exc


# --- Built-in profiles -------------------------------------------------------
#
# Defaults reflect the user's actual keys/hardware at build time:
#   - OpenAI: paid key, reliable -> primary cloud provider for every reasoning task.
#   - Gemini: free tier, unreliable -> available but never a default/fallback.
#   - Ollama (gemma4:e2b-mlx) + local BGE-M3 embeddings -> local-only / dev profile.
# These are defaults, not hardcoded behavior: change the active profile, not the code.

_OPENAI_MODEL = "gpt-4o-mini"
_OLLAMA_LLM_MODEL = "gemma4:e2b-mlx"
_LOCAL_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_PROFILES: dict[str, ProviderProfile] = {
    "local_only": ProviderProfile(
        name="local_only",
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        tasks={
            TaskName.PLANNER: TaskConfig(provider=ProviderName.OLLAMA, model=_OLLAMA_LLM_MODEL),
            TaskName.QUERY_EXPANSION: TaskConfig(provider=ProviderName.OLLAMA, model=_OLLAMA_LLM_MODEL),
            TaskName.EMBEDDINGS: TaskConfig(provider=ProviderName.LOCAL, model=_LOCAL_EMBEDDING_MODEL),
            TaskName.RERANKER: TaskConfig(provider=ProviderName.LOCAL, model="none"),
            TaskName.EVIDENCE_ANALYZER: TaskConfig(provider=ProviderName.OLLAMA, model=_OLLAMA_LLM_MODEL),
            TaskName.VERIFIER: TaskConfig(provider=ProviderName.OLLAMA, model=_OLLAMA_LLM_MODEL),
            TaskName.FINAL_SYNTHESIS: TaskConfig(provider=ProviderName.OLLAMA, model=_OLLAMA_LLM_MODEL),
        },
    ),
    "hybrid_openai_default": ProviderProfile(
        name="hybrid_openai_default",
        privacy_mode=PrivacyMode.HYBRID,
        tasks={
            TaskName.PLANNER: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            TaskName.QUERY_EXPANSION: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            TaskName.EMBEDDINGS: TaskConfig(provider=ProviderName.LOCAL, model=_LOCAL_EMBEDDING_MODEL),
            TaskName.RERANKER: TaskConfig(provider=ProviderName.LOCAL, model="none"),
            TaskName.EVIDENCE_ANALYZER: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            TaskName.VERIFIER: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            TaskName.FINAL_SYNTHESIS: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
        },
    ),
    "hybrid_openai_gemini_verify": ProviderProfile(
        name="hybrid_openai_gemini_verify",
        privacy_mode=PrivacyMode.HYBRID,
        tasks={
            TaskName.PLANNER: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            TaskName.QUERY_EXPANSION: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            TaskName.EMBEDDINGS: TaskConfig(provider=ProviderName.LOCAL, model=_LOCAL_EMBEDDING_MODEL),
            TaskName.RERANKER: TaskConfig(provider=ProviderName.LOCAL, model="none"),
            TaskName.EVIDENCE_ANALYZER: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
            # Gemini free tier is unreliable -> used only where explicitly chosen,
            # and its failure must surface as an error, never a silent OpenAI fallback.
            TaskName.VERIFIER: TaskConfig(provider=ProviderName.GEMINI, model="gemini-2.0-flash"),
            TaskName.FINAL_SYNTHESIS: TaskConfig(provider=ProviderName.OPENAI, model=_OPENAI_MODEL),
        },
    ),
}

DEFAULT_ACTIVE_PROFILE = "hybrid_openai_default"


class Settings(BaseSettings):
    """Environment-backed settings. Populate via `.env` (see `.env.example`)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Credentials -- never logged, never printed, never persisted in session output.
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"

    supabase_url: str | None = None
    supabase_key: str | None = None
    supabase_anon_key: str | None = None

    tavily_api_key: str | None = None

    active_profile: str = DEFAULT_ACTIVE_PROFILE
    log_level: str = "INFO"

    def profile(self) -> ProviderProfile:
        try:
            return DEFAULT_PROFILES[self.active_profile]
        except KeyError as exc:
            available = ", ".join(sorted(DEFAULT_PROFILES))
            raise ValueError(
                f"Unknown provider profile '{self.active_profile}'. Available: {available}"
            ) from exc

    def safe_dict(self) -> dict[str, str]:
        """Masked view for logging/UI display -- secrets never appear in full."""

        def mask(value: str | None) -> str:
            if not value:
                return "(not set)"
            return f"{'*' * max(len(value) - 4, 0)}{value[-4:]}"

        return {
            "openai_api_key": mask(self.openai_api_key),
            "gemini_api_key": mask(self.gemini_api_key),
            "ollama_host": self.ollama_host,
            "supabase_url": self.supabase_url or "(not set)",
            "supabase_key": mask(self.supabase_key),
            "supabase_anon_key": mask(self.supabase_anon_key),
            "tavily_api_key": mask(self.tavily_api_key),
            "active_profile": self.active_profile,
            "log_level": self.log_level,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
