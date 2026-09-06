import pytest

from legal_research_app.config import (
    PrivacyMode,
    ProviderName,
    ProviderProfile,
    Settings,
    TaskConfig,
    TaskName,
)
from legal_research_app.providers.base import PrivacyViolationError, ProviderError
from legal_research_app.providers.registry import ProviderRegistry


def test_hybrid_profile_resolves_openai_for_planner():
    settings = Settings(
        openai_api_key="sk-test",
        active_profile="hybrid_openai_default",
        _env_file=None,
    )
    registry = ProviderRegistry(settings)
    provider, model = registry.llm_for(TaskName.PLANNER)
    assert provider.name == "openai"
    assert model == "gpt-4o-mini"


def test_local_only_profile_resolves_to_ollama_not_cloud():
    settings = Settings(
        openai_api_key="sk-test",  # configured, but must NOT be used
        active_profile="local_only",
        _env_file=None,
    )
    registry = ProviderRegistry(settings)
    provider, model = registry.llm_for(TaskName.PLANNER)
    assert provider.name == "ollama"


def test_privacy_enforcement_raises_on_misconfigured_local_only_profile():
    """The core section-18 requirement: if a LOCAL_ONLY profile were ever
    misconfigured to point a task at a cloud provider, resolution must raise
    PrivacyViolationError -- it must NEVER silently substitute a different
    (e.g. local) provider on the caller's behalf."""
    settings = Settings(openai_api_key="sk-test", active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)
    # Simulate a misconfigured profile bypassing the (currently-safe) defaults.
    registry._profile = ProviderProfile(
        name="broken_local_only",
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        tasks={
            **registry._profile.tasks,
            TaskName.PLANNER: TaskConfig(provider=ProviderName.OPENAI, model="gpt-4o-mini"),
        },
    )
    with pytest.raises(PrivacyViolationError, match="LOCAL_ONLY"):
        registry.llm_for(TaskName.PLANNER)


def test_switching_active_profile_to_hybrid_with_no_openai_key_is_explicit_failure():
    """No OPENAI_API_KEY + hybrid profile -> calling the provider fails with a
    clear, actionable error, not a silent switch to another provider."""
    settings = Settings(openai_api_key=None, active_profile="hybrid_openai_default", _env_file=None)
    registry = ProviderRegistry(settings)
    provider, model = registry.llm_for(TaskName.PLANNER)
    assert provider.health().available is False
    with pytest.raises(ProviderError, match="OPENAI_API_KEY is not set"):
        provider.complete(messages=[], model=model)


def test_gemini_verify_profile_uses_gemini_only_for_verifier_task():
    settings = Settings(
        gemini_api_key="test-key",
        active_profile="hybrid_openai_gemini_verify",
        _env_file=None,
    )
    registry = ProviderRegistry(settings)
    provider, _ = registry.llm_for(TaskName.VERIFIER)
    assert provider.name == "gemini"
    provider, _ = registry.llm_for(TaskName.FINAL_SYNTHESIS)
    assert provider.name == "openai"


def test_health_report_does_not_include_local_provider():
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)
    report = registry.health_report()
    assert "local" not in report
    assert "ollama" in report


def test_web_search_forbidden_under_local_only_profile():
    settings = Settings(active_profile="local_only", tavily_api_key="test-key", _env_file=None)
    registry = ProviderRegistry(settings)
    with pytest.raises(PrivacyViolationError, match="LOCAL_ONLY"):
        registry.web_search()


def test_web_search_allowed_under_hybrid_profile():
    settings = Settings(active_profile="hybrid_openai_default", tavily_api_key="test-key", _env_file=None)
    registry = ProviderRegistry(settings)
    provider = registry.web_search()
    assert provider.name == "tavily"
    assert registry.web_search() is provider  # cached, not rebuilt each call
