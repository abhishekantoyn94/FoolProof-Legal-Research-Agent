"""Integration smoke test against a REAL local Ollama daemon. Skips cleanly if
Ollama isn't running or the configured model isn't pulled.
"""

from __future__ import annotations

import pytest

from legal_research_app.config import Settings, TaskName
from legal_research_app.providers.base import Message
from legal_research_app.providers.registry import ProviderRegistry


def test_local_only_profile_completes_a_real_chat_call():
    settings = Settings(active_profile="local_only", _env_file=None)
    registry = ProviderRegistry(settings)
    provider, model = registry.llm_for(TaskName.PLANNER)

    health = provider.health()
    if not health.available:
        pytest.skip(f"Ollama not reachable: {health.detail}")

    response = provider.complete(
        messages=[Message(role="user", content="Reply with exactly the word: pong")],
        model=model,
        max_tokens=10,
    )
    assert response.provider == "ollama"
    assert response.content.strip() != ""
