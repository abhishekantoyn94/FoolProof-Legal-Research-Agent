from legal_research_app.config import (
    DEFAULT_PROFILES,
    PrivacyMode,
    ProviderName,
    Settings,
    TaskName,
)


def test_default_profiles_cover_every_task():
    for profile in DEFAULT_PROFILES.values():
        for task in TaskName:
            assert task in profile.tasks, f"{profile.name} missing task {task.value}"


def test_local_only_profile_never_uses_cloud_providers():
    profile = DEFAULT_PROFILES["local_only"]
    assert profile.privacy_mode == PrivacyMode.LOCAL_ONLY
    for task_cfg in profile.tasks.values():
        assert task_cfg.provider in (ProviderName.OLLAMA, ProviderName.LOCAL)


def test_settings_safe_dict_masks_secrets():
    settings = Settings(
        openai_api_key="sk-supersecretvalue1234",
        gemini_api_key=None,
        supabase_key="anon-key-abcdefgh",
        _env_file=None,
    )
    safe = settings.safe_dict()
    assert "supersecretvalue1234" not in str(safe)
    assert safe["openai_api_key"].endswith("1234")
    assert safe["gemini_api_key"] == "(not set)"


def test_unknown_profile_raises_clear_error():
    settings = Settings(active_profile="does_not_exist", _env_file=None)
    try:
        settings.profile()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
        assert "hybrid_openai_default" in str(exc)
