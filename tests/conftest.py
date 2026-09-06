from __future__ import annotations

import uuid

import pytest

from legal_research_app.config import get_settings
from legal_research_app.db.client import SupabaseNotConfiguredError, build_anon_client, build_service_client


@pytest.fixture
def supabase_client():
    """Real local Supabase client, or a clean skip if it isn't running."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        pytest.skip("SUPABASE_URL/SUPABASE_KEY not configured; run `supabase start`.")
    try:
        client = build_service_client(settings)
        client.table("organizations").select("id").limit(1).execute()
    except SupabaseNotConfiguredError:
        pytest.skip("Supabase not configured.")
    except Exception as exc:
        pytest.skip(f"Local Supabase not reachable: {exc}")
    return client


@pytest.fixture
def test_org(supabase_client):
    """Creates a throwaway organization, yields its id, always cleans up."""
    org = supabase_client.table("organizations").insert({"name": "pytest-org"}).execute().data[0]
    try:
        yield org["id"]
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()


@pytest.fixture
def signed_up_user(supabase_client):
    """Factory for real, JWT-authenticated per-user Supabase clients (via the
    anon key + a real sign-up), for testing auth/RLS/RPC behavior as an actual
    end user would experience it -- not the service-role client. Cleans up
    every created user (and cascades their org_members rows) at teardown."""
    created_user_ids: list[str] = []

    def _create(email: str | None = None, password: str = "password123"):
        email = email or f"pytest-{uuid.uuid4().hex[:12]}@example.com"
        client = build_anon_client()
        response = client.auth.sign_up({"email": email, "password": password})
        client.auth.set_session(response.session.access_token, response.session.refresh_token)
        created_user_ids.append(response.user.id)
        return client, response.user.id, email

    yield _create

    for user_id in created_user_ids:
        try:
            supabase_client.auth.admin.delete_user(user_id)
        except Exception:
            pass
