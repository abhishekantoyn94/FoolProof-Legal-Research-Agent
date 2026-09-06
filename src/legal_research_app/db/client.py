"""Supabase client wrapper. The service-role client is for backend-only use
(background ingestion jobs, admin operations) and bypasses Row Level Security --
it must never be exposed to a browser/frontend. User-facing requests should use
a client authenticated with the requesting user's own JWT so RLS applies.
"""

from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from legal_research_app.config import Settings, get_settings


class SupabaseNotConfiguredError(RuntimeError):
    pass


def build_service_client(settings: Settings | None = None) -> Client:
    settings = settings or get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        raise SupabaseNotConfiguredError(
            "SUPABASE_URL / SUPABASE_KEY are not set. Run `supabase start` for local "
            "dev and copy the printed values into .env, or point at a hosted project."
        )
    return create_client(settings.supabase_url, settings.supabase_key)


@lru_cache
def get_service_client() -> Client:
    return build_service_client()


def build_anon_client(settings: Settings | None = None) -> Client:
    """Client for auth flows (sign up/in) and as the base for a per-user,
    JWT-authenticated client once a session exists (see ui/auth.py). Never
    used for data access before a user session is attached -- RLS with the
    anon key and no session denies everything except what policies explicitly
    grant to the `anon` role (nothing, in this schema)."""
    settings = settings or get_settings()
    if not settings.supabase_url or not settings.supabase_anon_key:
        raise SupabaseNotConfiguredError(
            "SUPABASE_URL / SUPABASE_ANON_KEY are not set. Run `supabase start` for local "
            "dev and copy the printed values into .env, or point at a hosted project."
        )
    return create_client(settings.supabase_url, settings.supabase_anon_key)
