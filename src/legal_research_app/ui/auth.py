"""Real Supabase Auth for the Streamlit UI (Phase 7). Replaces the dev-only
org-picker stand-in from Phase 6: the app now uses a per-user, JWT-authenticated
client (RLS-enforced) for everything user-driven, not the service-role client.
"""

from __future__ import annotations

from dataclasses import dataclass

from supabase import Client
from supabase_auth.errors import AuthApiError

from legal_research_app.db.client import build_anon_client


class AuthError(RuntimeError):
    pass


@dataclass
class AuthedSession:
    client: Client
    user_id: str
    email: str
    access_token: str
    refresh_token: str


def sign_up(email: str, password: str) -> AuthedSession:
    client = build_anon_client()
    try:
        response = client.auth.sign_up({"email": email, "password": password})
    except AuthApiError as exc:
        raise AuthError(str(exc)) from exc
    if response.session is None:
        raise AuthError(
            "Account created, but email confirmation is required before signing in. "
            "Check your inbox, then sign in."
        )
    client.auth.set_session(response.session.access_token, response.session.refresh_token)
    return AuthedSession(
        client=client,
        user_id=response.user.id,
        email=email,
        access_token=response.session.access_token,
        refresh_token=response.session.refresh_token,
    )


def sign_in(email: str, password: str) -> AuthedSession:
    client = build_anon_client()
    try:
        response = client.auth.sign_in_with_password({"email": email, "password": password})
    except AuthApiError as exc:
        raise AuthError("Incorrect email or password.") from exc
    client.auth.set_session(response.session.access_token, response.session.refresh_token)
    return AuthedSession(
        client=client,
        user_id=response.user.id,
        email=email,
        access_token=response.session.access_token,
        refresh_token=response.session.refresh_token,
    )


def restore_session(access_token: str, refresh_token: str) -> Client:
    """Rebuilds a working, RLS-authenticated client from tokens already held
    in st.session_state -- used on every Streamlit rerun, since the client
    object itself isn't cheaply cacheable across reruns tied to a specific token."""
    client = build_anon_client()
    client.auth.set_session(access_token, refresh_token)
    return client
