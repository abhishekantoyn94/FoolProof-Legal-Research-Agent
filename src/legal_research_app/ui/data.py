"""Thin query helpers for the Streamlit UI. Deliberately separate from the
section-57 services (DocumentService/SearchService/ResearchService/etc.) --
these are read-only list/lookup queries for populating dropdowns and tables,
not business logic, so they don't belong in those services.
"""

from __future__ import annotations

from supabase import Client


def list_organizations(db: Client) -> list[dict]:
    return db.table("organizations").select("*").order("name").execute().data


def create_organization(db: Client, name: str, privacy_mode: str, active_provider_profile: str) -> dict:
    """Must be called with a per-user (JWT-authenticated) client -- the RPC
    assigns ownership via auth.uid(), and organizations has no direct INSERT
    policy (creation is only ever allowed through this security-definer RPC)."""
    return db.rpc(
        "create_organization_with_owner",
        {"p_name": name, "p_privacy_mode": privacy_mode, "p_provider_profile": active_provider_profile},
    ).execute().data


def invite_member(db: Client, org_id: str, email: str, role: str = "member") -> dict:
    return db.rpc("invite_member_to_org", {"p_org_id": org_id, "p_email": email, "p_role": role}).execute().data


def list_org_members(db: Client, org_id: str) -> list[dict]:
    return db.rpc("list_org_members", {"p_org_id": org_id}).execute().data


def list_projects(db: Client, org_id: str) -> list[dict]:
    return db.table("projects").select("*").eq("org_id", org_id).order("created_at", desc=True).execute().data


def create_project(db: Client, org_id: str, name: str, jurisdiction: str | None, description: str | None) -> dict:
    return (
        db.table("projects")
        .insert({"org_id": org_id, "name": name, "jurisdiction": jurisdiction or None, "description": description or None})
        .execute()
        .data[0]
    )


def list_kb_categories(db: Client, org_id: str) -> list[dict]:
    return db.table("kb_categories").select("*").eq("org_id", org_id).order("created_at", desc=True).execute().data


def create_kb_category(db: Client, org_id: str, name: str, jurisdiction: str | None, description: str | None) -> dict:
    return (
        db.table("kb_categories")
        .insert({"org_id": org_id, "name": name, "jurisdiction": jurisdiction or None, "description": description or None})
        .execute()
        .data[0]
    )


def list_documents(db: Client, org_id: str, project_id: str | None = None) -> list[dict]:
    query = db.table("documents").select("*, projects(name)").eq("org_id", org_id)
    if project_id is not None:
        query = query.eq("project_id", project_id)
    rows = query.order("created_at", desc=True).execute().data
    for row in rows:
        project = row.pop("projects", None)
        row["collection_name"] = (project or {}).get("name") or "(unknown)"
    return rows


def list_research_sessions(db: Client, org_id: str, project_id: str | None = None) -> list[dict]:
    query = (
        db.table("research_sessions")
        .select("id, question, research_mode, status, created_at, provider_profile")
        .eq("org_id", org_id)
    )
    if project_id is not None:
        query = query.eq("project_id", project_id)
    return query.order("created_at", desc=True).execute().data


def get_research_session(db: Client, session_id: str) -> dict:
    return db.table("research_sessions").select("*").eq("id", session_id).execute().data[0]
