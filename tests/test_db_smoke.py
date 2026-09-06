"""Integration smoke test against a REAL local Supabase instance.

Requires `supabase start` to be running. Skips itself cleanly if not (so the
rest of the suite stays runnable without Docker/Supabase up).
"""

from __future__ import annotations

import pytest

from legal_research_app.config import get_settings
from legal_research_app.db.client import SupabaseNotConfiguredError, build_service_client


def _client_or_skip():
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        pytest.skip("SUPABASE_URL/SUPABASE_KEY not configured; run `supabase start`.")
    try:
        client = build_service_client(settings)
        client.table("organizations").select("id").limit(1).execute()
    except SupabaseNotConfiguredError:
        pytest.skip("Supabase not configured.")
    except Exception as exc:  # local stack not running
        pytest.skip(f"Local Supabase not reachable: {exc}")
    return client


def test_full_document_hierarchy_round_trip():
    client = _client_or_skip()

    org = client.table("organizations").insert(
        {"name": "Test Org (pytest)", "privacy_mode": "hybrid"}
    ).execute().data[0]
    org_id = org["id"]

    try:
        project = client.table("projects").insert(
            {"org_id": org_id, "name": "Smith v. Jones", "jurisdiction": "US-NY"}
        ).execute().data[0]

        kb = client.table("kb_categories").insert(
            {"org_id": org_id, "name": "Contract Templates"}
        ).execute().data[0]

        # A document must belong to exactly one of {project, kb_category} -- the
        # DB constraint, not just application code, enforces this.
        doc = client.table("documents").insert(
            {
                "org_id": org_id,
                "project_id": project["id"],
                "filename": "complaint.pdf",
                "storage_path": f"{org_id}/complaint.pdf",
                "status": "pending",
            }
        ).execute().data[0]

        chunk = client.table("document_chunks").insert(
            {
                "document_id": doc["id"],
                "org_id": org_id,
                "chunk_index": 0,
                "content": "This is a test chunk about patent infringement.",
                "page_number": 1,
            }
        ).execute().data[0]

        # Lexical (BM25-equivalent) search via the generated tsvector column works.
        found = (
            client.table("document_chunks")
            .select("id, content")
            .text_search("content_tsv", "patent")
            .execute()
            .data
        )
        assert any(c["id"] == chunk["id"] for c in found)

        # KB category is independent of the project and still queryable.
        kb_read = client.table("kb_categories").select("id, name").eq("id", kb["id"]).execute().data
        assert kb_read[0]["name"] == "Contract Templates"

    finally:
        # Cleanup: cascade deletes handle projects/documents/chunks/kb_categories.
        client.table("organizations").delete().eq("id", org_id).execute()


def test_document_cannot_belong_to_both_project_and_kb_category():
    client = _client_or_skip()

    org = client.table("organizations").insert({"name": "Constraint Test Org"}).execute().data[0]
    org_id = org["id"]
    try:
        project = client.table("projects").insert(
            {"org_id": org_id, "name": "P1"}
        ).execute().data[0]
        kb = client.table("kb_categories").insert(
            {"org_id": org_id, "name": "KB1"}
        ).execute().data[0]

        with pytest.raises(Exception):
            client.table("documents").insert(
                {
                    "org_id": org_id,
                    "project_id": project["id"],
                    "kb_category_id": kb["id"],  # both set -- must violate the CHECK constraint
                    "filename": "bad.pdf",
                    "storage_path": "bad.pdf",
                }
            ).execute()
    finally:
        client.table("organizations").delete().eq("id", org_id).execute()
