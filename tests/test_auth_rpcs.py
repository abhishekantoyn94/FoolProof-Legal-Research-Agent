"""Real end-to-end auth tests: actual sign-up against local GoTrue, actual
per-user JWT clients, actual RLS enforcement -- not mocked. This is the one
place correctness genuinely depends on Postgres RLS policies and the
security-definer RPCs behaving exactly as intended, so it's tested against
the real thing rather than assumed from reading the SQL.
"""

from __future__ import annotations


def test_new_user_sees_no_organizations(signed_up_user):
    client, _user_id, _email = signed_up_user()
    assert client.table("organizations").select("*").execute().data == []


def test_create_organization_with_owner_makes_org_and_membership_visible(signed_up_user, supabase_client):
    client, user_id, _email = signed_up_user()
    org = client.rpc(
        "create_organization_with_owner",
        {"p_name": "Test Org", "p_privacy_mode": "hybrid", "p_provider_profile": "hybrid_openai_default"},
    ).execute().data
    try:
        assert org["name"] == "Test Org"

        visible_orgs = client.table("organizations").select("*").execute().data
        assert [o["id"] for o in visible_orgs] == [org["id"]]

        membership = client.table("org_members").select("*").execute().data
        assert membership == [{"org_id": org["id"], "user_id": user_id, "role": "owner", "created_at": membership[0]["created_at"]}]
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()


def test_a_users_org_is_invisible_to_an_unrelated_user(signed_up_user, supabase_client):
    owner_client, _owner_id, _owner_email = signed_up_user()
    other_client, _other_id, _other_email = signed_up_user()

    org = owner_client.rpc(
        "create_organization_with_owner", {"p_name": "Private Org"}
    ).execute().data
    try:
        assert other_client.table("organizations").select("*").execute().data == []
        assert org["id"] not in {
            o["id"] for o in other_client.table("organizations").select("*").execute().data
        }
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()


def test_invite_makes_org_visible_to_invited_user(signed_up_user, supabase_client):
    owner_client, _owner_id, _owner_email = signed_up_user()
    member_client, member_id, member_email = signed_up_user()

    org = owner_client.rpc("create_organization_with_owner", {"p_name": "Shared Org"}).execute().data
    try:
        result = owner_client.rpc(
            "invite_member_to_org", {"p_org_id": org["id"], "p_email": member_email, "p_role": "member"}
        ).execute().data
        assert result["user_id"] == member_id
        assert result["role"] == "member"

        member_orgs = member_client.table("organizations").select("id").execute().data
        assert [o["id"] for o in member_orgs] == [org["id"]]
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()


def test_plain_member_cannot_invite_others(signed_up_user, supabase_client):
    owner_client, _owner_id, _owner_email = signed_up_user()
    member_client, _member_id, member_email = signed_up_user()
    outsider_client, _outsider_id, outsider_email = signed_up_user()

    org = owner_client.rpc("create_organization_with_owner", {"p_name": "Locked Down Org"}).execute().data
    try:
        owner_client.rpc(
            "invite_member_to_org", {"p_org_id": org["id"], "p_email": member_email, "p_role": "member"}
        ).execute()

        try:
            member_client.rpc(
                "invite_member_to_org", {"p_org_id": org["id"], "p_email": outsider_email, "p_role": "member"}
            ).execute()
            assert False, "a plain member must not be able to invite other users"
        except Exception as exc:
            assert "owner or admin" in str(exc).lower()
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()


def test_inviting_an_unregistered_email_fails_clearly(signed_up_user, supabase_client):
    owner_client, _owner_id, _owner_email = signed_up_user()
    org = owner_client.rpc("create_organization_with_owner", {"p_name": "Solo Org"}).execute().data
    try:
        try:
            owner_client.rpc(
                "invite_member_to_org",
                {"p_org_id": org["id"], "p_email": "no-such-user@example.com", "p_role": "member"},
            ).execute()
            assert False, "inviting a non-existent user must fail"
        except Exception as exc:
            assert "no registered user found" in str(exc).lower()
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()


def test_authenticated_user_can_write_document_chunks_for_their_own_org(signed_up_user, supabase_client):
    """Regression test for the document_chunks RLS fix: a real per-user JWT
    client (not service-role) must be able to insert chunks for a document in
    an org it belongs to -- this was broken (SELECT-only policy) before this phase."""
    client, _user_id, _email = signed_up_user()
    org = client.rpc("create_organization_with_owner", {"p_name": "Ingestion Org"}).execute().data
    try:
        project = client.table("projects").insert({"org_id": org["id"], "name": "P1"}).execute().data[0]
        doc = client.table("documents").insert(
            {"org_id": org["id"], "project_id": project["id"], "filename": "f.pdf", "storage_path": "x"}
        ).execute().data[0]

        chunk = client.table("document_chunks").insert(
            {"document_id": doc["id"], "org_id": org["id"], "chunk_index": 0, "content": "hello"}
        ).execute().data[0]
        assert chunk["content"] == "hello"
    finally:
        supabase_client.table("organizations").delete().eq("id", org["id"]).execute()
