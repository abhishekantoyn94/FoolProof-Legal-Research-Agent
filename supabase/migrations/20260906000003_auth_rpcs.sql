-- Phase 7: real multi-user auth. Organization creation and member invites are
-- funneled through security-definer RPCs (same pattern as is_org_member from
-- Phase 1) rather than direct table INSERT policies -- this avoids needing to
-- reason about a general "who can insert an org_members row for an arbitrary
-- org_id" policy, which is easy to get wrong. The functions internally
-- validate via auth.uid() and the caller's own membership/role.

create or replace function create_organization_with_owner(
    p_name text,
    p_privacy_mode text default 'hybrid',
    p_provider_profile text default 'hybrid_openai_default'
)
returns organizations
language plpgsql
security definer
set search_path = public
as $$
declare
    new_org organizations;
begin
    if auth.uid() is null then
        raise exception 'Must be authenticated to create an organization.';
    end if;

    insert into organizations (name, privacy_mode, active_provider_profile)
    values (p_name, p_privacy_mode, p_provider_profile)
    returning * into new_org;

    insert into org_members (org_id, user_id, role)
    values (new_org.id, auth.uid(), 'owner');

    return new_org;
end;
$$;

grant execute on function create_organization_with_owner(text, text, text) to authenticated;

create or replace function invite_member_to_org(
    p_org_id uuid,
    p_email text,
    p_role text default 'member'
)
returns org_members
language plpgsql
security definer
set search_path = public
as $$
declare
    caller_role text;
    target_user_id uuid;
    new_member org_members;
begin
    if auth.uid() is null then
        raise exception 'Must be authenticated to invite a member.';
    end if;

    if p_role not in ('owner', 'admin', 'member') then
        raise exception 'Invalid role: %', p_role;
    end if;

    select role into caller_role from org_members
    where org_id = p_org_id and user_id = auth.uid();

    if caller_role is null or caller_role not in ('owner', 'admin') then
        raise exception 'Only an owner or admin of this organization can invite members.';
    end if;

    select id into target_user_id from auth.users where email = p_email;
    if target_user_id is null then
        raise exception 'No registered user found with email %.', p_email;
    end if;

    insert into org_members (org_id, user_id, role)
    values (p_org_id, target_user_id, p_role)
    on conflict (org_id, user_id) do update set role = excluded.role
    returning * into new_member;

    return new_member;
end;
$$;

grant execute on function invite_member_to_org(uuid, text, text) to authenticated;

-- Fix: document_chunks only had a SELECT policy, which was fine for the
-- service-role client used everywhere so far, but blocks a real per-user JWT
-- client from writing chunks during ingestion. Bring it in line with every
-- other tenant-scoped table (projects/kb_categories/documents/research_sessions).
drop policy if exists "members can view chunks" on document_chunks;
create policy "members can manage chunks" on document_chunks
    for all using (is_org_member(org_id)) with check (is_org_member(org_id));
