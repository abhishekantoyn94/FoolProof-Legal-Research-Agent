-- Small supporting RPC for the invite UI: shows who's already in an org.
-- auth.users isn't exposed via the REST API directly, and is_org_member()
-- inside the WHERE clause still evaluates against the real caller (auth.uid()
-- is unaffected by security definer), so this only ever returns rows for an
-- org the calling user actually belongs to.

create or replace function list_org_members(p_org_id uuid)
returns table(user_id uuid, email text, role text, joined_at timestamptz)
language sql
security definer
set search_path = public
stable
as $$
    select m.user_id, u.email, m.role, m.created_at
    from org_members m
    join auth.users u on u.id = m.user_id
    where m.org_id = p_org_id and is_org_member(p_org_id);
$$;

grant execute on function list_org_members(uuid) to authenticated;
