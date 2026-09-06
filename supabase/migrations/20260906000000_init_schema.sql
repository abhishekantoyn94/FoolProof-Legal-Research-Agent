-- Phase 1 foundation schema.
-- Multi-tenant model: organizations -> (Projects | Knowledge Base categories) -> Documents -> Chunks.
-- A document belongs to exactly one Project OR one KB category (its retrieval "collection").
-- Row Level Security enforces org isolation throughout (master prompt section 63).

create extension if not exists vector;
create extension if not exists pgcrypto; -- for gen_random_uuid()

-- ---------------------------------------------------------------------------
-- Organizations & membership
-- ---------------------------------------------------------------------------

create table organizations (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    -- Per-org privacy default; a session may not exceed it (enforced in application code).
    privacy_mode text not null default 'hybrid'
        check (privacy_mode in ('local_only', 'hybrid', 'cloud_assisted')),
    active_provider_profile text not null default 'hybrid_openai_default',
    created_at timestamptz not null default now()
);

create table org_members (
    org_id uuid not null references organizations(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    role text not null default 'member' check (role in ('owner', 'admin', 'member')),
    created_at timestamptz not null default now(),
    primary key (org_id, user_id)
);

-- ---------------------------------------------------------------------------
-- Projects (matter-scoped) and Knowledge Base categories (org-wide, reusable)
-- ---------------------------------------------------------------------------

create table projects (
    id uuid primary key default gen_random_uuid(),
    org_id uuid not null references organizations(id) on delete cascade,
    name text not null,
    description text,
    jurisdiction text,
    status text not null default 'active' check (status in ('active', 'archived')),
    created_by uuid references auth.users(id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table kb_categories (
    id uuid primary key default gen_random_uuid(),
    org_id uuid not null references organizations(id) on delete cascade,
    name text not null,
    description text,
    jurisdiction text,
    created_by uuid references auth.users(id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (org_id, name)
);

-- ---------------------------------------------------------------------------
-- Documents & chunks
-- ---------------------------------------------------------------------------

create table documents (
    id uuid primary key default gen_random_uuid(),
    org_id uuid not null references organizations(id) on delete cascade,
    project_id uuid references projects(id) on delete cascade,
    kb_category_id uuid references kb_categories(id) on delete cascade,
    filename text not null,
    storage_path text not null,
    mime_type text,
    file_hash text,
    page_count int,
    status text not null default 'pending'
        check (status in ('pending', 'processing', 'indexed', 'error')),
    error_message text,
    uploaded_by uuid references auth.users(id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint document_belongs_to_exactly_one_collection
        check (num_nonnulls(project_id, kb_category_id) = 1)
);

create index documents_org_id_idx on documents(org_id);
create index documents_project_id_idx on documents(project_id) where project_id is not null;
create index documents_kb_category_id_idx on documents(kb_category_id) where kb_category_id is not null;

-- BGE-M3 embedding dimension is 1024. Embedding metadata is tracked per-chunk so
-- changing the embedding model later is a visible migration, not silent drift
-- (master prompt section 21).
create table document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid not null references documents(id) on delete cascade,
    org_id uuid not null references organizations(id) on delete cascade,
    chunk_index int not null,
    content text not null,
    heading text,
    section text,
    page_number int,
    token_count int,
    embedding vector(1024),
    embedding_model text not null default 'BAAI/bge-m3',
    embedding_version int not null default 1,
    content_tsv tsvector generated always as (to_tsvector('english', content)) stored,
    created_at timestamptz not null default now(),
    unique (document_id, chunk_index)
);

create index document_chunks_org_id_idx on document_chunks(org_id);
create index document_chunks_document_id_idx on document_chunks(document_id);
-- Lexical/BM25-equivalent retrieval (master prompt section 22).
create index document_chunks_tsv_idx on document_chunks using gin(content_tsv);
-- Dense retrieval. ivfflat requires an approximate row-count estimate at build
-- time; fine to add now and REINDEX later once real corpus size is known.
create index document_chunks_embedding_idx on document_chunks
    using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- ---------------------------------------------------------------------------
-- Research sessions
-- Findings/evidence/operational-trace shapes are intentionally kept as jsonb
-- for now -- Phase 5 (research agent) will define and normalize their real
-- structure once the agent loop exists; normalizing prematurely here would be
-- guessing at a shape we don't have yet.
-- ---------------------------------------------------------------------------

create table research_sessions (
    id uuid primary key default gen_random_uuid(),
    org_id uuid not null references organizations(id) on delete cascade,
    project_id uuid references projects(id) on delete set null,
    created_by uuid references auth.users(id),
    question text not null,
    jurisdiction text,
    research_mode text not null default 'standard'
        check (research_mode in ('quick', 'standard', 'deep')),
    privacy_mode text not null,
    provider_profile text not null,
    status text not null default 'pending'
        check (status in ('pending', 'planning', 'running', 'paused', 'completed', 'failed')),
    operational_trace jsonb not null default '[]'::jsonb,
    final_answer jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index research_sessions_org_id_idx on research_sessions(org_id);

-- ---------------------------------------------------------------------------
-- Row Level Security: every tenant-scoped table is only visible to members
-- of the owning organization.
-- ---------------------------------------------------------------------------

create or replace function is_org_member(target_org_id uuid)
returns boolean
language sql
security definer
stable
as $$
    select exists (
        select 1 from org_members
        where org_id = target_org_id and user_id = auth.uid()
    );
$$;

alter table organizations enable row level security;
alter table org_members enable row level security;
alter table projects enable row level security;
alter table kb_categories enable row level security;
alter table documents enable row level security;
alter table document_chunks enable row level security;
alter table research_sessions enable row level security;

create policy "members can view their orgs" on organizations
    for select using (is_org_member(id));

create policy "members can view org membership" on org_members
    for select using (is_org_member(org_id));

create policy "members can manage projects" on projects
    for all using (is_org_member(org_id)) with check (is_org_member(org_id));

create policy "members can manage kb categories" on kb_categories
    for all using (is_org_member(org_id)) with check (is_org_member(org_id));

create policy "members can manage documents" on documents
    for all using (is_org_member(org_id)) with check (is_org_member(org_id));

create policy "members can view chunks" on document_chunks
    for select using (is_org_member(org_id));

create policy "members can manage research sessions" on research_sessions
    for all using (is_org_member(org_id)) with check (is_org_member(org_id));
