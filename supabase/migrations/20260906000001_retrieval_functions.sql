-- Retrieval functions (master prompt section 22): dense (pgvector) and
-- lexical (Postgres full-text, BM25-equivalent) search, each independently
-- filterable by project/KB-category/document scope. Fusion of the two ranked
-- lists happens in application code (Reciprocal Rank Fusion) since combining
-- a cosine distance and a ts_rank score meaningfully requires rank-based
-- fusion, not naive score blending across incomparable scales.
--
-- Boolean/exact-phrase search (section 22) is deliberately NOT a bespoke
-- parser: Postgres's built-in `websearch_to_tsquery` already supports quoted
-- phrases, implicit AND, "OR", and a "-" exclusion prefix -- exactly what
-- Synonimise (Phase 4) needs to emit, so lexical search below is driven by it
-- directly rather than reinventing that syntax.

create or replace function match_document_chunks_dense(
    p_org_id uuid,
    p_query_embedding vector(1024),
    p_project_ids uuid[] default null,
    p_kb_category_ids uuid[] default null,
    p_document_ids uuid[] default null,
    p_match_count int default 20
)
returns table (
    chunk_id uuid,
    document_id uuid,
    content text,
    heading text,
    section text,
    page_number int,
    distance float
)
language sql stable
as $$
    select
        dc.id as chunk_id,
        dc.document_id,
        dc.content,
        dc.heading,
        dc.section,
        dc.page_number,
        dc.embedding <=> p_query_embedding as distance
    from document_chunks dc
    join documents d on d.id = dc.document_id
    where dc.org_id = p_org_id
      and dc.embedding is not null
      and (p_document_ids is null or dc.document_id = any(p_document_ids))
      and (p_project_ids is null or d.project_id = any(p_project_ids))
      and (p_kb_category_ids is null or d.kb_category_id = any(p_kb_category_ids))
    order by dc.embedding <=> p_query_embedding
    limit p_match_count;
$$;

create or replace function match_document_chunks_lexical(
    p_org_id uuid,
    p_query text,
    p_project_ids uuid[] default null,
    p_kb_category_ids uuid[] default null,
    p_document_ids uuid[] default null,
    p_match_count int default 20
)
returns table (
    chunk_id uuid,
    document_id uuid,
    content text,
    heading text,
    section text,
    page_number int,
    rank float
)
language sql stable
as $$
    select
        dc.id as chunk_id,
        dc.document_id,
        dc.content,
        dc.heading,
        dc.section,
        dc.page_number,
        ts_rank(dc.content_tsv, websearch_to_tsquery('english', p_query)) as rank
    from document_chunks dc
    join documents d on d.id = dc.document_id
    where dc.org_id = p_org_id
      and dc.content_tsv @@ websearch_to_tsquery('english', p_query)
      and (p_document_ids is null or dc.document_id = any(p_document_ids))
      and (p_project_ids is null or d.project_id = any(p_project_ids))
      and (p_kb_category_ids is null or d.kb_category_id = any(p_kb_category_ids))
    order by rank desc
    limit p_match_count;
$$;

grant execute on function match_document_chunks_dense(uuid, vector, uuid[], uuid[], uuid[], int)
    to authenticated, service_role;
grant execute on function match_document_chunks_lexical(uuid, text, uuid[], uuid[], uuid[], int)
    to authenticated, service_role;
