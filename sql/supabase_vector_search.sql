-- =========================================================
-- Supabase Vector Search (canonical)
-- Table col + index + 2 stable RPC functions
-- =========================================================

-- (Optional) Ensure pgvector is available
-- create extension if not exists vector;

-- 1) Embedding column (OpenAI 1536 dims)
alter table public.summaries
  add column if not exists embedding vector(1536);

-- 2) Clean up any previous vector indexes (idempotent)
drop index if exists idx_summaries_embedding;
drop index if exists summaries_embedding_idx;
drop index if exists idx_summaries_embedding_hnsw;

-- 3) HNSW index for cosine similarity (works on constrained plans)
create index idx_summaries_embedding_hnsw
  on public.summaries
  using hnsw (embedding vector_cosine_ops);

-- 4) Update stats for better plans
analyze public.summaries;

-- 5) Remove older function signatures to avoid ambiguity (idempotent)
drop function if exists public.search_summaries_by_similarity(vector(1536), real, integer);
drop function if exists public.search_summaries_by_similarity(vector(1536), double precision, integer);
drop function if exists public.find_similar_summaries(bigint, integer);

-- 6) RPC: search by query embedding
create or replace function public.search_summaries_by_similarity(
  query_embedding vector(1536),
  match_threshold real default 0.7,
  match_count integer default 5
)
returns table (
  id bigint,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamptz,
  similarity real
)
language sql
stable
as $
  select
    s.id,
    s.title,
    s.summary,
    s.url,
    s.video_id,
    s.summary_type,
    s.created_at,
    (1 - (s.embedding <=> query_embedding))::real as similarity
  from public.summaries s
  where s.embedding is not null
    and (1 - (s.embedding <=> query_embedding))::real > match_threshold
  order by s.embedding <=> query_embedding
  limit match_count
$;

-- 7) RPC: find items similar to a given summary row
create or replace function public.find_similar_summaries(
  summary_id bigint,
  match_count integer default 5
)
returns table (
  id bigint,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamptz,
  similarity real
)
language sql
stable
as $
  select
    s.id,
    s.title,
    s.summary,
    s.url,
    s.video_id,
    s.summary_type,
    s.created_at,
    (1 - (s.embedding <=> ref.embedding))::real as similarity
  from public.summaries s
  join public.summaries ref on ref.id = summary_id
  where s.id <> summary_id
    and s.embedding is not null
    and ref.embedding is not null
  order by s.embedding <=> ref.embedding
  limit match_count
$;

-- =========================================================
-- End
-- =========================================================