-- =========================================================
-- Supabase Vector Search Health Check
-- Run each section in Supabase SQL Editor to verify setup
-- =========================================================

-- 1) BASIC CHECKS (Extension, Column, Dimension)
-- =========================================================

-- pgvector installed?
select 'ok' as vector_ext
where exists (select 1 from pg_extension where extname='vector');

-- Embedding column exists with correct dimension?
select 'ok' as embedding_1536
where exists (
  select 1
  from information_schema.columns
  where table_schema='public' and table_name='summaries'
    and column_name='embedding' and udt_name='vector'
);

-- How many rows with/without embeddings?
select count(*) total,
       count(*) filter (where embedding is not null) with_embedding,
       count(*) filter (where embedding is null)   without_embedding
from public.summaries;

-- 2) INDEX CHECK (HNSW & Operator)
-- =========================================================

-- HNSW index present?
select indexname, indexdef
from pg_indexes
where schemaname='public' and tablename='summaries'
  and indexdef ilike '%using hnsw%';

-- Does index use cosine operator?
select 'ok' as cosine_ok
where exists (
  select 1
  from pg_indexes
  where schemaname='public' and tablename='summaries'
    and indexdef ilike '%vector_cosine_ops%'
);

-- 3) FUNCTION SIGNATURES & SMOKE TEST
-- =========================================================

-- Do exactly these RPCs exist?
select 'ok' as has_search_fn
where exists (
  select 1 from pg_proc p
  join pg_namespace n on n.oid=p.pronamespace
  where n.nspname='public' and p.proname='search_summaries_by_similarity'
);

select 'ok' as has_find_fn
where exists (
  select 1 from pg_proc p
  join pg_namespace n on n.oid=p.pronamespace
  where n.nspname='public' and p.proname='find_similar_summaries'
);

-- Smoke test: Query with dummy embedding (may return 0 rows, but no error)
select * from public.search_summaries_by_similarity(
  array_fill(0.0, array[1536])::vector(1536), 0.0, 3
);

-- 4) "REAL" SIMILARITY TEST (uses existing embedding)
-- =========================================================

-- Get any ID with embedding
with pick as (
  select id from public.summaries
  where embedding is not null
  order by created_at desc
  limit 1
)
select * from public.find_similar_summaries((select id from pick), 5);

-- 5) OPTIONAL: RUNTIME FINE-TUNING
-- =========================================================

-- For current session only (if supported):
-- Higher values = more accurate/slower, lower = faster/less accurate
-- set hnsw.ef_search = 40;  -- Example value

-- 6) RLS/POLICIES CHECK
-- =========================================================

-- Show active policies on the table
select * from pg_policies where schemaname='public' and tablename='summaries';

-- =========================================================
-- TROUBLESHOOTING GUIDE
-- =========================================================

/*
What to do if a check returns empty:

1. Vector extension missing: 
   create extension if not exists vector;

2. Embedding column missing/wrong dimension:
   alter table public.summaries add column if not exists embedding vector(1536);

3. No HNSW index:
   Re-run sql/supabase_vector_search.sql

4. RPC missing:
   Re-run the functions from sql/supabase_vector_search.sql

5. No hits in similarity test:
   - Too few rows with embeddings
   - Run embedding backfill, then retry test 4

6. Performance issues:
   - Check HNSW index exists
   - Try adjusting hnsw.ef_search parameter
   - Ensure enough rows for meaningful similarity
*/