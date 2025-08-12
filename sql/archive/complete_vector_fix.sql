-- Complete Vector Functions Fix for Supabase
-- This handles all possible function signatures and type mismatches

-- Drop all possible variations of the search function
DROP FUNCTION IF EXISTS search_summaries_by_similarity(vector, float, int);
DROP FUNCTION IF EXISTS search_summaries_by_similarity(vector(1536), float, int);

-- Drop all possible variations of the similar summaries function  
DROP FUNCTION IF EXISTS find_similar_summaries(int, int);
DROP FUNCTION IF EXISTS find_similar_summaries(bigint, int);
DROP FUNCTION IF EXISTS find_similar_summaries(integer, integer);
DROP FUNCTION IF EXISTS find_similar_summaries(bigint, integer);

-- Drop the helper function
DROP FUNCTION IF EXISTS get_summaries_without_embeddings();

-- Recreate search function with correct Supabase types
CREATE OR REPLACE FUNCTION search_summaries_by_similarity(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id bigint,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamptz,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    s.id,
    s.title,
    s.summary,
    s.url,
    s.video_id,
    s.summary_type,
    s.created_at,
    1 - (s.embedding <=> query_embedding) as similarity
  FROM summaries s
  WHERE s.embedding IS NOT NULL
    AND 1 - (s.embedding <=> query_embedding) > match_threshold
  ORDER BY s.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Recreate similar summaries function with correct types
CREATE OR REPLACE FUNCTION find_similar_summaries(
  summary_id bigint,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id bigint,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamptz,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    s.id,
    s.title,
    s.summary,
    s.url,
    s.video_id,
    s.summary_type,
    s.created_at,
    1 - (s.embedding <=> ref.embedding) as similarity
  FROM summaries s
  CROSS JOIN summaries ref
  WHERE s.id != summary_id
    AND ref.id = summary_id
    AND s.embedding IS NOT NULL
    AND ref.embedding IS NOT NULL
  ORDER BY s.embedding <=> ref.embedding
  LIMIT match_count;
END;
$$;

-- Recreate helper function with correct types
CREATE OR REPLACE FUNCTION get_summaries_without_embeddings()
RETURNS TABLE (
  id bigint,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamptz,
  transcript_length int
)
LANGUAGE sql
AS $$
  SELECT id, title, summary, url, video_id, summary_type, created_at, transcript_length
  FROM summaries
  WHERE embedding IS NULL;
$$;

-- Test the functions
SELECT 'All vector functions updated successfully with correct Supabase types' as status;