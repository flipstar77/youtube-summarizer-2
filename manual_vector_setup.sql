-- Manual Vector Database Setup for Supabase
-- Run these commands one by one in the Supabase SQL Editor
-- Go to: https://app.supabase.com/project/[your-project]/sql

-- Step 1: Enable the vector extension
-- (This might already be enabled, if so you'll get a notice)
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Add the embedding column to summaries table
-- This adds a vector column with 1536 dimensions (for OpenAI text-embedding-ada-002)
ALTER TABLE summaries 
ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- Step 3: Create an index for fast similarity search
-- This index makes vector similarity searches much faster
CREATE INDEX IF NOT EXISTS summaries_embedding_idx 
ON summaries USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Step 4: Create function for semantic search
CREATE OR REPLACE FUNCTION search_summaries_by_similarity(
  query_embedding vector(1536),
  match_threshold float DEFAULT 0.7,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id int,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamp,
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

-- Step 5: Create function to find similar content to a specific summary
CREATE OR REPLACE FUNCTION find_similar_summaries(
  summary_id int,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id int,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamp,
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

-- Step 6: Create helper function to get summaries without embeddings
CREATE OR REPLACE FUNCTION get_summaries_without_embeddings()
RETURNS TABLE (
  id int,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamp,
  transcript_length int
)
LANGUAGE sql
AS $$
  SELECT id, title, summary, url, video_id, summary_type, created_at, transcript_length
  FROM summaries
  WHERE embedding IS NULL;
$$;

-- Step 7: Verify the setup
-- Check if the embedding column was added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'summaries' AND column_name = 'embedding';

-- Check how many summaries need embeddings
SELECT COUNT(*) as summaries_without_embeddings 
FROM summaries 
WHERE embedding IS NULL;

-- Check how many summaries already have embeddings
SELECT COUNT(*) as summaries_with_embeddings 
FROM summaries 
WHERE embedding IS NOT NULL;

-- Test the search function with a dummy query
SELECT 'Vector functions created successfully' as status;