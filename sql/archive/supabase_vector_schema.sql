-- Enable the vector extension in Supabase
CREATE EXTENSION IF NOT EXISTS vector;

-- Add vector column to summaries table
ALTER TABLE summaries 
ADD COLUMN IF NOT EXISTS embedding vector(1536);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS summaries_embedding_idx 
ON summaries USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create a function for semantic search
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

-- Create a function to find similar content
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

-- Create a function to get summaries without embeddings
CREATE OR REPLACE FUNCTION get_summaries_without_embeddings()
RETURNS TABLE (
  id int,
  title text,
  summary text,
  url text,
  video_id text,
  summary_type text,
  created_at timestamp
)
LANGUAGE sql
AS $$
  SELECT id, title, summary, url, video_id, summary_type, created_at
  FROM summaries
  WHERE embedding IS NULL;
$$;