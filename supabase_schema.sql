-- YouTube Summarizer Database Schema for Supabase
-- This file contains the database schema for the YouTube Summarizer application

-- Enable Row Level Security (RLS)
-- Create summaries table
CREATE TABLE IF NOT EXISTS public.summaries (
    id BIGSERIAL PRIMARY KEY,
    video_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    summary_type TEXT NOT NULL CHECK (summary_type IN ('brief', 'detailed', 'bullet')),
    summary TEXT NOT NULL,
    transcript_length INTEGER DEFAULT 0,
    audio_file TEXT,
    voice_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::TEXT, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::TEXT, NOW()) NOT NULL,
    
    -- Add constraints
    CONSTRAINT valid_video_id CHECK (LENGTH(video_id) > 0),
    CONSTRAINT valid_url CHECK (url LIKE 'https://www.youtube.com%' OR url LIKE 'https://youtu.be%'),
    CONSTRAINT valid_summary CHECK (LENGTH(summary) > 10)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_summaries_video_id ON public.summaries(video_id);
CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON public.summaries(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_summary_type ON public.summaries(summary_type);

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::TEXT, NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-update updated_at
DROP TRIGGER IF EXISTS set_updated_at ON public.summaries;
CREATE TRIGGER set_updated_at
    BEFORE UPDATE ON public.summaries
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- Enable RLS on the summaries table
ALTER TABLE public.summaries ENABLE ROW LEVEL SECURITY;

-- Create a policy that allows all operations for now (you can make this more restrictive later)
CREATE POLICY "Allow all operations on summaries" ON public.summaries
    FOR ALL USING (true)
    WITH CHECK (true);

-- Create a view for summary statistics
CREATE OR REPLACE VIEW public.summary_stats AS
SELECT 
    COUNT(*) as total_summaries,
    COUNT(CASE WHEN audio_file IS NOT NULL THEN 1 END) as summaries_with_audio,
    COUNT(CASE WHEN created_at >= NOW() - INTERVAL '7 days' THEN 1 END) as recent_summaries,
    AVG(transcript_length) as avg_transcript_length,
    summary_type,
    COUNT(*) as type_count
FROM public.summaries 
GROUP BY summary_type
ORDER BY type_count DESC;

-- Create a function to get summaries with pagination
CREATE OR REPLACE FUNCTION public.get_summaries_paginated(
    page_size INTEGER DEFAULT 20,
    page_offset INTEGER DEFAULT 0,
    summary_type_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    video_id TEXT,
    url TEXT,
    title TEXT,
    summary_type TEXT,
    summary TEXT,
    transcript_length INTEGER,
    audio_file TEXT,
    voice_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.id, s.video_id, s.url, s.title, s.summary_type, 
        s.summary, s.transcript_length, s.audio_file, s.voice_id,
        s.created_at, s.updated_at
    FROM public.summaries s
    WHERE (summary_type_filter IS NULL OR s.summary_type = summary_type_filter)
    ORDER BY s.created_at DESC
    LIMIT page_size
    OFFSET page_offset;
END;
$$ LANGUAGE plpgsql;

-- Create a function to search summaries
CREATE OR REPLACE FUNCTION public.search_summaries(search_query TEXT)
RETURNS TABLE (
    id BIGINT,
    video_id TEXT,
    url TEXT,
    title TEXT,
    summary_type TEXT,
    summary TEXT,
    transcript_length INTEGER,
    audio_file TEXT,
    voice_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    rank REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.id, s.video_id, s.url, s.title, s.summary_type, 
        s.summary, s.transcript_length, s.audio_file, s.voice_id,
        s.created_at, s.updated_at,
        ts_rank(
            to_tsvector('english', COALESCE(s.title, '') || ' ' || s.summary),
            plainto_tsquery('english', search_query)
        ) as rank
    FROM public.summaries s
    WHERE to_tsvector('english', COALESCE(s.title, '') || ' ' || s.summary) 
          @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC, s.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Add some sample data for testing (optional)
-- INSERT INTO public.summaries (video_id, url, title, summary_type, summary, transcript_length) 
-- VALUES 
--     ('dQw4w9WgXcQ', 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'Rick Astley - Never Gonna Give You Up', 'brief', 'A classic music video that became an internet meme.', 1500),
--     ('jNQXAC9IVRw', 'https://www.youtube.com/watch?v=jNQXAC9IVRw', 'Me at the zoo', 'detailed', 'The first ever video uploaded to YouTube by co-founder Jawed Karim at the San Diego Zoo.', 800);

COMMENT ON TABLE public.summaries IS 'Stores YouTube video summaries with metadata';
COMMENT ON COLUMN public.summaries.video_id IS 'YouTube video ID extracted from URL';
COMMENT ON COLUMN public.summaries.url IS 'Full YouTube URL';
COMMENT ON COLUMN public.summaries.title IS 'Video title (can be enhanced with YouTube API)';
COMMENT ON COLUMN public.summaries.summary_type IS 'Type of summary: brief, detailed, or bullet';
COMMENT ON COLUMN public.summaries.summary IS 'AI-generated summary text';
COMMENT ON COLUMN public.summaries.transcript_length IS 'Length of original transcript in characters';
COMMENT ON COLUMN public.summaries.audio_file IS 'Path to generated audio file';
COMMENT ON COLUMN public.summaries.voice_id IS 'ElevenLabs voice ID used for audio generation';