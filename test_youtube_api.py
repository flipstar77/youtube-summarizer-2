#!/usr/bin/env python3

from youtube_transcript_api import YouTubeTranscriptApi

# Test the API to see available methods
print("Testing YouTube Transcript API...")
print("Available methods:", dir(YouTubeTranscriptApi))

# Test with a simple video
video_id = "jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video

try:
    # Try creating an instance first
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    print(f"Success with instance: Got {len(transcript)} transcript entries")
    print("First entry:", transcript[0])
except Exception as e:
    print(f"Error with instance: {str(e)}")

try:
    # Check the actual API documentation approach
    from youtube_transcript_api import YouTubeTranscriptApi as YTAPI
    
    # Let's check what actually works by trying different approaches
    try:
        transcript = YTAPI.get_transcript(video_id)
        print("get_transcript method works!")
    except:
        try:
            transcript = YTAPI.fetch(video_id)
            print("fetch method works!")
        except:
            print("Neither method works - checking version...")
            import youtube_transcript_api
            print(f"Version: {youtube_transcript_api.__version__ if hasattr(youtube_transcript_api, '__version__') else 'unknown'}")
            
except Exception as e:
    print(f"API test error: {str(e)}")