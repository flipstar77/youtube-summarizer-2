#!/usr/bin/env python3

from transcript_extractor import TranscriptExtractor

def test_transcript_extraction():
    """Test transcript extraction with a known working video"""
    extractor = TranscriptExtractor()
    
    # Test with a video that should have transcripts
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" - first YouTube video
    
    try:
        result = extractor.get_transcript(test_url)
        print(f"✅ Transcript extracted successfully!")
        print(f"Video ID: {result['video_id']}")
        print(f"Transcript length: {len(result['transcript'])} characters")
        print(f"First 200 characters: {result['transcript'][:200]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_url_parsing():
    """Test URL parsing functionality"""
    extractor = TranscriptExtractor()
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
    ]
    
    expected_id = "dQw4w9WgXcQ"
    
    for url in test_urls:
        try:
            video_id = extractor.extract_video_id(url)
            if video_id == expected_id:
                print(f"✅ URL parsing successful: {url}")
            else:
                print(f"❌ URL parsing failed: {url} -> {video_id}")
        except Exception as e:
            print(f"❌ URL parsing error: {url} -> {str(e)}")

if __name__ == "__main__":
    print("Testing YouTube Summarizer Components")
    print("=" * 40)
    
    print("\n1. Testing URL parsing...")
    test_url_parsing()
    
    print("\n2. Testing transcript extraction...")
    print("(This requires internet connection)")
    test_transcript_extraction()