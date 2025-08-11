#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from transcript_extractor import TranscriptExtractor
from summarizer import TextSummarizer
from supabase_client import SupabaseDatabase

load_dotenv()

def quick_test():
    """Quick test of the complete workflow"""
    print("Quick Test - YouTube Summarizer Workflow")
    print("=" * 50)
    
    try:
        # Test URL
        test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
        
        # Step 1: Extract transcript
        print("1. Extracting transcript...")
        extractor = TranscriptExtractor()
        transcript_data = extractor.get_transcript(test_url, 'en')
        print(f"   [OK] Transcript extracted: {len(transcript_data['transcript'])} chars")
        
        # Step 2: Generate summary
        print("2. Generating summary...")
        api_key = os.getenv('OPENAI_API_KEY')
        summarizer = TextSummarizer(api_key)
        summary = summarizer.summarize(transcript_data['transcript'], 'brief')
        print(f"   [OK] Summary generated: {len(summary)} chars")
        
        # Step 3: Save to database
        print("3. Saving to database...")
        db = SupabaseDatabase()
        summary_id = db.save_summary(
            video_id=transcript_data['video_id'],
            url=test_url,
            title=f"Test Video {transcript_data['video_id']}",
            summary_type='brief',
            summary=summary,
            transcript_length=len(transcript_data['transcript'])
        )
        print(f"   [OK] Saved to database with ID: {summary_id}")
        
        # Step 4: Verify
        print("4. Verifying save...")
        saved_summary = db.get_summary(summary_id)
        if saved_summary:
            print(f"   [OK] Summary verified in database")
            print(f"   Video ID: {saved_summary['video_id']}")
            print(f"   Summary: {saved_summary['summary'][:100]}...")
        else:
            print(f"   [ERROR] Summary not found after saving")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n[SUCCESS] Complete workflow is working!")
    else:
        print("\n[FAILED] Workflow has issues")