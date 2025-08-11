#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Highlight & Chaptering System
"""

import json
from video_highlight_extractor import VideoHighlightExtractor

def test_enhanced_system():
    """Test the enhanced highlight and chaptering system"""
    
    print("🎬 Enhanced Highlight & Chaptering System Test")
    print("=" * 70)
    
    # Test Database-ID 19
    summary_id = 19
    
    try:
        extractor = VideoHighlightExtractor()
        
        print(f"🔍 Testing Video ID: {summary_id}")
        
        # Test complete extraction with both highlights and chapters
        result = extractor.extract_highlights_from_video(
            video_id=str(summary_id),
            video_url="https://www.youtube.com/watch?v=0trT-uZ8Rgw",  # This will be looked up from DB
            srt_content=None,  # Let it generate
            highlight_count=5,
            min_duration=15,
            max_duration=60
        )
        
        print("\n" + "=" * 70)
        print("📊 COMPLETE RESULTS:")
        print("=" * 70)
        
        if result.get('status') == 'success':
            print(f"✅ Status: SUCCESS")
            print(f"📹 Video ID: {result.get('video_id')}")
            print(f"🎯 Highlights Identified: {result.get('highlights_identified', 0)}")
            print(f"🎬 Clips Extracted: {result.get('clips_extracted', 0)}")
            print(f"❌ Clips Failed: {result.get('clips_failed', 0)}")
            
            # Show highlights
            clips = result.get('clips', [])
            if clips:
                print(f"\n🌟 HIGHLIGHTS FOUND ({len(clips)}):")
                print("-" * 50)
                for i, clip in enumerate(clips, 1):
                    status_emoji = "✅" if clip.get('status') == 'extracted' else "⏰"
                    print(f"{status_emoji} Highlight {i}: {clip['start_time']} - {clip['end_time']}")
                    print(f"   📝 Text: {clip['text'][:100]}...")
                    print(f"   📈 Score: {clip['score']:.1f}/10")
                    print(f"   💡 Reason: {clip['reason']}")
                    print(f"   ⏱️  Duration: {clip['duration']}")
                    print()
            
            # Show chapters
            chapters = result.get('chapters', [])
            if chapters:
                print(f"📚 VIDEO CHAPTERS ({len(chapters)}):")
                print("-" * 50)
                for i, chapter in enumerate(chapters, 1):
                    print(f"📖 Chapter {i}: {chapter['timestamp']} - {chapter['title']}")
                    print(f"   📄 Description: {chapter['description']}")
                    print(f"   🔗 YouTube Link: &t={chapter['timestamp_seconds']}s")
                    print()
            
            # Show YouTube description
            youtube_desc = result.get('youtube_description', '')
            if youtube_desc:
                print("📝 YOUTUBE DESCRIPTION:")
                print("-" * 50)
                print(youtube_desc)
            
        else:
            print(f"❌ Status: ERROR")
            print(f"💥 Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_system()