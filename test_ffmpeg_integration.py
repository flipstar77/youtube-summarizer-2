#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick FFmpeg Integration Test
"""

import os
import subprocess
from video_highlight_extractor import VideoHighlightExtractor

def test_ffmpeg_basic():
    """Test basic FFmpeg functionality"""
    print("🔧 Testing FFmpeg Basic Functionality")
    print("=" * 50)
    
    # Test FFmpeg command directly
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is accessible via command line")
            version_line = result.stdout.split('\n')[0]
            print(f"   Version: {version_line}")
        else:
            print("❌ FFmpeg command failed")
            return False
    except Exception as e:
        print(f"❌ FFmpeg not found: {str(e)}")
        return False
    
    return True

def test_video_clip_extraction():
    """Test actual video clip extraction"""
    print("\n🎬 Testing Video Clip Extraction")
    print("=" * 50)
    
    # Check if we have a test video
    temp_dir = "D:/mcp/temp"
    test_video = None
    
    for file in os.listdir(temp_dir):
        if file.endswith('.mp4'):
            test_video = os.path.join(temp_dir, file)
            print(f"✅ Found test video: {file}")
            break
    
    if not test_video:
        print("❌ No test video found in temp directory")
        return False
    
    # Test a simple clip extraction
    extractor = VideoHighlightExtractor()
    
    # Create a test segment
    test_segment = {
        "start_time": "00:00:10,000",
        "end_time": "00:00:15,000", 
        "start_seconds": 10.0,
        "end_seconds": 15.0,
        "text": "Test segment for FFmpeg verification"
    }
    
    print("🎯 Testing 5-second clip extraction (10s-15s)...")
    
    try:
        clip_path = extractor._extract_video_clip(
            test_video, test_segment, "test", 1
        )
        
        if clip_path and os.path.exists(clip_path):
            file_size = os.path.getsize(clip_path)
            print(f"✅ Clip extracted successfully!")
            print(f"   Output: {os.path.basename(clip_path)}")
            print(f"   Size: {file_size / 1024:.1f} KB")
            
            # Verify with ffprobe if available
            try:
                probe_result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    clip_path
                ], capture_output=True, text=True)
                
                if probe_result.returncode == 0:
                    duration = float(probe_result.stdout.strip())
                    print(f"   Duration: {duration:.2f} seconds")
                    if 4.5 <= duration <= 5.5:  # Allow some tolerance
                        print("✅ Duration verification passed")
                    else:
                        print(f"⚠️  Duration seems off (expected ~5s)")
                
            except Exception as e:
                print(f"⚠️  Could not verify duration: {str(e)}")
            
            return True
        else:
            print("❌ Clip extraction returned no file")
            return False
            
    except Exception as e:
        print(f"❌ Clip extraction failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 FFmpeg Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Basic FFmpeg availability
    if not test_ffmpeg_basic():
        print("\n❌ FFmpeg basic test failed - stopping here")
        return
    
    # Test 2: Video clip extraction  
    if test_video_clip_extraction():
        print(f"\n✅ ALL TESTS PASSED!")
        print("🎉 FFmpeg integration is working correctly!")
    else:
        print(f"\n❌ Clip extraction test failed")
        print("💡 Check video files and FFmpeg configuration")

if __name__ == "__main__":
    main()