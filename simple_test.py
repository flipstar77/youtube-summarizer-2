"""
Simple Test for Video Highlight Extraction System
"""

import os
import sys

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from srt_parser import SRTParser
        print("[OK] srt_parser imported")
    except Exception as e:
        print(f"[FAIL] srt_parser: {e}")
        return False
    
    try:
        from highlight_analyzer import HighlightAnalyzer
        print("[OK] highlight_analyzer imported")
    except Exception as e:
        print(f"[FAIL] highlight_analyzer: {e}")
        return False
    
    try:
        from ffmpeg_manager import FFmpegManager
        print("[OK] ffmpeg_manager imported")
    except Exception as e:
        print(f"[FAIL] ffmpeg_manager: {e}")
        return False
    
    try:
        from video_highlight_extractor import VideoHighlightExtractor
        print("[OK] video_highlight_extractor imported")
    except Exception as e:
        print(f"[FAIL] video_highlight_extractor: {e}")
        return False
    
    return True

def test_srt_parser():
    """Test basic SRT parsing"""
    print("\nTesting SRT Parser...")
    
    try:
        from srt_parser import SRTParser
        
        parser = SRTParser()
        
        # Test sample SRT
        sample_srt = """1
00:00:01,000 --> 00:00:05,500
Hello world test subtitle

2
00:00:05,500 --> 00:00:10,000
This is a second subtitle"""
        
        segments = parser.parse_content(sample_srt)
        
        if len(segments) == 2:
            print(f"[OK] Parsed {len(segments)} segments correctly")
            return True
        else:
            print(f"[FAIL] Expected 2 segments, got {len(segments)}")
            return False
            
    except Exception as e:
        print(f"[FAIL] SRT Parser test: {e}")
        return False

def test_ffmpeg_check():
    """Test FFmpeg availability"""
    print("\nTesting FFmpeg availability...")
    
    try:
        from ffmpeg_manager import FFmpegManager
        
        # Try to create FFmpeg manager
        ffmpeg = FFmpegManager()
        print("[OK] FFmpeg Manager created successfully")
        
        # Check if we have existing video files
        videos_dir = "static/downloads"
        if os.path.exists(videos_dir):
            videos = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
            if videos:
                test_video = os.path.join(videos_dir, videos[0])
                try:
                    info = ffmpeg.get_video_info(test_video)
                    print(f"[OK] Got video info for {videos[0]}: {info.duration:.1f}s")
                    return True
                except Exception as e:
                    print(f"[FAIL] Could not get video info: {e}")
                    return False
            else:
                print("[WARN] No video files found for testing")
                return True
        else:
            print("[WARN] Downloads directory not found")
            return True
            
    except Exception as e:
        print(f"[FAIL] FFmpeg test: {e}")
        return False

def test_system_integration():
    """Test basic system integration"""
    print("\nTesting system integration...")
    
    try:
        from video_highlight_extractor import VideoHighlightExtractor, create_extraction_config
        
        # Create test config
        config = create_extraction_config(
            max_highlights=2,
            min_highlight_score=5.0
        )
        
        # Create extractor
        extractor = VideoHighlightExtractor(config=config)
        print("[OK] VideoHighlightExtractor created successfully")
        
        # Test stats
        stats = extractor.get_extraction_stats()
        print(f"[OK] Got stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Integration test: {e}")
        return False

def run_simple_tests():
    """Run all simple tests"""
    print("=== Video Highlight Extraction System - Simple Tests ===")
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("SRT Parser Test", test_srt_parser),
        ("FFmpeg Test", test_ffmpeg_check),
        ("Integration Test", test_system_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
    
    print()
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("All tests passed! System is ready.")
    elif passed >= total * 0.75:
        print("Most tests passed. System is mostly functional.")
    else:
        print("Several tests failed. Please check the issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)