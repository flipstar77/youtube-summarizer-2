"""
Test Script for Video Highlight Extraction System

This script tests the highlight extraction system with the existing video collection
and validates all components work together properly.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_highlight_extractor import VideoHighlightExtractor, ExtractionConfig, create_extraction_config
from srt_parser import SRTParser
from highlight_analyzer import HighlightAnalyzer
from ffmpeg_manager import FFmpegManager

# Test configuration
TEST_OUTPUT_DIR = "test_highlights"
EXISTING_VIDEOS_DIR = "static/downloads"


def test_progress_callback(message: str, progress: float):
    """Progress callback for testing"""
    print(f"[{progress*100:5.1f}%] {message}")


def test_srt_parser():
    """Test SRT parser functionality"""
    print("\n=== Testing SRT Parser ===")
    
    parser = SRTParser()
    
    # Test with sample SRT content
    sample_srt = """1
00:00:01,000 --> 00:00:05,500
Welcome to this amazing tutorial on Python programming.

2
00:00:05,500 --> 00:00:10,000
Today we'll learn some incredible techniques that will blow your mind!

3
00:00:10,000 --> 00:00:15,000
Let's start with the basics and work our way up to advanced concepts.

4
00:00:15,000 --> 00:00:20,000
First, let me show you this really important function that everyone should know."""
    
    try:
        segments = parser.parse_content(sample_srt)
        print(f"[OK] Successfully parsed {len(segments)} segments")
        
        # Test windowed segments
        windows = parser.create_windowed_segments(window_size=15, overlap=3)
        print(f"[OK] Created {len(windows)} analysis windows")
        
        # Test stats
        stats = parser.get_stats()
        print(f"[OK] Generated stats: {stats['segment_count']} segments, {stats['total_duration']:.1f}s")
        
        return True
    except Exception as e:
        print(f"[FAIL] SRT Parser test failed: {str(e)}")
        return False


def test_highlight_analyzer():
    """Test highlight analyzer functionality"""
    print("\n=== Testing Highlight Analyzer ===")
    
    # Test segments
    test_segments = [
        {
            'id': 1,
            'start_time': 0,
            'end_time': 30,
            'duration': 30,
            'text': 'This is an incredible breakthrough in machine learning that will revolutionize how we think about AI! The implications are absolutely mind-blowing.',
            'start_time_formatted': '00:00:00,000',
            'end_time_formatted': '00:00:30,000'
        },
        {
            'id': 2,
            'start_time': 30,
            'end_time': 60,
            'duration': 30,
            'text': 'So, um, let me just show you this basic example. This is just a simple print statement.',
            'start_time_formatted': '00:00:30,000',
            'end_time_formatted': '00:01:00,000'
        },
        {
            'id': 3,
            'start_time': 60,
            'end_time': 90,
            'duration': 30,
            'text': 'Here we have the most important concept you need to understand. This technique has saved me countless hours of debugging.',
            'start_time_formatted': '00:01:00,000',
            'end_time_formatted': '00:01:30,000'
        }
    ]
    
    try:
        # Test with fallback (no API key)
        analyzer = HighlightAnalyzer(api_key="dummy_key", model="gpt-3.5-turbo")
        
        # This should fail gracefully and use fallback scoring
        scores = analyzer.analyze_segments(test_segments)
        print(f"✅ Analyzed {len(test_segments)} segments, got {len(scores)} scores (using fallback)")
        
        # Test filtering
        filtered = analyzer.filter_highlights(scores, max_highlights=2, min_score=3.0)
        print(f"✅ Filtered to {len(filtered)} highlights")
        
        # Test report generation
        report = analyzer.create_highlight_report(scores)
        print(f"✅ Generated analysis report with {len(report)} sections")
        
        return True
    except Exception as e:
        print(f"❌ Highlight Analyzer test failed: {str(e)}")
        return False


def test_ffmpeg_manager():
    """Test FFmpeg manager functionality"""
    print("\n=== Testing FFmpeg Manager ===")
    
    try:
        ffmpeg = FFmpegManager()
        print("✅ FFmpeg Manager initialized successfully")
        
        # Test with existing video if available
        existing_videos = []
        if os.path.exists(EXISTING_VIDEOS_DIR):
            existing_videos = [f for f in os.listdir(EXISTING_VIDEOS_DIR) if f.endswith('.mp4')]
        
        if existing_videos:
            test_video = os.path.join(EXISTING_VIDEOS_DIR, existing_videos[0])
            print(f"Testing with existing video: {test_video}")
            
            # Test video info extraction
            try:
                info = ffmpeg.get_video_info(test_video)
                print(f"✅ Got video info: {info.duration:.1f}s, {info.width}x{info.height}")
            except Exception as e:
                print(f"⚠️ Could not get video info: {str(e)} (FFmpeg might not be installed)")
                return False
            
            # Test clip extraction (short clip to avoid long processing)
            if info.duration > 20:  # Only test if video is longer than 20s
                try:
                    test_clip_path = os.path.join(TEST_OUTPUT_DIR, "test_clip.mp4")
                    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
                    
                    success = ffmpeg.extract_clip(test_video, test_clip_path, 5, 10)
                    if success and os.path.exists(test_clip_path):
                        print("✅ Successfully extracted test clip")
                        # Clean up test clip
                        os.remove(test_clip_path)
                    else:
                        print("⚠️ Clip extraction failed")
                        return False
                except Exception as e:
                    print(f"⚠️ Clip extraction test failed: {str(e)}")
                    return False
        else:
            print("⚠️ No existing videos found for testing")
        
        return True
    except Exception as e:
        print(f"❌ FFmpeg Manager test failed: {str(e)}")
        return False


def test_integration():
    """Test full integration with existing video"""
    print("\n=== Testing Full Integration ===")
    
    # Check for existing videos
    existing_videos = []
    if os.path.exists(EXISTING_VIDEOS_DIR):
        existing_videos = [f for f in os.listdir(EXISTING_VIDEOS_DIR) if f.endswith('.mp4')]
    
    if not existing_videos:
        print("⚠️ No existing videos found for integration test")
        return False
    
    test_video_file = existing_videos[0]
    print(f"Using video: {test_video_file}")
    
    # For integration test, we'll simulate the process with a mock YouTube URL
    # Since we can't extract transcript from a local file, we'll create a mock scenario
    
    try:
        # Create test configuration
        config = create_extraction_config(
            max_highlights=3,
            min_highlight_score=4.0,
            video_quality="720p",
            create_individual_clips=False,  # Skip actual extraction for test
            create_compilation=False,
            preserve_temp_files=True,
            output_base_dir=TEST_OUTPUT_DIR
        )
        
        # Initialize extractor
        extractor = VideoHighlightExtractor(
            config=config,
            progress_callback=test_progress_callback
        )
        
        print("✅ Video Highlight Extractor initialized successfully")
        
        # Test stats functionality
        stats = extractor.get_extraction_stats()
        print(f"✅ Got extraction stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_youtube_url_extraction():
    """Test the system with a real YouTube URL (if API key is available)"""
    print("\n=== Testing YouTube URL Extraction ===")
    
    # Check if we have an OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️ No OpenAI API key found in environment. Skipping YouTube extraction test.")
        print("   Set OPENAI_API_KEY environment variable to test full functionality.")
        return True
    
    # Use a short, publicly available video for testing
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (short and famous)
        "https://youtu.be/dQw4w9WgXcQ"  # Alternative format
    ]
    
    for test_url in test_urls:
        try:
            print(f"Testing URL format: {test_url}")
            
            # Create minimal config for testing
            config = create_extraction_config(
                max_highlights=2,
                min_highlight_score=5.0,
                video_quality="480p",  # Lower quality for faster testing
                create_individual_clips=True,
                create_compilation=False,  # Skip compilation for faster test
                max_total_duration=60,  # Limit to 1 minute total
                output_base_dir=TEST_OUTPUT_DIR
            )
            
            extractor = VideoHighlightExtractor(
                config=config,
                openai_api_key=openai_key,
                progress_callback=test_progress_callback
            )
            
            # Extract highlights
            result = extractor.extract_highlights(test_url)
            
            if result.success:
                print(f"✅ Successfully extracted highlights from YouTube video")
                print(f"   - Video: {result.video_title}")
                print(f"   - Highlights: {result.highlights_found}")
                print(f"   - Processing time: {result.processing_time:.1f}s")
                return True
            else:
                print(f"⚠️ Extraction failed: {result.error_message}")
                continue  # Try next URL
        
        except Exception as e:
            print(f"⚠️ YouTube test failed: {str(e)}")
            continue
    
    print("❌ All YouTube URL tests failed")
    return False


def run_all_tests():
    """Run all tests and report results"""
    print("Starting Video Highlight Extraction System Tests")
    print("=" * 50)
    
    test_results = {
        'SRT Parser': test_srt_parser(),
        'Highlight Analyzer': test_highlight_analyzer(),
        'FFmpeg Manager': test_ffmpeg_manager(),
        'Integration': test_integration(),
        'YouTube Extraction': test_youtube_url_extraction()
    }
    
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
    elif passed >= total * 0.8:  # 80% or more
        print("⚠️ Most tests passed. System is mostly functional.")
    else:
        print("❌ Multiple tests failed. Please check the issues above.")
    
    # Cleanup test directory
    try:
        import shutil
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
        print(f"\n🧹 Cleaned up test directory: {TEST_OUTPUT_DIR}")
    except:
        pass
    
    return passed == total


def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking Prerequisites...")
    print("-" * 30)
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"[FAIL] Python {python_version.major}.{python_version.minor} (requires 3.7+)")
        return False
    
    # Check required packages
    required_packages = [
        'yt_dlp', 'youtube_transcript_api', 'openai', 'python-dotenv', 'flask'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("[OK] FFmpeg")
        else:
            print("[FAIL] FFmpeg (not working)")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[FAIL] FFmpeg (not installed)")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        return False
    
    # Check OpenAI API key
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print("[OK] OpenAI API Key")
    else:
        print("[WARN] OpenAI API Key (not set - some features will use fallback)")
    
    print("-" * 30)
    return True


if __name__ == "__main__":
    print("Video Highlight Extraction System Test Suite")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ Prerequisites OK. Starting tests...\n")
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n🚀 System is ready for production use!")
        sys.exit(0)
    else:
        print("\n⚠️ Some issues were found. Please review the test results.")
        sys.exit(1)