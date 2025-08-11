#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from text_to_speech import TextToSpeech

load_dotenv()

def test_elevenlabs_connection():
    """Test ElevenLabs API connection"""
    try:
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            print("❌ ELEVENLABS_API_KEY not found in .env file")
            return False
        
        print(f"✅ API key found: {api_key[:8]}...")
        
        tts = TextToSpeech(api_key)
        print("✅ TextToSpeech object created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_voice_list():
    """Test getting available voices"""
    try:
        api_key = os.getenv('ELEVENLABS_API_KEY')
        tts = TextToSpeech(api_key)
        
        print("\n📋 Testing voice list retrieval...")
        voices = tts.get_available_voices()
        
        print(f"✅ Found {len(voices)} voices:")
        for voice in voices[:3]:  # Show first 3 voices
            print(f"   - {voice['name']}: {voice['description']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting voices: {str(e)}")
        return False

def test_short_audio_generation():
    """Test generating a short audio clip"""
    try:
        api_key = os.getenv('ELEVENLABS_API_KEY')
        tts = TextToSpeech(api_key)
        
        test_text = "Hello! This is a test of the YouTube Summarizer audio generation feature."
        
        print(f"\n🎵 Testing audio generation...")
        print(f"Text: {test_text}")
        
        audio_file = tts.generate_speech(test_text, output_filename="test_audio.mp3")
        
        print(f"✅ Audio generated successfully: {audio_file}")
        
        # Check if file exists and has content
        if os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            print(f"✅ Audio file size: {file_size} bytes")
            
            if file_size > 1000:  # Basic sanity check
                print("✅ Audio file appears to have content")
                return True
            else:
                print("❌ Audio file seems too small")
                return False
        else:
            print("❌ Audio file was not created")
            return False
        
    except Exception as e:
        print(f"❌ Error generating audio: {str(e)}")
        return False

def main():
    print("🧪 Testing ElevenLabs Audio Integration")
    print("=" * 50)
    
    tests = [
        ("API Connection", test_elevenlabs_connection),
        ("Voice List", test_voice_list),
        ("Audio Generation", test_short_audio_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Audio generation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your ElevenLabs API key and internet connection.")

if __name__ == "__main__":
    main()