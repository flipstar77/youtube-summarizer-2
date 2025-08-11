import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class TextToSpeech:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable or pass it directly.")
        
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Create audio directory if it doesn't exist
        self.audio_dir = Path("static/audio")
        self.audio_dir.mkdir(exist_ok=True, parents=True)
    
    def get_voices(self):
        """Get available voices from ElevenLabs"""
        try:
            response = requests.get(f"{self.base_url}/voices", headers={"xi-api-key": self.api_key})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Error getting voices: {str(e)}")
    
    def generate_speech(self, text, voice_id="21m00Tcm4TlvDq8ikWAM", output_filename=None):
        """
        Generate speech from text using ElevenLabs API
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID (default is Rachel)
            output_filename: Output file name (if None, auto-generated)
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Limit text length for reasonable audio files
            if len(text) > 5000:
                text = text[:4950] + "... (summary truncated for audio)"
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
            
            # Generate filename if not provided
            if not output_filename:
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                output_filename = f"summary_{text_hash}.mp3"
            
            output_path = self.audio_dir / output_filename
            
            # Save the audio file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            # Return web-compatible path with forward slashes
            return str(output_path).replace('\\', '/')
            
        except Exception as e:
            raise Exception(f"Error generating speech: {str(e)}")
    
    def get_available_voices(self):
        """Get a simplified list of available voices for UI"""
        try:
            voices_data = self.get_voices()
            voices = []
            
            for voice in voices_data.get('voices', []):
                voices.append({
                    'voice_id': voice['voice_id'],
                    'name': voice['name'],
                    'category': voice.get('category', 'Unknown'),
                    'description': voice.get('description', ''),
                    'preview_url': voice.get('preview_url', '')
                })
            
            return voices
        except Exception as e:
            # Return default voices if API call fails
            print(f"[ERROR] get_available_voices failed, using fallback: {str(e)}")
            import traceback
            traceback.print_exc()
            return [
                {
                    'voice_id': '21m00Tcm4TlvDq8ikWAM',
                    'name': 'Rachel',
                    'category': 'Narrative',
                    'description': 'American female voice, calm and clear'
                },
                {
                    'voice_id': 'AZnzlk1XvdvUeBnXmlld',
                    'name': 'Domi',
                    'category': 'Narrative', 
                    'description': 'American female voice, confident'
                },
                {
                    'voice_id': 'EXAVITQu4vr4xnSDxMaL',
                    'name': 'Bella',
                    'category': 'Narrative',
                    'description': 'American female voice, friendly'
                }
            ]