"""
Caption fetcher with YouTube → yt-dlp → Whisper fallback
Implements robust caption extraction with multiple sources
"""
import os
import json
import subprocess
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from .captions_types import CaptionChunk, Source
from .captions_normalize import parse_srt, parse_vtt

logger = logging.getLogger(__name__)

@dataclass 
class FetchResult:
    """Result of caption fetching attempt"""
    chunks: List[CaptionChunk]
    source: Source
    language: str
    success: bool
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

class CaptionFetcher:
    """
    Fetches captions using multiple fallback sources:
    1) YouTube auto-captions (fastest, when available)
    2) yt-dlp extraction (.vtt/.srt if YouTube fails)  
    3) Whisper transcription (expensive/slow fallback)
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
    def fetch_captions(self, video_id: str, language: str = "en") -> FetchResult:
        """
        Main entry point - tries all sources in order
        Returns first successful result or final failure
        """
        logger.info(f"Fetching captions for {video_id} (lang: {language})")
        
        # Strategy 1: YouTube auto-captions
        try:
            result = self._fetch_youtube_captions(video_id, language)
            if result.success and result.chunks:
                logger.info(f"✅ YouTube captions: {len(result.chunks)} chunks")
                return result
        except Exception as e:
            logger.warning(f"YouTube caption fetch failed: {e}")
        
        # Strategy 2: yt-dlp extraction
        try:
            result = self._fetch_ytdlp_captions(video_id, language)
            if result.success and result.chunks:
                logger.info(f"✅ yt-dlp captions: {len(result.chunks)} chunks")
                return result
        except Exception as e:
            logger.warning(f"yt-dlp caption fetch failed: {e}")
        
        # Strategy 3: Whisper fallback (expensive)
        try:
            logger.info("🔄 Falling back to Whisper transcription (this may take time)")
            result = self._fetch_whisper_captions(video_id, language)
            if result.success and result.chunks:
                logger.info(f"✅ Whisper transcription: {len(result.chunks)} chunks")
                return result
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
        
        # All strategies failed
        return FetchResult(
            chunks=[], 
            source="youtube", 
            language=language, 
            success=False,
            error="All caption sources failed"
        )
    
    def _fetch_youtube_captions(self, video_id: str, language: str) -> FetchResult:
        """Fetch auto-generated captions directly from YouTube"""
        try:
            # Use yt-dlp to get caption info without downloading video
            cmd = [
                "yt-dlp",
                "--list-subs",
                "--write-auto-sub", 
                "--sub-lang", language,
                "--no-download",
                f"https://youtube.com/watch?v={video_id}"
            ]
            
            # Get available subtitle info
            info_result = subprocess.run(
                ["yt-dlp", "--dump-json", "--no-download", f"https://youtube.com/watch?v={video_id}"],
                capture_output=True, text=True, timeout=30
            )
            
            if info_result.returncode != 0:
                raise Exception(f"yt-dlp info failed: {info_result.stderr}")
            
            video_info = json.loads(info_result.stdout)
            duration = video_info.get('duration')
            
            # Download auto-generated subtitles
            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = [
                    "yt-dlp",
                    "--write-auto-sub",
                    "--sub-lang", language,
                    "--sub-format", "vtt",
                    "--skip-download",
                    "--output", os.path.join(temp_dir, "%(title)s.%(ext)s"),
                    f"https://youtube.com/watch?v={video_id}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    raise Exception(f"YouTube caption download failed: {result.stderr}")
                
                # Find the downloaded VTT file
                vtt_files = [f for f in os.listdir(temp_dir) if f.endswith(f'.{language}.vtt')]
                if not vtt_files:
                    raise Exception("No VTT file found after download")
                
                vtt_path = os.path.join(temp_dir, vtt_files[0])
                with open(vtt_path, 'r', encoding='utf-8') as f:
                    vtt_content = f.read()
                
                chunks = parse_vtt(vtt_content, language, "youtube")
                
                return FetchResult(
                    chunks=chunks,
                    source="youtube",
                    language=language,
                    success=True,
                    duration_seconds=duration
                )
                
        except Exception as e:
            return FetchResult(
                chunks=[], 
                source="youtube", 
                language=language, 
                success=False,
                error=str(e)
            )
    
    def _fetch_ytdlp_captions(self, video_id: str, language: str) -> FetchResult:
        """Extract captions using yt-dlp (manual/uploaded subs)"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to get both manual and auto subs
                cmd = [
                    "yt-dlp",
                    "--write-sub",
                    "--write-auto-sub", 
                    "--sub-lang", language,
                    "--sub-format", "vtt/srt/best",
                    "--skip-download",
                    "--output", os.path.join(temp_dir, "%(title)s.%(ext)s"),
                    f"https://youtube.com/watch?v={video_id}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                
                if result.returncode != 0:
                    raise Exception(f"yt-dlp subtitle extraction failed: {result.stderr}")
                
                # Look for subtitle files (prefer manual over auto)
                sub_files = []
                for f in os.listdir(temp_dir):
                    if f.endswith(('.vtt', '.srt')) and language in f:
                        sub_files.append(f)
                
                if not sub_files:
                    raise Exception("No subtitle files found")
                
                # Prefer manual subs over auto-generated
                manual_subs = [f for f in sub_files if not 'auto' in f.lower()]
                chosen_file = manual_subs[0] if manual_subs else sub_files[0]
                
                sub_path = os.path.join(temp_dir, chosen_file)
                with open(sub_path, 'r', encoding='utf-8') as f:
                    sub_content = f.read()
                
                # Parse based on file extension
                if chosen_file.endswith('.vtt'):
                    chunks = parse_vtt(sub_content, language, "yt-dlp")
                else:
                    chunks = parse_srt(sub_content, language, "yt-dlp")
                
                return FetchResult(
                    chunks=chunks,
                    source="yt-dlp",
                    language=language,
                    success=True
                )
                
        except Exception as e:
            return FetchResult(
                chunks=[], 
                source="yt-dlp", 
                language=language, 
                success=False,
                error=str(e)
            )
    
    def _fetch_whisper_captions(self, video_id: str, language: str) -> FetchResult:
        """
        Fallback to Whisper transcription
        Downloads audio and transcribes locally (expensive/slow)
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download audio only
                audio_path = os.path.join(temp_dir, f"{video_id}.%(ext)s")
                download_cmd = [
                    "yt-dlp", 
                    "-f", "bestaudio",
                    "--extract-audio",
                    "--audio-format", "mp3",
                    "--output", audio_path,
                    f"https://youtube.com/watch?v={video_id}"
                ]
                
                logger.info("⬇️ Downloading audio for Whisper transcription...")
                download_result = subprocess.run(
                    download_cmd, capture_output=True, text=True, timeout=300
                )
                
                if download_result.returncode != 0:
                    raise Exception(f"Audio download failed: {download_result.stderr}")
                
                # Find downloaded audio file
                audio_files = [f for f in os.listdir(temp_dir) if f.startswith(video_id) and f.endswith('.mp3')]
                if not audio_files:
                    raise Exception("No audio file found after download")
                
                actual_audio_path = os.path.join(temp_dir, audio_files[0])
                
                # Run Whisper transcription
                logger.info("🤖 Running Whisper transcription (this may take several minutes)...")
                whisper_cmd = [
                    "whisper",
                    actual_audio_path,
                    "--language", language,
                    "--output_format", "srt",
                    "--output_dir", temp_dir
                ]
                
                whisper_result = subprocess.run(
                    whisper_cmd, capture_output=True, text=True, timeout=1800  # 30 min max
                )
                
                if whisper_result.returncode != 0:
                    raise Exception(f"Whisper transcription failed: {whisper_result.stderr}")
                
                # Find generated SRT file
                srt_files = [f for f in os.listdir(temp_dir) if f.endswith('.srt')]
                if not srt_files:
                    raise Exception("No SRT file generated by Whisper")
                
                srt_path = os.path.join(temp_dir, srt_files[0])
                with open(srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                
                chunks = parse_srt(srt_content, language, "whisper")
                
                # Add confidence scores if available from Whisper
                # (Whisper doesn't provide per-chunk confidence in SRT, but we could enhance this)
                for chunk in chunks:
                    chunk.confidence = 0.85  # Default confidence for Whisper
                
                return FetchResult(
                    chunks=chunks,
                    source="whisper",
                    language=language,
                    success=True
                )
                
        except Exception as e:
            return FetchResult(
                chunks=[], 
                source="whisper", 
                language=language, 
                success=False,
                error=str(e)
            )
    
    def get_video_duration(self, video_id: str) -> Optional[float]:
        """Get video duration in seconds using yt-dlp"""
        try:
            cmd = ["yt-dlp", "--dump-json", "--no-download", f"https://youtube.com/watch?v={video_id}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                return info.get('duration')
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
        
        return None


# Convenience function for simple usage
def fetch_captions(video_id: str, language: str = "en") -> List[CaptionChunk]:
    """
    Simple wrapper - returns caption chunks or empty list
    Usage: chunks = fetch_captions("dQw4w9WgXcQ")
    """
    fetcher = CaptionFetcher()
    result = fetcher.fetch_captions(video_id, language)
    return result.chunks if result.success else []