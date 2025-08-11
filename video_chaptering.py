#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Video Chaptering System
Creates smart chapter markers for YouTube videos based on content analysis
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv

load_dotenv()

class VideoChapteringSystem:
    """Creates intelligent video chapters based on transcript analysis"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def create_video_chapters(self, video_id: str, transcript_text: str, 
                            chapter_count: int = 8, min_duration: int = 30) -> Dict[str, Any]:
        """
        Creates intelligent chapter markers for a video
        
        Args:
            video_id: YouTube Video ID
            transcript_text: Full video transcript
            chapter_count: Desired number of chapters
            min_duration: Minimum chapter duration in seconds
        """
        try:
            print(f"[INFO] Creating {chapter_count} chapters for video {video_id}")
            
            # 1. Split transcript into time-based segments
            segments = self._split_transcript_by_time(transcript_text, min_duration)
            
            if not segments:
                return {"error": "Failed to parse transcript segments"}
            
            print(f"[INFO] Split transcript into {len(segments)} time segments")
            
            # 2. AI-based chapter identification
            chapters = self._identify_chapters_with_ai(segments, chapter_count)
            
            # 3. Create YouTube-compatible timestamps
            formatted_chapters = self._format_chapters_for_youtube(chapters)
            
            return {
                "status": "success",
                "video_id": video_id,
                "chapters": formatted_chapters,
                "total_chapters": len(formatted_chapters),
                "youtube_description": self._create_youtube_description(formatted_chapters)
            }
            
        except Exception as e:
            return {"error": f"Chapter creation failed: {str(e)}"}
    
    def _split_transcript_by_time(self, transcript_text: str, min_duration: int) -> List[Dict]:
        """Split transcript into time-based segments"""
        try:
            # Simple approach: split by sentences and estimate timing
            sentences = []
            for sentence in transcript_text.replace('\n', ' ').split('.'):
                sentence = sentence.strip()
                if sentence and len(sentence) > 20:  # Only meaningful sentences
                    sentences.append(sentence)
            
            if not sentences:
                return []
            
            segments = []
            current_time = 0
            words_per_minute = 150  # Average speaking speed
            
            for i, sentence in enumerate(sentences):
                words = len(sentence.split())
                duration = max(int((words / words_per_minute) * 60), 3)  # Min 3 seconds per sentence
                
                segments.append({
                    "index": i,
                    "start_seconds": current_time,
                    "end_seconds": current_time + duration,
                    "text": sentence.strip(),
                    "word_count": words
                })
                
                current_time += duration + 1  # 1 second pause between sentences
            
            return segments
            
        except Exception as e:
            print(f"[ERROR] Transcript splitting failed: {str(e)}")
            return []
    
    def _identify_chapters_with_ai(self, segments: List[Dict], chapter_count: int) -> List[Dict]:
        """Use AI to identify natural chapter breaks"""
        try:
            # Group segments into larger chunks for chapter analysis
            chunk_size = max(len(segments) // (chapter_count * 2), 3)  # More chunks than needed chapters
            chunks = []
            
            for i in range(0, len(segments), chunk_size):
                chunk_segments = segments[i:i + chunk_size]
                combined_text = " ".join([seg["text"] for seg in chunk_segments])
                
                chunks.append({
                    "start_seconds": chunk_segments[0]["start_seconds"],
                    "end_seconds": chunk_segments[-1]["end_seconds"],
                    "text": combined_text[:500],  # Limit text for API
                    "segment_count": len(chunk_segments)
                })
            
            print(f"[INFO] Created {len(chunks)} chunks for chapter analysis")
            
            # AI analysis for chapter identification
            prompt = f"""Analyze this video transcript and identify the {chapter_count} most logical chapter breaks.
Look for natural transitions like:
- Topic changes
- New sections or points
- Introductions to new concepts
- "Next", "Now", "Let's move on" phrases
- Shifts in discussion focus

Text chunks:
{json.dumps([{"time": f"{c['start_seconds']}s", "text": c['text'][:200]} for c in chunks[:15]], indent=2)}

Return ONLY a JSON array with {chapter_count} chapters in this format:
[
  {{"timestamp_seconds": 0, "title": "Introduction", "description": "Video overview and setup"}},
  {{"timestamp_seconds": 120, "title": "First Main Topic", "description": "Detailed explanation of..."}}
]

Make titles concise (2-6 words) and descriptions informative (5-15 words)."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result.startswith("```json"):
                result = result.split("```json")[1].split("```")[0]
            elif result.startswith("```"):
                result = result.split("```")[1].split("```")[0]
            
            chapters = json.loads(result)
            
            # Validate and sort chapters
            chapters = [ch for ch in chapters if "timestamp_seconds" in ch and "title" in ch]
            chapters.sort(key=lambda x: x["timestamp_seconds"])
            
            print(f"[INFO] AI identified {len(chapters)} chapters")
            return chapters
            
        except Exception as e:
            print(f"[ERROR] AI chapter analysis failed: {str(e)}")
            # Fallback: Create evenly spaced chapters
            return self._create_fallback_chapters(segments, chapter_count)
    
    def _create_fallback_chapters(self, segments: List[Dict], chapter_count: int) -> List[Dict]:
        """Create evenly spaced chapters as fallback"""
        try:
            if not segments:
                return []
            
            total_duration = segments[-1]["end_seconds"]
            chapter_interval = total_duration // chapter_count
            
            chapters = []
            for i in range(chapter_count):
                timestamp = i * chapter_interval
                chapters.append({
                    "timestamp_seconds": timestamp,
                    "title": f"Chapter {i + 1}",
                    "description": f"Content from {self._seconds_to_timestamp(timestamp)}"
                })
            
            return chapters
            
        except Exception as e:
            print(f"[ERROR] Fallback chapter creation failed: {str(e)}")
            return []
    
    def _format_chapters_for_youtube(self, chapters: List[Dict]) -> List[Dict]:
        """Format chapters with YouTube-compatible timestamps"""
        formatted = []
        
        for chapter in chapters:
            timestamp_seconds = chapter.get("timestamp_seconds", 0)
            
            formatted.append({
                "timestamp": self._seconds_to_timestamp(timestamp_seconds),
                "timestamp_seconds": timestamp_seconds,
                "title": chapter.get("title", "Chapter"),
                "description": chapter.get("description", ""),
                "youtube_link": self._create_youtube_timestamp_link(timestamp_seconds)
            })
        
        return formatted
    
    def _seconds_to_timestamp(self, seconds: int) -> str:
        """Convert seconds to MM:SS or HH:MM:SS format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def _create_youtube_timestamp_link(self, seconds: int) -> str:
        """Create YouTube URL with timestamp parameter"""
        return f"&t={seconds}s"
    
    def _create_youtube_description(self, chapters: List[Dict]) -> str:
        """Create YouTube description with chapter timestamps"""
        description_lines = ["📚 **Video Chapters:**", ""]
        
        for chapter in chapters:
            line = f"⏰ {chapter['timestamp']} - {chapter['title']}"
            if chapter.get('description'):
                line += f" - {chapter['description']}"
            description_lines.append(line)
        
        description_lines.extend([
            "",
            "🤖 Generated with AI-powered video analysis"
        ])
        
        return "\n".join(description_lines)

def main():
    """Test the video chaptering system"""
    print("🎬 Video Chaptering System Test")
    print("=" * 50)
    
    # Test with sample transcript
    sample_transcript = """
    Hello everyone, welcome to this tutorial. Today we'll learn about Python programming.
    First, let's start with the basics of Python syntax. Variables are fundamental.
    Now let's move on to functions. Functions help organize your code better.
    Next, we'll discuss classes and object-oriented programming concepts.
    Finally, we'll look at some practical examples and real-world applications.
    Thank you for watching, and don't forget to subscribe for more content.
    """
    
    chaptering = VideoChapteringSystem()
    result = chaptering.create_video_chapters(
        video_id="test_video",
        transcript_text=sample_transcript,
        chapter_count=5
    )
    
    print(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()