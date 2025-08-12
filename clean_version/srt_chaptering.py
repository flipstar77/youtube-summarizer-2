#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT Chaptering System
Processes uploaded SRT files and creates intelligent chapter markers for content creators
"""

import os
import re
import json
import openai
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

class SRTChapteringSystem:
    """Creates chapters from uploaded SRT files using AI analysis"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.output_dir = "D:/mcp/srt_uploads"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_srt_file(self, srt_content: str) -> List[Dict[str, Any]]:
        """Parse SRT file content into structured segments"""
        segments = []
        
        # Clean and normalize the content
        srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = srt_content.strip().split('\n\n')
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    # Parse subtitle number
                    subtitle_num = int(lines[0].strip())
                    
                    # Parse timestamp
                    timestamp_line = lines[1].strip()
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                    
                    if time_match:
                        start_time = time_match.group(1)
                        end_time = time_match.group(2)
                        
                        # Parse text (can be multiple lines)
                        text = '\n'.join(lines[2:]).strip()
                        
                        # Convert timestamps to seconds for easier processing
                        start_seconds = self._timestamp_to_seconds(start_time)
                        end_seconds = self._timestamp_to_seconds(end_time)
                        
                        segments.append({
                            'number': subtitle_num,
                            'start_time': start_time,
                            'end_time': end_time,
                            'start_seconds': start_seconds,
                            'end_seconds': end_seconds,
                            'text': text,
                            'duration': end_seconds - start_seconds
                        })
                        
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Skipping malformed SRT block: {e}")
                    continue
        
        return segments
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""
        try:
            time_part, ms_part = timestamp.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            return h * 3600 + m * 60 + s + ms / 1000.0
        except Exception:
            return 0.0
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds back to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    
    def create_chapters_from_srt(self, srt_content: str, 
                                video_title: str = "Uploaded Video",
                                chapter_count: int = 8,
                                min_chapter_duration: int = 60) -> Dict[str, Any]:
        """Create intelligent chapters from SRT content"""
        try:
            print(f"[INFO] Processing SRT file for chaptering...")
            
            # Parse SRT segments
            segments = self.parse_srt_file(srt_content)
            if not segments:
                return {"error": "No valid SRT segments found"}
            
            print(f"[INFO] Parsed {len(segments)} SRT segments")
            
            # Combine segments into larger text blocks for analysis
            full_transcript = self._combine_segments_for_analysis(segments)
            
            # Use AI to identify chapter boundaries
            chapters = self._identify_chapter_points(
                full_transcript, 
                segments,
                chapter_count, 
                min_chapter_duration
            )
            
            # Generate chapter descriptions
            enhanced_chapters = self._enhance_chapters_with_descriptions(chapters, segments)
            
            # Create output formats
            result = {
                "status": "success",
                "video_title": video_title,
                "total_duration": segments[-1]['end_seconds'] if segments else 0,
                "total_segments": len(segments),
                "chapters": enhanced_chapters,
                "chapter_count": len(enhanced_chapters),
                "formats": {
                    "youtube_description": self._create_youtube_description(enhanced_chapters),
                    "video_chapters": self._create_video_chapters_json(enhanced_chapters),
                    "premiere_markers": self._create_premiere_markers(enhanced_chapters),
                    "srt_chapters": self._create_srt_chapters(enhanced_chapters)
                }
            }
            
            return result
            
        except Exception as e:
            print(f"[ERROR] SRT chaptering failed: {str(e)}")
            return {"error": f"Chaptering failed: {str(e)}"}
    
    def _combine_segments_for_analysis(self, segments: List[Dict[str, Any]]) -> str:
        """Combine SRT segments with timestamps for AI analysis"""
        combined_text = ""
        
        for segment in segments:
            timestamp = self._seconds_to_readable_time(segment['start_seconds'])
            combined_text += f"[{timestamp}] {segment['text']}\n"
        
        return combined_text
    
    def _seconds_to_readable_time(self, seconds: float) -> str:
        """Convert seconds to readable time format (MM:SS or HH:MM:SS)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _identify_chapter_points(self, transcript: str, segments: List[Dict[str, Any]], 
                               chapter_count: int, min_duration: int) -> List[Dict[str, Any]]:
        """Use AI to identify optimal chapter break points"""
        try:
            total_duration = segments[-1]['end_seconds'] if segments else 0
            
            prompt = f"""
            Analyze this video transcript with timestamps and identify {chapter_count} optimal chapter break points.
            
            Video duration: {self._seconds_to_readable_time(total_duration)}
            Minimum chapter duration: {min_duration} seconds
            
            Requirements:
            1. Chapters should represent distinct topics or segments
            2. Each chapter should be at least {min_duration} seconds long
            3. Look for natural topic transitions, speaker changes, or content shifts
            4. Provide meaningful chapter titles (3-8 words)
            5. Return ONLY a JSON array in this exact format:
            
            [
                {{
                    "start_time": "00:00",
                    "title": "Introduction and Overview"
                }},
                {{
                    "start_time": "05:30", 
                    "title": "Main Topic Discussion"
                }}
            ]
            
            Transcript:
            {transcript[:8000]}  
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                chapters_data = json.loads(json_match.group())
                
                # Validate and process chapters
                processed_chapters = []
                for i, chapter in enumerate(chapters_data):
                    try:
                        # Convert time format to seconds
                        time_str = chapter.get('start_time', '00:00')
                        start_seconds = self._time_string_to_seconds(time_str)
                        
                        processed_chapters.append({
                            "chapter_number": i + 1,
                            "start_time": time_str,
                            "start_seconds": start_seconds,
                            "title": chapter.get('title', f'Chapter {i + 1}'),
                            "timestamp_youtube": self._seconds_to_youtube_timestamp(start_seconds)
                        })
                    except Exception as e:
                        print(f"[WARNING] Error processing chapter {i}: {e}")
                        continue
                
                return processed_chapters
            
            # Fallback: create evenly spaced chapters
            return self._create_fallback_chapters(total_duration, chapter_count)
            
        except Exception as e:
            print(f"[WARNING] AI chapter identification failed: {e}")
            return self._create_fallback_chapters(segments[-1]['end_seconds'] if segments else 300, chapter_count)
    
    def _time_string_to_seconds(self, time_str: str) -> float:
        """Convert time string (MM:SS or HH:MM:SS) to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return 0
        except Exception:
            return 0
    
    def _seconds_to_youtube_timestamp(self, seconds: float) -> str:
        """Convert seconds to YouTube timestamp format (for descriptions)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _create_fallback_chapters(self, total_duration: float, chapter_count: int) -> List[Dict[str, Any]]:
        """Create evenly spaced fallback chapters if AI fails"""
        chapters = []
        chapter_duration = total_duration / chapter_count
        
        for i in range(chapter_count):
            start_seconds = i * chapter_duration
            chapters.append({
                "chapter_number": i + 1,
                "start_time": self._seconds_to_readable_time(start_seconds),
                "start_seconds": start_seconds,
                "title": f"Chapter {i + 1}",
                "timestamp_youtube": self._seconds_to_youtube_timestamp(start_seconds)
            })
        
        return chapters
    
    def _enhance_chapters_with_descriptions(self, chapters: List[Dict[str, Any]], 
                                          segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add descriptions to chapters based on content"""
        enhanced_chapters = []
        
        for i, chapter in enumerate(chapters):
            # Find segments that belong to this chapter
            chapter_start = chapter['start_seconds']
            chapter_end = chapters[i + 1]['start_seconds'] if i + 1 < len(chapters) else segments[-1]['end_seconds']
            
            # Extract text for this chapter
            chapter_segments = [s for s in segments 
                              if chapter_start <= s['start_seconds'] < chapter_end]
            
            chapter_text = ' '.join([s['text'] for s in chapter_segments])
            
            # Create enhanced chapter info
            enhanced_chapter = {
                **chapter,
                "end_seconds": chapter_end,
                "duration": chapter_end - chapter_start,
                "description": chapter_text[:200] + "..." if len(chapter_text) > 200 else chapter_text,
                "segment_count": len(chapter_segments)
            }
            
            enhanced_chapters.append(enhanced_chapter)
        
        return enhanced_chapters
    
    def _create_youtube_description(self, chapters: List[Dict[str, Any]]) -> str:
        """Generate YouTube description with chapters"""
        description = "📚 CHAPTERS:\n\n"
        
        for chapter in chapters:
            timestamp = chapter['timestamp_youtube']
            title = chapter['title']
            description += f"{timestamp} - {title}\n"
        
        description += "\n🤖 Generated with AI Chapter Creator"
        return description
    
    def _create_video_chapters_json(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create JSON format for video editing software"""
        return [
            {
                "name": chapter['title'],
                "startTime": chapter['start_seconds'],
                "endTime": chapter.get('end_seconds', 0)
            }
            for chapter in chapters
        ]
    
    def _create_premiere_markers(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create Adobe Premiere Pro compatible markers"""
        return [
            {
                "name": chapter['title'],
                "time": chapter['start_seconds'],
                "type": "Chapter",
                "comment": chapter.get('description', '')
            }
            for chapter in chapters
        ]
    
    def _create_srt_chapters(self, chapters: List[Dict[str, Any]]) -> str:
        """Create an SRT file with chapter markers"""
        srt_content = ""
        
        for i, chapter in enumerate(chapters):
            start_time = self._seconds_to_timestamp(chapter['start_seconds'])
            end_time = self._seconds_to_timestamp(chapter['start_seconds'] + 5)  # 5 second duration
            
            srt_content += f"{i + 1}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"📚 {chapter['title']}\n\n"
        
        return srt_content
    
    def save_chaptering_result(self, result: Dict[str, Any], filename: str) -> str:
        """Save chaptering result to file"""
        try:
            output_path = os.path.join(self.output_dir, f"chapters_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Chaptering result saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Failed to save chaptering result: {e}")
            return ""