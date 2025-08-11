"""
Enhanced SRT Parser and Timing Module

This module provides comprehensive SRT subtitle file parsing with advanced timing
functionality for video highlight extraction. It handles various SRT formats,
time calculations, and segment extraction for integration with video processing.
"""

import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SRTSegment:
    """Represents a single SRT subtitle segment"""
    index: int
    start_time: str
    end_time: str
    text: str
    start_seconds: float
    end_seconds: float
    duration: float
    
    def __post_init__(self):
        """Calculate additional timing properties"""
        if not self.start_seconds:
            self.start_seconds = self._time_to_seconds(self.start_time)
        if not self.end_seconds:
            self.end_seconds = self._time_to_seconds(self.end_time)
        if not self.duration:
            self.duration = self.end_seconds - self.start_seconds
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert SRT time format to seconds"""
        # Handle format: 00:00:00,000 or 00:00:00.000
        time_str = time_str.replace(',', '.')
        parts = time_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0.0
    
    def overlaps_with(self, other: 'SRTSegment', buffer_seconds: float = 0.5) -> bool:
        """Check if this segment overlaps with another segment"""
        return (self.start_seconds - buffer_seconds <= other.end_seconds and 
                self.end_seconds + buffer_seconds >= other.start_seconds)
    
    def merge_with(self, other: 'SRTSegment') -> 'SRTSegment':
        """Merge this segment with another to create a combined segment"""
        start_seconds = min(self.start_seconds, other.start_seconds)
        end_seconds = max(self.end_seconds, other.end_seconds)
        
        combined_text = f"{self.text} {other.text}".strip()
        
        return SRTSegment(
            index=self.index,
            start_time=self._seconds_to_time(start_seconds),
            end_time=self._seconds_to_time(end_seconds),
            text=combined_text,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration=end_seconds - start_seconds
        )
    
    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        milliseconds = int((secs % 1) * 1000)
        secs = int(secs)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


class SRTParser:
    """Advanced SRT parser with timing and segment management"""
    
    def __init__(self):
        self.segments: List[SRTSegment] = []
        self.total_duration: float = 0.0
        self.segment_count: int = 0
    
    def parse_file(self, srt_path: str) -> List[SRTSegment]:
        """Parse SRT file and return list of segments"""
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self.parse_content(content)
    
    def parse_content(self, srt_content: str) -> List[SRTSegment]:
        """Parse SRT content string and return segments"""
        self.segments = []
        
        # Split by double newlines to get individual subtitle blocks
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        
        for block in blocks:
            if block.strip():
                segment = self._parse_segment_block(block.strip())
                if segment:
                    self.segments.append(segment)
        
        self._calculate_stats()
        return self.segments
    
    def _parse_segment_block(self, block: str) -> Optional[SRTSegment]:
        """Parse individual SRT segment block"""
        lines = block.split('\n')
        
        if len(lines) < 3:
            return None
        
        try:
            # First line: segment index
            index = int(lines[0].strip())
            
            # Second line: timing (e.g., "00:00:01,234 --> 00:00:03,456")
            timing_match = re.match(r'(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})', lines[1])
            if not timing_match:
                return None
            
            start_time = timing_match.group(1).replace('.', ',')
            end_time = timing_match.group(2).replace('.', ',')
            
            # Remaining lines: subtitle text
            text = ' '.join(lines[2:]).strip()
            # Clean up common SRT formatting
            text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
            text = re.sub(r'\{[^}]+\}', '', text)  # Remove styling tags
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            return SRTSegment(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
                start_seconds=0,  # Will be calculated in __post_init__
                end_seconds=0,
                duration=0
            )
        
        except (ValueError, IndexError):
            return None
    
    def _calculate_stats(self):
        """Calculate statistics about the parsed segments"""
        if self.segments:
            self.segment_count = len(self.segments)
            self.total_duration = max(seg.end_seconds for seg in self.segments)
        else:
            self.segment_count = 0
            self.total_duration = 0.0
    
    def get_segments_in_range(self, start_seconds: float, end_seconds: float) -> List[SRTSegment]:
        """Get all segments that fall within a time range"""
        return [
            seg for seg in self.segments
            if (seg.start_seconds < end_seconds and seg.end_seconds > start_seconds)
        ]
    
    def get_text_in_range(self, start_seconds: float, end_seconds: float) -> str:
        """Get combined text for all segments in a time range"""
        segments = self.get_segments_in_range(start_seconds, end_seconds)
        return ' '.join(seg.text for seg in segments)
    
    def create_windowed_segments(self, window_size: int = 30, overlap: int = 5) -> List[Dict]:
        """
        Create windowed segments for highlight analysis
        
        Args:
            window_size: Size of each window in seconds
            overlap: Overlap between windows in seconds
            
        Returns:
            List of window dictionaries with timing and text
        """
        windows = []
        current_start = 0.0
        window_id = 1
        
        while current_start < self.total_duration:
            current_end = min(current_start + window_size, self.total_duration)
            
            # Get text for this window
            window_text = self.get_text_in_range(current_start, current_end)
            
            if window_text.strip():  # Only include windows with text
                windows.append({
                    'id': window_id,
                    'start_time': current_start,
                    'end_time': current_end,
                    'duration': current_end - current_start,
                    'text': window_text.strip(),
                    'start_time_formatted': SRTSegment._seconds_to_time(current_start),
                    'end_time_formatted': SRTSegment._seconds_to_time(current_end),
                    'segments': self.get_segments_in_range(current_start, current_end)
                })
                window_id += 1
            
            # Move to next window with overlap
            current_start += window_size - overlap
        
        return windows
    
    def merge_consecutive_segments(self, segments: List[SRTSegment], 
                                 max_gap_seconds: float = 2.0) -> List[SRTSegment]:
        """
        Merge consecutive segments that are close together
        
        Args:
            segments: List of segments to merge
            max_gap_seconds: Maximum gap between segments to allow merging
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_seconds)
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last_merged = merged[-1]
            
            # Check if segments are close enough to merge
            gap = current.start_seconds - last_merged.end_seconds
            
            if gap <= max_gap_seconds:
                # Merge with previous segment
                merged[-1] = last_merged.merge_with(current)
            else:
                # Add as new segment
                merged.append(current)
        
        return merged
    
    def extract_highlight_context(self, highlight_start: float, highlight_end: float,
                                context_seconds: float = 5.0) -> Dict:
        """
        Extract context around a highlight segment
        
        Args:
            highlight_start: Start time of highlight in seconds
            highlight_end: End time of highlight in seconds
            context_seconds: Seconds of context to include before/after
            
        Returns:
            Dictionary with extended timing and text information
        """
        # Calculate extended range
        extended_start = max(0, highlight_start - context_seconds)
        extended_end = min(self.total_duration, highlight_end + context_seconds)
        
        # Get segments and text
        core_segments = self.get_segments_in_range(highlight_start, highlight_end)
        extended_segments = self.get_segments_in_range(extended_start, extended_end)
        
        return {
            'core_start': highlight_start,
            'core_end': highlight_end,
            'extended_start': extended_start,
            'extended_end': extended_end,
            'core_duration': highlight_end - highlight_start,
            'extended_duration': extended_end - extended_start,
            'core_text': ' '.join(seg.text for seg in core_segments),
            'extended_text': ' '.join(seg.text for seg in extended_segments),
            'core_segments': core_segments,
            'extended_segments': extended_segments,
            'core_start_formatted': SRTSegment._seconds_to_time(highlight_start),
            'core_end_formatted': SRTSegment._seconds_to_time(highlight_end),
            'extended_start_formatted': SRTSegment._seconds_to_time(extended_start),
            'extended_end_formatted': SRTSegment._seconds_to_time(extended_end),
        }
    
    def generate_srt_content(self, segments: List[SRTSegment]) -> str:
        """Generate SRT content from segments"""
        srt_content = []
        
        for i, segment in enumerate(segments, 1):
            srt_content.extend([
                str(i),
                f"{segment.start_time} --> {segment.end_time}",
                segment.text,
                ""  # Empty line between segments
            ])
        
        return '\n'.join(srt_content)
    
    def save_segments_as_srt(self, segments: List[SRTSegment], output_path: str):
        """Save segments as SRT file"""
        srt_content = self.generate_srt_content(segments)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
    
    def get_stats(self) -> Dict:
        """Get statistics about the parsed SRT"""
        if not self.segments:
            return {
                'segment_count': 0,
                'total_duration': 0,
                'average_segment_duration': 0,
                'total_text_length': 0
            }
        
        durations = [seg.duration for seg in self.segments]
        text_lengths = [len(seg.text) for seg in self.segments]
        
        return {
            'segment_count': self.segment_count,
            'total_duration': round(self.total_duration, 2),
            'total_duration_formatted': SRTSegment._seconds_to_time(self.total_duration),
            'average_segment_duration': round(sum(durations) / len(durations), 2),
            'min_segment_duration': round(min(durations), 2),
            'max_segment_duration': round(max(durations), 2),
            'total_text_length': sum(text_lengths),
            'average_text_length': round(sum(text_lengths) / len(text_lengths), 1),
            'segments_per_minute': round(self.segment_count / (self.total_duration / 60), 1)
        }


# Utility functions for working with YouTube transcripts
def convert_youtube_transcript_to_srt(transcript_segments, output_path: Optional[str] = None) -> str:
    """
    Convert YouTube transcript segments to SRT format
    
    Args:
        transcript_segments: List of transcript segments from YouTube API
        output_path: Optional path to save SRT file
        
    Returns:
        SRT content as string
    """
    srt_segments = []
    
    for i, segment in enumerate(transcript_segments, 1):
        start_time = getattr(segment, 'start', 0)
        duration = getattr(segment, 'duration', 0)
        text = getattr(segment, 'text', '')
        
        end_time = start_time + duration
        
        srt_segment = SRTSegment(
            index=i,
            start_time=SRTSegment._seconds_to_time(start_time),
            end_time=SRTSegment._seconds_to_time(end_time),
            text=text,
            start_seconds=start_time,
            end_seconds=end_time,
            duration=duration
        )
        srt_segments.append(srt_segment)
    
    parser = SRTParser()
    srt_content = parser.generate_srt_content(srt_segments)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
    
    return srt_content


def extract_key_phrases(text: str, min_length: int = 3) -> List[str]:
    """Extract key phrases from subtitle text"""
    # Remove common filler words and split into phrases
    filler_words = {'um', 'uh', 'like', 'you know', 'so', 'well', 'okay', 'right'}
    
    # Split by punctuation and filter
    phrases = re.split(r'[.!?]+', text.lower())
    key_phrases = []
    
    for phrase in phrases:
        words = phrase.strip().split()
        if len(words) >= min_length:
            # Remove filler words
            clean_words = [w for w in words if w not in filler_words]
            if len(clean_words) >= min_length:
                key_phrases.append(' '.join(clean_words))
    
    return key_phrases


if __name__ == "__main__":
    # Example usage
    parser = SRTParser()
    
    # Test with a sample SRT content
    sample_srt = """1
00:00:01,000 --> 00:00:03,500
Hello and welcome to this tutorial

2
00:00:03,500 --> 00:00:06,000
Today we'll learn about video processing

3
00:00:06,000 --> 00:00:09,000
This is an exciting topic that many find useful"""
    
    segments = parser.parse_content(sample_srt)
    print(f"Parsed {len(segments)} segments")
    
    # Create windowed segments
    windows = parser.create_windowed_segments(window_size=10, overlap=2)
    print(f"Created {len(windows)} analysis windows")
    
    # Print stats
    stats = parser.get_stats()
    print("\nSRT Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")