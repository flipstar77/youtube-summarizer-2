"""
Caption data types and models
Shared types for the caption system
"""
from dataclasses import dataclass
from typing import Literal, Optional, List

Source = Literal["youtube", "yt-dlp", "whisper"]

@dataclass
class CaptionChunk:
    start_ms: int
    end_ms: int
    text: str
    lang: str
    source: Source
    confidence: Optional[float] = None  # nur whisper sinnvoll
    
    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds"""
        return self.end_ms - self.start_ms
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds"""
        return self.duration_ms / 1000.0
    
    def overlaps_with(self, other: 'CaptionChunk') -> bool:
        """Check if this chunk overlaps with another"""
        return self.start_ms < other.end_ms and self.end_ms > other.start_ms