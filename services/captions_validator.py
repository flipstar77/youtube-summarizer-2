"""
Caption validator for quality checking and filtering
Ensures caption quality meets readability standards
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from .captions_types import CaptionChunk

@dataclass
class ValidationResult:
    """Result of caption validation"""
    is_valid: bool
    score: float  # 0.0-1.0 quality score
    issues: List[str]
    suggestions: List[str]
    stats: Dict[str, Any]

class CaptionValidator:
    """
    Validates caption quality based on:
    - Timing consistency (no overlaps, gaps)
    - Reading speed (chars per second)
    - Text quality (length, repetition, gibberish)  
    - Coverage (missing segments)
    """
    
    def __init__(self):
        # Quality thresholds
        self.min_cps = 8          # too slow (boring)
        self.max_cps = 25         # too fast (unreadable)
        self.optimal_cps = 18     # target reading speed
        
        self.min_duration_ms = 500    # too short
        self.max_duration_ms = 6000   # too long
        self.max_gap_ms = 2000        # silence gaps
        
        self.min_text_length = 3      # too short
        self.max_text_length = 120    # too long per chunk
        
        # Common auto-caption artifacts
        self.noise_patterns = [
            r'\[.*?\]',           # [Music], [Applause]
            r'\(.*?\)',           # (inaudible)
            r'>>.*?<<',           # speaker tags
            r'^\s*-\s*$',         # lone dashes
            r'^[♪♫♬].*?[♪♫♬]$',   # music notes
        ]
        
        self.repetition_threshold = 0.7  # max allowed repetition ratio
    
    def validate(self, chunks: List[CaptionChunk], video_duration_seconds: Optional[float] = None) -> ValidationResult:
        """
        Comprehensive validation of caption chunks
        Returns validation result with score and recommendations
        """
        if not chunks:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["No captions provided"],
                suggestions=["Add captions using fallback methods"],
                stats={}
            )
        
        issues = []
        suggestions = []
        stats = self._calculate_stats(chunks, video_duration_seconds)
        
        # Check timing issues
        timing_issues = self._validate_timing(chunks)
        issues.extend(timing_issues)
        
        # Check reading speed
        speed_issues = self._validate_reading_speed(chunks)
        issues.extend(speed_issues)
        
        # Check text quality
        text_issues = self._validate_text_quality(chunks)
        issues.extend(text_issues)
        
        # Check coverage
        if video_duration_seconds:
            coverage_issues = self._validate_coverage(chunks, video_duration_seconds)
            issues.extend(coverage_issues)
        
        # Generate suggestions based on issues
        suggestions = self._generate_suggestions(issues, stats)
        
        # Calculate overall quality score
        score = self._calculate_quality_score(chunks, issues, stats)
        
        return ValidationResult(
            is_valid=len(issues) == 0 or score > 0.6,  # acceptable threshold
            score=score,
            issues=issues,
            suggestions=suggestions,
            stats=stats
        )
    
    def _calculate_stats(self, chunks: List[CaptionChunk], video_duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not chunks:
            return {}
        
        # Basic stats
        total_text = " ".join(chunk.text for chunk in chunks)
        total_chars = sum(len(chunk.text) for chunk in chunks)
        total_duration_ms = sum(chunk.duration_ms for chunk in chunks)
        
        # Timing stats
        start_time = chunks[0].start_ms
        end_time = chunks[-1].end_ms
        span_duration_ms = end_time - start_time
        
        # Reading speed stats
        cps_values = []
        for chunk in chunks:
            if chunk.duration_ms > 0:
                cps = len(chunk.text) / (chunk.duration_ms / 1000.0)
                cps_values.append(cps)
        
        avg_cps = sum(cps_values) / len(cps_values) if cps_values else 0
        
        # Coverage calculation
        coverage_ratio = None
        if video_duration_seconds:
            coverage_ratio = span_duration_ms / (video_duration_seconds * 1000)
        
        # Text quality
        words = total_text.split()
        unique_words = set(words)
        word_diversity = len(unique_words) / len(words) if words else 0
        
        return {
            "chunk_count": len(chunks),
            "total_chars": total_chars,
            "total_words": len(words),
            "unique_words": len(unique_words),
            "word_diversity": word_diversity,
            "total_duration_seconds": total_duration_ms / 1000.0,
            "span_duration_seconds": span_duration_ms / 1000.0,
            "coverage_ratio": coverage_ratio,
            "avg_cps": avg_cps,
            "min_cps": min(cps_values) if cps_values else 0,
            "max_cps": max(cps_values) if cps_values else 0,
            "avg_chunk_duration_ms": total_duration_ms / len(chunks),
            "gaps_detected": self._count_gaps(chunks),
            "overlaps_detected": self._count_overlaps(chunks),
        }
    
    def _validate_timing(self, chunks: List[CaptionChunk]) -> List[str]:
        """Check for timing inconsistencies"""
        issues = []
        
        # Check for overlaps
        overlaps = 0
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]
            
            if current.end_ms > next_chunk.start_ms:
                overlaps += 1
        
        if overlaps > len(chunks) * 0.1:  # >10% overlaps
            issues.append(f"Too many overlapping segments ({overlaps})")
        
        # Check for large gaps
        large_gaps = 0
        for i in range(len(chunks) - 1):
            gap = chunks[i + 1].start_ms - chunks[i].end_ms
            if gap > self.max_gap_ms:
                large_gaps += 1
        
        if large_gaps > len(chunks) * 0.2:  # >20% large gaps
            issues.append(f"Too many large gaps in timeline ({large_gaps})")
        
        # Check chunk durations
        too_short = sum(1 for c in chunks if c.duration_ms < self.min_duration_ms)
        too_long = sum(1 for c in chunks if c.duration_ms > self.max_duration_ms)
        
        if too_short > len(chunks) * 0.3:
            issues.append(f"Many chunks too short ({too_short})")
        if too_long > len(chunks) * 0.1:
            issues.append(f"Some chunks too long ({too_long})")
        
        return issues
    
    def _validate_reading_speed(self, chunks: List[CaptionChunk]) -> List[str]:
        """Check reading speed distribution"""
        issues = []
        
        cps_values = []
        for chunk in chunks:
            if chunk.duration_ms > 0:
                cps = len(chunk.text) / (chunk.duration_ms / 1000.0)
                cps_values.append(cps)
        
        if not cps_values:
            return ["Cannot calculate reading speed"]
        
        avg_cps = sum(cps_values) / len(cps_values)
        too_fast = sum(1 for cps in cps_values if cps > self.max_cps)
        too_slow = sum(1 for cps in cps_values if cps < self.min_cps)
        
        if avg_cps > self.max_cps:
            issues.append(f"Average reading speed too fast ({avg_cps:.1f} cps)")
        elif avg_cps < self.min_cps:
            issues.append(f"Average reading speed too slow ({avg_cps:.1f} cps)")
        
        if too_fast > len(cps_values) * 0.2:
            issues.append(f"Many segments too fast to read ({too_fast})")
        if too_slow > len(cps_values) * 0.3:
            issues.append(f"Many segments unnecessarily slow ({too_slow})")
        
        return issues
    
    def _validate_text_quality(self, chunks: List[CaptionChunk]) -> List[str]:
        """Check text content quality"""
        issues = []
        
        # Check for noise patterns
        noisy_chunks = 0
        for chunk in chunks:
            for pattern in self.noise_patterns:
                if re.search(pattern, chunk.text, re.IGNORECASE):
                    noisy_chunks += 1
                    break
        
        if noisy_chunks > len(chunks) * 0.1:
            issues.append(f"Contains noise/artifacts ({noisy_chunks} chunks)")
        
        # Check text lengths
        too_short_text = sum(1 for c in chunks if len(c.text.strip()) < self.min_text_length)
        too_long_text = sum(1 for c in chunks if len(c.text) > self.max_text_length)
        
        if too_short_text > len(chunks) * 0.2:
            issues.append(f"Many chunks with minimal text ({too_short_text})")
        if too_long_text > len(chunks) * 0.1:
            issues.append(f"Some chunks with excessive text ({too_long_text})")
        
        # Check for excessive repetition
        all_text = " ".join(chunk.text for chunk in chunks).lower()
        words = all_text.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_count = max(word_counts.values())
            repetition_ratio = max_count / len(words)
            
            if repetition_ratio > self.repetition_threshold:
                most_repeated = max(word_counts.items(), key=lambda x: x[1])
                issues.append(f"Excessive repetition: '{most_repeated[0]}' appears {most_repeated[1]} times")
        
        return issues
    
    def _validate_coverage(self, chunks: List[CaptionChunk], video_duration_seconds: float) -> List[str]:
        """Check caption coverage of video"""
        issues = []
        
        if not chunks:
            return ["No coverage"]
        
        # Calculate coverage ratio
        span_ms = chunks[-1].end_ms - chunks[0].start_ms
        video_duration_ms = video_duration_seconds * 1000
        coverage_ratio = span_ms / video_duration_ms
        
        if coverage_ratio < 0.8:  # Less than 80% coverage
            issues.append(f"Low coverage: only {coverage_ratio*100:.1f}% of video")
        
        # Check for large uncovered gaps at start/end
        start_delay = chunks[0].start_ms / 1000.0  # seconds
        end_gap = video_duration_seconds - (chunks[-1].end_ms / 1000.0)
        
        if start_delay > 30:  # More than 30s before first caption
            issues.append(f"Large gap at start: {start_delay:.1f}s uncovered")
        if end_gap > 30:  # More than 30s after last caption
            issues.append(f"Large gap at end: {end_gap:.1f}s uncovered")
        
        return issues
    
    def _generate_suggestions(self, issues: List[str], stats: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on detected issues"""
        suggestions = []
        
        issue_text = " ".join(issues).lower()
        
        if "overlap" in issue_text:
            suggestions.append("Fix overlapping segments with proper timing adjustment")
        
        if "gap" in issue_text:
            suggestions.append("Fill large gaps or adjust timing to reduce silence")
        
        if "fast" in issue_text:
            suggestions.append("Slow down reading speed by extending duration or splitting text")
        
        if "slow" in issue_text:
            suggestions.append("Optimize reading speed by merging short segments")
        
        if "noise" in issue_text or "artifacts" in issue_text:
            suggestions.append("Clean up caption text by removing [Music], (inaudible) tags")
        
        if "repetition" in issue_text:
            suggestions.append("Review for transcription errors causing word repetition")
        
        if "coverage" in issue_text:
            suggestions.append("Try different caption source or improve extraction method")
        
        if "short" in issue_text:
            suggestions.append("Merge consecutive short segments for better readability")
        
        if "long" in issue_text:
            suggestions.append("Split long segments at sentence boundaries")
        
        return suggestions
    
    def _calculate_quality_score(self, chunks: List[CaptionChunk], issues: List[str], stats: Dict[str, Any]) -> float:
        """Calculate overall quality score 0.0-1.0"""
        if not chunks:
            return 0.0
        
        score = 1.0
        
        # Deduct for each issue category
        issue_categories = {
            "overlap": 0.1,
            "gap": 0.1, 
            "fast": 0.2,
            "slow": 0.1,
            "noise": 0.15,
            "repetition": 0.2,
            "coverage": 0.25,
            "short": 0.05,
            "long": 0.05
        }
        
        issue_text = " ".join(issues).lower()
        for category, penalty in issue_categories.items():
            if category in issue_text:
                score -= penalty
        
        # Bonus for good metrics
        if stats.get("avg_cps", 0) >= 15 and stats.get("avg_cps", 0) <= 20:
            score += 0.1  # Good reading speed
        
        if stats.get("word_diversity", 0) > 0.6:
            score += 0.05  # Good vocabulary diversity
        
        if stats.get("coverage_ratio", 0) > 0.9:
            score += 0.1  # Excellent coverage
        
        return max(0.0, min(1.0, score))
    
    def _count_gaps(self, chunks: List[CaptionChunk]) -> int:
        """Count significant gaps between chunks"""
        gaps = 0
        for i in range(len(chunks) - 1):
            gap = chunks[i + 1].start_ms - chunks[i].end_ms
            if gap > self.max_gap_ms:
                gaps += 1
        return gaps
    
    def _count_overlaps(self, chunks: List[CaptionChunk]) -> int:
        """Count overlapping chunks"""
        overlaps = 0
        for i in range(len(chunks) - 1):
            if chunks[i].end_ms > chunks[i + 1].start_ms:
                overlaps += 1
        return overlaps


def validate_captions(chunks: List[CaptionChunk], video_duration_seconds: Optional[float] = None) -> ValidationResult:
    """
    Convenience function for simple validation
    Usage: result = validate_captions(chunks, 1800.0)
    """
    validator = CaptionValidator()
    return validator.validate(chunks, video_duration_seconds)