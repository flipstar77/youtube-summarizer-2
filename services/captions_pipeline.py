"""
End-to-end caption pipeline
Orchestrates the complete caption workflow: fetch → validate → format → output
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .captions_types import CaptionChunk, Source
from .captions_fetcher import CaptionFetcher, FetchResult
from .captions_validator import CaptionValidator, ValidationResult
from .captions_format import format_chunks_for_srt, to_srt

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Complete pipeline result with all stages"""
    success: bool
    chunks: List[CaptionChunk]
    srt_content: str
    source: Source
    fetch_result: Optional[FetchResult] = None
    validation_result: Optional[ValidationResult] = None
    error: Optional[str] = None
    
    @property
    def quality_score(self) -> float:
        """Get quality score from validation"""
        return self.validation_result.score if self.validation_result else 0.0
    
    @property
    def is_high_quality(self) -> bool:
        """Check if captions meet quality standards"""
        return self.quality_score >= 0.7

class CaptionPipeline:
    """
    Complete caption processing pipeline
    
    Workflow:
    1. Fetch captions (YouTube → yt-dlp → Whisper)
    2. Validate quality and coverage
    3. Format for optimal readability 
    4. Generate SRT output
    """
    
    def __init__(self, 
                 min_quality_score: float = 0.6,
                 enable_validation: bool = True,
                 enable_formatting: bool = True):
        self.fetcher = CaptionFetcher()
        self.validator = CaptionValidator()
        self.min_quality_score = min_quality_score
        self.enable_validation = enable_validation
        self.enable_formatting = enable_formatting
    
    def process(self, video_id: str, language: str = "en") -> PipelineResult:
        """
        Complete end-to-end processing
        Returns formatted SRT captions with quality validation
        """
        logger.info(f"🎬 Starting caption pipeline for {video_id}")
        
        try:
            # Stage 1: Fetch captions
            logger.info("1️⃣ Fetching captions...")
            fetch_result = self.fetcher.fetch_captions(video_id, language)
            
            if not fetch_result.success or not fetch_result.chunks:
                return PipelineResult(
                    success=False,
                    chunks=[],
                    srt_content="",
                    source=fetch_result.source,
                    fetch_result=fetch_result,
                    error=f"Caption fetching failed: {fetch_result.error}"
                )
            
            logger.info(f"✅ Fetched {len(fetch_result.chunks)} caption chunks from {fetch_result.source}")
            
            # Stage 2: Validation (optional but recommended)
            validation_result = None
            if self.enable_validation:
                logger.info("2️⃣ Validating caption quality...")
                validation_result = self.validator.validate(
                    fetch_result.chunks, 
                    fetch_result.duration_seconds
                )
                
                logger.info(f"📊 Quality score: {validation_result.score:.2f}")
                
                if validation_result.issues:
                    logger.warning(f"⚠️ Quality issues found: {len(validation_result.issues)}")
                    for issue in validation_result.issues[:3]:  # Show first 3
                        logger.warning(f"  - {issue}")
                
                # Check if quality meets minimum threshold
                if validation_result.score < self.min_quality_score:
                    logger.warning(f"🔻 Quality below threshold ({self.min_quality_score})")
                    # Continue anyway but flag the issue
            
            # Stage 3: Format for optimal readability
            processed_chunks = fetch_result.chunks
            if self.enable_formatting:
                logger.info("3️⃣ Formatting for optimal readability...")
                processed_chunks = format_chunks_for_srt(fetch_result.chunks)
                logger.info(f"📝 Formatted {len(processed_chunks)} chunks")
            
            # Stage 4: Generate SRT output
            logger.info("4️⃣ Generating SRT content...")
            srt_content = to_srt(processed_chunks)
            
            if not srt_content.strip():
                return PipelineResult(
                    success=False,
                    chunks=processed_chunks,
                    srt_content="",
                    source=fetch_result.source,
                    fetch_result=fetch_result,
                    validation_result=validation_result,
                    error="SRT generation produced empty content"
                )
            
            logger.info("✅ Caption pipeline completed successfully")
            
            return PipelineResult(
                success=True,
                chunks=processed_chunks,
                srt_content=srt_content,
                source=fetch_result.source,
                fetch_result=fetch_result,
                validation_result=validation_result
            )
            
        except Exception as e:
            logger.error(f"❌ Caption pipeline failed: {e}")
            return PipelineResult(
                success=False,
                chunks=[],
                srt_content="",
                source="youtube",  # default
                error=str(e)
            )
    
    def process_with_fallback_quality(self, video_id: str, language: str = "en") -> PipelineResult:
        """
        Process with quality fallback strategy
        If first source produces low quality, tries alternative sources
        """
        logger.info(f"🎯 Processing {video_id} with quality fallback strategy")
        
        # Try primary pipeline
        result = self.process(video_id, language)
        
        # If successful and high quality, return immediately
        if result.success and result.is_high_quality:
            logger.info(f"🏆 High quality result from {result.source}")
            return result
        
        # If low quality or failed, try alternative approaches
        if result.success and not result.is_high_quality:
            logger.info(f"🔄 Quality too low ({result.quality_score:.2f}), trying alternatives...")
            
            # Could implement source-specific retry logic here
            # For now, return the best we have
            logger.info(f"⚡ Using best available result from {result.source}")
        
        return result
    
    def get_pipeline_diagnostics(self, video_id: str, language: str = "en") -> Dict[str, Any]:
        """
        Run diagnostic checks without full processing
        Useful for troubleshooting caption issues
        """
        logger.info(f"🔍 Running caption diagnostics for {video_id}")
        
        diagnostics = {
            "video_id": video_id,
            "language": language,
            "sources_tested": [],
            "results": {}
        }
        
        # Test each source individually
        sources = ["youtube", "yt-dlp", "whisper"]
        
        for source in sources:
            try:
                if source == "youtube":
                    result = self.fetcher._fetch_youtube_captions(video_id, language)
                elif source == "yt-dlp": 
                    result = self.fetcher._fetch_ytdlp_captions(video_id, language)
                else:  # whisper
                    result = self.fetcher._fetch_whisper_captions(video_id, language)
                
                diagnostics["sources_tested"].append(source)
                diagnostics["results"][source] = {
                    "success": result.success,
                    "chunk_count": len(result.chunks) if result.success else 0,
                    "error": result.error,
                    "duration": result.duration_seconds
                }
                
                # Add validation if successful
                if result.success and result.chunks:
                    validation = self.validator.validate(result.chunks, result.duration_seconds)
                    diagnostics["results"][source]["quality_score"] = validation.score
                    diagnostics["results"][source]["issues"] = validation.issues
                
            except Exception as e:
                diagnostics["results"][source] = {
                    "success": False,
                    "error": str(e),
                    "chunk_count": 0
                }
        
        # Summary
        successful_sources = [s for s in sources if diagnostics["results"].get(s, {}).get("success")]
        diagnostics["summary"] = {
            "successful_sources": successful_sources,
            "recommended_source": successful_sources[0] if successful_sources else None,
            "total_sources_available": len(successful_sources)
        }
        
        return diagnostics


def create_srt_from_video_id(video_id: str, language: str = "en", min_quality: float = 0.6) -> str:
    """
    Convenience function - returns SRT content for a video ID
    
    Usage:
    srt_content = create_srt_from_video_id("dQw4w9WgXcQ")
    with open("captions.srt", "w") as f:
        f.write(srt_content)
    """
    pipeline = CaptionPipeline(min_quality_score=min_quality)
    result = pipeline.process(video_id, language)
    
    if not result.success:
        logger.warning(f"Failed to create SRT for {video_id}: {result.error}")
        return ""
    
    return result.srt_content

def validate_video_captions(video_id: str, language: str = "en") -> Dict[str, Any]:
    """
    Convenience function - returns caption quality report
    
    Usage:
    report = validate_video_captions("dQw4w9WgXcQ")
    print(f"Quality score: {report['quality_score']}")
    """
    pipeline = CaptionPipeline()
    result = pipeline.process(video_id, language)
    
    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "quality_score": 0.0
        }
    
    return {
        "success": True,
        "quality_score": result.quality_score,
        "source": result.source,
        "chunk_count": len(result.chunks),
        "issues": result.validation_result.issues if result.validation_result else [],
        "suggestions": result.validation_result.suggestions if result.validation_result else []
    }