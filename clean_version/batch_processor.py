#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Video Processor
Handle multiple videos with progress tracking and parallel processing
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re

from websocket_manager import progress_tracker, ProgressContext
from cache_manager import cache

@dataclass
class BatchJob:
    """Batch processing job"""
    urls: List[str]
    ai_provider: str = 'openai'
    model_name: str = 'gpt-4o-mini'
    custom_prompt: str = ''
    max_parallel: int = 3
    user_id: str = 'default'
    job_id: str = ''
    
class BatchVideoProcessor:
    """Process multiple videos with progress tracking"""
    
    def __init__(self, summarizer, database):
        self.summarizer = summarizer
        self.db = database
        self.max_workers = 5  # Maximum parallel workers
        
    def extract_urls_from_text(self, text: str) -> List[str]:
        """Extract YouTube URLs from text input"""
        # YouTube URL patterns
        patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)',
        ]
        
        urls = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
                
            # Check if it's already a full URL
            if 'youtube.com' in line or 'youtu.be' in line:
                urls.append(line)
            else:
                # Try to match patterns
                for pattern in patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        urls.append(f"https://www.youtube.com/watch?v={match}")
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in urls:
            if url not in seen:
                unique_urls.append(url)
                seen.add(url)
        
        return unique_urls
    
    def validate_batch_job(self, job: BatchJob) -> Dict[str, Any]:
        """Validate batch job parameters"""
        errors = []
        warnings = []
        
        # Check URLs
        if not job.urls:
            errors.append("No URLs provided")
        
        if len(job.urls) > 100:  # Reasonable limit
            errors.append("Maximum 100 videos per batch")
        
        # Check for duplicates
        if len(job.urls) != len(set(job.urls)):
            warnings.append("Duplicate URLs detected and removed")
            job.urls = list(set(job.urls))
        
        # Check AI provider
        available_providers = self.summarizer.get_available_providers()
        if job.ai_provider not in available_providers:
            errors.append(f"AI provider '{job.ai_provider}' not available")
        
        # Check parallel processing limit
        if job.max_parallel > 10:
            warnings.append("Limiting parallel processing to 10 videos")
            job.max_parallel = min(job.max_parallel, 10)
        
        # Check for already processed videos
        processed_count = 0
        for url in job.urls:
            if self._is_video_already_processed(url):
                processed_count += 1
        
        if processed_count > 0:
            warnings.append(f"{processed_count} videos already processed (will skip)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'estimated_time': self._estimate_processing_time(len(job.urls)),
            'total_videos': len(job.urls)
        }
    
    def _is_video_already_processed(self, url: str) -> bool:
        """Check if video is already in database"""
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(url)
            if not video_id:
                return False
            
            # Check database
            summaries = self.db.search_summaries(video_id, limit=1)
            return len(summaries) > 0
            
        except Exception:
            return False
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'youtu\.be/([a-zA-Z0-9_-]+)',
            r'youtube\.com/embed/([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _estimate_processing_time(self, video_count: int) -> str:
        """Estimate total processing time"""
        # Average 2-3 minutes per video, considering parallel processing
        avg_time_per_video = 150  # seconds
        parallel_factor = 0.4  # 40% time reduction with parallel processing
        
        estimated_seconds = (video_count * avg_time_per_video) * parallel_factor
        
        if estimated_seconds < 60:
            return f"{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            return f"{int(estimated_seconds // 60)} minutes"
        else:
            hours = int(estimated_seconds // 3600)
            minutes = int((estimated_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    async def process_batch(self, job: BatchJob) -> Dict[str, Any]:
        """Process batch of videos with progress tracking"""
        
        # Create progress tracking job
        with ProgressContext("batch_processing", user_id=job.user_id) as progress:
            try:
                progress.update(0, f"Starting batch processing of {len(job.urls)} videos")
                
                # Filter out already processed videos
                urls_to_process = []
                skipped_count = 0
                
                for url in job.urls:
                    if not self._is_video_already_processed(url):
                        urls_to_process.append(url)
                    else:
                        skipped_count += 1
                
                if skipped_count > 0:
                    progress.update(5, f"Skipped {skipped_count} already processed videos")
                
                if not urls_to_process:
                    progress.update(100, "All videos already processed")
                    return {
                        'status': 'completed',
                        'processed': 0,
                        'skipped': skipped_count,
                        'failed': 0,
                        'results': []
                    }
                
                progress.update(10, f"Processing {len(urls_to_process)} videos with {job.max_parallel} parallel workers")
                
                # Process videos in parallel
                results = await self._process_videos_parallel(
                    urls_to_process, job, progress
                )
                
                # Calculate final statistics
                successful = len([r for r in results if r['status'] == 'success'])
                failed = len([r for r in results if r['status'] == 'error'])
                
                progress.update(100, f"Batch completed: {successful} successful, {failed} failed, {skipped_count} skipped")
                
                return {
                    'status': 'completed',
                    'processed': successful,
                    'skipped': skipped_count,
                    'failed': failed,
                    'results': results,
                    'total_time': self._format_duration(
                        (datetime.now() - datetime.fromisoformat(progress.job_id)).total_seconds()
                    ) if hasattr(progress, 'start_time') else 'Unknown'
                }
                
            except Exception as e:
                progress.error(f"Batch processing failed: {str(e)}")
                return {
                    'status': 'error',
                    'message': str(e),
                    'processed': 0,
                    'failed': len(job.urls),
                    'results': []
                }
    
    async def _process_videos_parallel(self, urls: List[str], job: BatchJob, progress) -> List[Dict[str, Any]]:
        """Process videos in parallel with controlled concurrency"""
        
        results = []
        completed_count = 0
        total_count = len(urls)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=job.max_parallel) as executor:
            
            # Submit all jobs
            future_to_url = {
                executor.submit(self._process_single_video, url, job): url 
                for url in urls
            }
            
            # Process completed futures
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1
                    
                    # Update progress
                    progress_percent = 10 + (completed_count / total_count) * 80  # 10-90% range
                    progress.update(
                        int(progress_percent), 
                        f"Processed {completed_count}/{total_count} videos"
                    )
                    
                except Exception as e:
                    results.append({
                        'url': url,
                        'status': 'error',
                        'message': str(e)
                    })
                    completed_count += 1
                    
                    progress_percent = 10 + (completed_count / total_count) * 80
                    progress.update(
                        int(progress_percent), 
                        f"Processed {completed_count}/{total_count} videos ({len([r for r in results if r['status'] == 'error'])} errors)"
                    )
        
        return results
    
    def _process_single_video(self, url: str, job: BatchJob) -> Dict[str, Any]:
        """Process a single video"""
        try:
            print(f"[BATCH] Processing video: {url}")
            
            # Check cache first
            cache_key = f"batch_summary:{self._extract_video_id(url) or url}"
            cached_result = cache.get(cache_key)
            
            if cached_result:
                print(f"[BATCH] Using cached result for: {url}")
                return cached_result
            
            # Process the video
            result = self.summarizer.process_video(
                video_url=url,
                ai_provider=job.ai_provider,
                model_name=job.model_name,
                custom_prompt=job.custom_prompt
            )
            
            if result.get('status') == 'success':
                # Save to database
                summary_id = self.db.save_summary(result)
                result['summary_id'] = summary_id
                
                # Cache the result
                cache.set(cache_key, result, expire=3600)  # 1 hour
                
                return {
                    'url': url,
                    'status': 'success',
                    'summary_id': summary_id,
                    'title': result.get('title', 'Unknown'),
                    'duration': result.get('metadata', {}).get('duration', 'Unknown')
                }
            else:
                return {
                    'url': url,
                    'status': 'error',
                    'message': result.get('message', 'Processing failed')
                }
                
        except Exception as e:
            print(f"[BATCH ERROR] {url}: {str(e)}")
            return {
                'url': url,
                'status': 'error',
                'message': str(e)
            }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_batch_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined batch processing templates"""
        return {
            'educational': {
                'name': 'Educational Content',
                'description': 'Optimize for educational videos with detailed summaries',
                'ai_provider': 'openai',
                'model': 'gpt-4o-mini',
                'custom_prompt': 'Focus on educational content, key learning points, and actionable insights.',
                'max_parallel': 3
            },
            'news': {
                'name': 'News & Current Events',
                'description': 'Quick summaries for news and current events',
                'ai_provider': 'openai',
                'model': 'gpt-4o-mini',
                'custom_prompt': 'Provide concise summaries focusing on key facts, dates, and implications.',
                'max_parallel': 5
            },
            'entertainment': {
                'name': 'Entertainment',
                'description': 'Casual summaries for entertainment content',
                'ai_provider': 'openai',
                'model': 'gpt-4o-mini', 
                'custom_prompt': 'Create engaging summaries highlighting key moments and entertainment value.',
                'max_parallel': 4
            },
            'technical': {
                'name': 'Technical Content',
                'description': 'Detailed technical analysis with code examples',
                'ai_provider': 'claude',
                'model': 'claude-3-sonnet-20240229',
                'custom_prompt': 'Focus on technical details, code examples, implementation steps, and best practices.',
                'max_parallel': 2
            }
        }