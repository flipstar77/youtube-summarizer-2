import os
import sqlite3
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Import our existing components
from channel_subscriptions import ChannelSubscriptionManager
from settings_manager import SettingsManager
from enhanced_summarizer import EnhancedSummarizer
from transcript_extractor import TranscriptExtractor
from vector_embeddings import SummaryVectorizer
from database import Database
from supabase_client import SupabaseDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ProcessingPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ProcessingJob:
    id: str
    video_id: str
    channel_subscription_id: int
    video_url: str
    video_title: str
    channel_name: str
    priority: ProcessingPriority
    status: ProcessingStatus
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass 
class ProcessingRule:
    id: str
    name: str
    enabled: bool
    channel_filter: Optional[str] = None  # Channel name/ID pattern
    title_keywords: List[str] = None      # Required keywords in title
    exclude_keywords: List[str] = None    # Excluded keywords  
    min_duration: Optional[int] = None    # Minimum duration in seconds
    max_duration: Optional[int] = None    # Maximum duration in seconds
    min_views: Optional[int] = None       # Minimum view count
    max_age_days: Optional[int] = None    # Maximum age in days
    auto_highlight: bool = False          # Automatically extract highlights
    auto_audio: bool = False              # Automatically generate audio
    priority: ProcessingPriority = ProcessingPriority.NORMAL

class AutomationScheduler:
    """
    Advanced automation scheduler for processing subscribed YouTube channels
    """
    
    def __init__(self):
        """Initialize the automation scheduler"""
        self.scheduler = BackgroundScheduler(timezone='UTC')
        self.subscription_manager = ChannelSubscriptionManager()
        self.settings_manager = SettingsManager()
        
        # Initialize database for job tracking
        self.use_supabase = False  # Force SQLite for now
        self.db = Database()
        self._init_automation_tables()
        
        # Initialize processing components
        self.summarizer = None
        self.transcript_extractor = None
        self.vectorizer = None
        
        try:
            self.summarizer = EnhancedSummarizer()
            self.transcript_extractor = TranscriptExtractor()
            logger.info("[OK] Processing components initialized")
        except Exception as e:
            logger.error(f"[WARNING] Some processing components failed to initialize: {str(e)}")
        
        # Processing state
        self.active_jobs = {}
        self.processing_lock = threading.Lock()
        
        # Default settings
        self.default_check_interval = 1800  # 30 minutes
        self.max_concurrent_jobs = 3
        self.job_timeout = 600  # 10 minutes per job
        
        logger.info("[OK] Automation Scheduler initialized")
    
    def _init_automation_tables(self):
        """Initialize automation-related database tables"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Processing jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    channel_subscription_id INTEGER,
                    video_url TEXT NOT NULL,
                    video_title TEXT NOT NULL,
                    channel_name TEXT NOT NULL,
                    priority INTEGER DEFAULT 2,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scheduled_for TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    summary_id INTEGER,
                    FOREIGN KEY (channel_subscription_id) REFERENCES channel_subscriptions (id),
                    FOREIGN KEY (summary_id) REFERENCES summaries (id)
                )
            ''')
            
            # Processing rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    channel_filter TEXT,
                    title_keywords TEXT,
                    exclude_keywords TEXT,
                    min_duration INTEGER,
                    max_duration INTEGER,
                    min_views INTEGER,
                    max_age_days INTEGER,
                    auto_highlight BOOLEAN DEFAULT FALSE,
                    auto_audio BOOLEAN DEFAULT FALSE,
                    priority INTEGER DEFAULT 2,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Automation logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS automation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_message TEXT NOT NULL,
                    related_job_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (related_job_id) REFERENCES processing_jobs (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("[OK] Automation database tables initialized")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize automation tables: {str(e)}")
    
    def start(self):
        """Start the automation scheduler"""
        try:
            # Add default jobs
            self._add_default_jobs()
            
            # Start the scheduler
            self.scheduler.start()
            logger.info("[OK] Automation Scheduler started")
            
            # Log startup event
            self._log_event("scheduler_started", "Automation scheduler started successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start automation scheduler: {str(e)}")
            raise
    
    def stop(self):
        """Stop the automation scheduler"""
        try:
            self.scheduler.shutdown(wait=False)
            logger.info("[OK] Automation Scheduler stopped")
            self._log_event("scheduler_stopped", "Automation scheduler stopped")
        except Exception as e:
            logger.error(f"[ERROR] Failed to stop scheduler: {str(e)}")
    
    def _add_default_jobs(self):
        """Add default scheduled jobs"""
        # Get check interval from settings
        check_interval = self.settings_manager.get_setting('subscription_check_interval', self.default_check_interval)
        
        # Schedule RSS feed monitoring
        self.scheduler.add_job(
            func=self._check_subscriptions_job,
            trigger=IntervalTrigger(seconds=check_interval),
            id='subscription_monitor',
            name='RSS Feed Monitor',
            replace_existing=True,
            max_instances=1
        )
        
        # Schedule job processing (every 2 minutes)
        self.scheduler.add_job(
            func=self._process_pending_jobs,
            trigger=IntervalTrigger(seconds=120),
            id='job_processor',
            name='Job Processor',
            replace_existing=True,
            max_instances=1
        )
        
        # Schedule cleanup (daily at 2 AM)
        self.scheduler.add_job(
            func=self._cleanup_old_jobs,
            trigger=CronTrigger(hour=2, minute=0),
            id='cleanup_jobs',
            name='Job Cleanup',
            replace_existing=True
        )
        
        # Schedule retry failed jobs (every hour)
        self.scheduler.add_job(
            func=self._retry_failed_jobs,
            trigger=CronTrigger(minute=0),
            id='retry_failed',
            name='Retry Failed Jobs',
            replace_existing=True
        )
        
        logger.info("[OK] Default automation jobs scheduled")
    
    def _check_subscriptions_job(self):
        """Background job to check subscriptions for new videos"""
        try:
            logger.info("[INFO] Starting subscription check...")
            
            # Refresh all subscriptions
            result = self.subscription_manager.refresh_all_subscriptions()
            
            if result['success']:
                logger.info(f"[INFO] Subscription check completed: {result['new_videos']} new videos found")
                
                # Create processing jobs for new videos
                self._create_jobs_for_new_videos()
                
                self._log_event("subscription_check", 
                              f"Found {result['new_videos']} new videos from {result['refreshed_channels']} channels")
            else:
                logger.error(f"[ERROR] Subscription check failed: {result['error']}")
                self._log_event("subscription_check_failed", result['error'])
                
        except Exception as e:
            logger.error(f"[ERROR] Subscription check job failed: {str(e)}")
            self._log_event("subscription_check_error", str(e))
    
    def _create_jobs_for_new_videos(self):
        """Create processing jobs for newly discovered videos"""
        try:
            # Get recently discovered unprocessed videos
            discovered_videos = self.subscription_manager.get_discovered_videos(limit=50)
            
            jobs_created = 0
            
            for video in discovered_videos:
                # Skip if already processed or has existing job
                if video.get('auto_processed') or self._job_exists(video['video_id']):
                    continue
                
                # Check if channel has auto-processing enabled
                subscriptions = self.subscription_manager.get_all_subscriptions()
                channel_sub = next((s for s in subscriptions if s['id'] == video['channel_subscription_id']), None)
                
                if not channel_sub or not channel_sub.get('auto_process'):
                    continue
                
                # Apply processing rules
                if not self._should_process_video(video):
                    logger.info(f"[INFO] Video {video['title'][:50]}... skipped by rules")
                    continue
                
                # Create processing job
                job = self._create_processing_job(video)
                if job:
                    jobs_created += 1
            
            if jobs_created > 0:
                logger.info(f"[INFO] Created {jobs_created} new processing jobs")
                self._log_event("jobs_created", f"Created {jobs_created} new processing jobs")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to create processing jobs: {str(e)}")
    
    def _should_process_video(self, video: Dict[str, Any]) -> bool:
        """Check if video should be processed based on rules"""
        try:
            # Get processing rules
            rules = self._get_processing_rules()
            
            for rule in rules:
                if not rule.enabled:
                    continue
                
                # Channel filter
                if rule.channel_filter:
                    if rule.channel_filter.lower() not in video.get('channel_name', '').lower():
                        continue
                
                # Title keyword requirements
                if rule.title_keywords:
                    title_lower = video.get('title', '').lower()
                    if not any(keyword.lower() in title_lower for keyword in rule.title_keywords):
                        continue
                
                # Excluded keywords
                if rule.exclude_keywords:
                    title_lower = video.get('title', '').lower()
                    if any(keyword.lower() in title_lower for keyword in rule.exclude_keywords):
                        continue
                
                # Age check
                if rule.max_age_days:
                    try:
                        published_date = datetime.fromisoformat(video.get('published_date', ''))
                        age_days = (datetime.now() - published_date).days
                        if age_days > rule.max_age_days:
                            continue
                    except:
                        pass
                
                # If we reach here, video passes this rule
                return True
            
            # If no rules match, check default auto-process setting
            return True  # Default to processing if no rules defined
            
        except Exception as e:
            logger.error(f"[ERROR] Error checking processing rules: {str(e)}")
            return True  # Default to processing on error
    
    def _create_processing_job(self, video: Dict[str, Any]) -> Optional[ProcessingJob]:
        """Create a new processing job"""
        try:
            job_id = f"job_{video['video_id']}_{int(time.time())}"
            
            job = ProcessingJob(
                id=job_id,
                video_id=video['video_id'],
                channel_subscription_id=video['channel_subscription_id'],
                video_url=video['url'],
                video_title=video['title'],
                channel_name=video.get('channel_name', 'Unknown'),
                priority=ProcessingPriority.NORMAL,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now()
            )
            
            # Save to database
            self._save_job(job)
            
            logger.info(f"[INFO] Created processing job for: {video['title'][:50]}...")
            return job
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create processing job: {str(e)}")
            return None
    
    def _save_job(self, job: ProcessingJob):
        """Save processing job to database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_jobs 
                (id, video_id, channel_subscription_id, video_url, video_title, 
                 channel_name, priority, status, created_at, scheduled_for, 
                 retry_count, max_retries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.id, job.video_id, job.channel_subscription_id, job.video_url,
                job.video_title, job.channel_name, job.priority.value, job.status.value,
                job.created_at.isoformat(), 
                job.scheduled_for.isoformat() if job.scheduled_for else None,
                job.retry_count, job.max_retries
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save job to database: {str(e)}")
    
    def _process_pending_jobs(self):
        """Process pending jobs in the queue"""
        try:
            # Get pending jobs
            pending_jobs = self._get_pending_jobs()
            
            if not pending_jobs:
                return
            
            logger.info(f"[INFO] Processing {len(pending_jobs)} pending jobs")
            
            # Limit concurrent processing
            active_count = len(self.active_jobs)
            available_slots = max(0, self.max_concurrent_jobs - active_count)
            
            # Sort by priority and creation time
            pending_jobs.sort(key=lambda j: (j.priority.value, j.created_at), reverse=True)
            
            # Process up to available slots
            for job in pending_jobs[:available_slots]:
                self._start_job_processing(job)
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to process pending jobs: {str(e)}")
    
    def _get_pending_jobs(self) -> List[ProcessingJob]:
        """Get pending jobs from database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM processing_jobs 
                WHERE status = 'pending' 
                AND (scheduled_for IS NULL OR scheduled_for <= ?)
                ORDER BY priority DESC, created_at ASC
                LIMIT 10
            ''', (datetime.now().isoformat(),))
            
            rows = cursor.fetchall()
            conn.close()
            
            jobs = []
            for row in rows:
                job = ProcessingJob(
                    id=row[0],
                    video_id=row[1],
                    channel_subscription_id=row[2],
                    video_url=row[3],
                    video_title=row[4],
                    channel_name=row[5],
                    priority=ProcessingPriority(row[6]),
                    status=ProcessingStatus(row[7]),
                    created_at=datetime.fromisoformat(row[8]),
                    scheduled_for=datetime.fromisoformat(row[9]) if row[9] else None,
                    started_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    completed_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    error_message=row[12],
                    retry_count=row[13],
                    max_retries=row[14]
                )
                jobs.append(job)
            
            return jobs
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get pending jobs: {str(e)}")
            return []
    
    def _start_job_processing(self, job: ProcessingJob):
        """Start processing a job in a background thread"""
        try:
            # Mark job as processing
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.now()
            self._update_job(job)
            
            # Add to active jobs
            with self.processing_lock:
                self.active_jobs[job.id] = job
            
            # Start processing thread
            thread = threading.Thread(target=self._process_job, args=(job,))
            thread.daemon = True
            thread.start()
            
            logger.info(f"[INFO] Started processing job: {job.video_title[:50]}...")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to start job processing: {str(e)}")
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            self._update_job(job)
    
    def _process_job(self, job: ProcessingJob):
        """Actually process a video (runs in background thread)"""
        try:
            logger.info(f"[INFO] Processing video: {job.video_title}")
            
            # Extract video ID from URL if needed
            video_id = self._extract_video_id_from_url(job.video_url)
            if not video_id:
                raise Exception("Could not extract video ID from URL")
            
            # Create summary using existing system
            if not self.summarizer:
                raise Exception("Summarizer not available")
            
            # Get or create transcript
            try:
                transcript_extractor = TranscriptExtractor()
                transcript = transcript_extractor.extract_transcript(video_id)
                if not transcript:
                    raise Exception("Could not extract transcript")
            except Exception as e:
                logger.error(f"[ERROR] Transcript extraction failed: {str(e)}")
                raise
            
            # Generate summary
            try:
                summary_result = self.summarizer.summarize(transcript, 'detailed')
                if not summary_result:
                    raise Exception("Summary generation failed")
            except Exception as e:
                logger.error(f"[ERROR] Summary generation failed: {str(e)}")
                raise
            
            # Save to database
            try:
                summary_id = self.db.save_summary(
                    video_id=video_id,
                    url=job.video_url,
                    title=job.video_title,
                    summary_type='detailed',
                    summary=summary_result,
                    transcript_length=len(transcript)
                )
                
                # Update job with summary ID
                job.summary_id = summary_id
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to save summary: {str(e)}")
                raise
            
            # Check for additional processing (highlights, audio)
            rules = self._get_processing_rules()
            for rule in rules:
                if not rule.enabled:
                    continue
                    
                # Auto-highlight processing
                if rule.auto_highlight:
                    try:
                        self._process_highlights(job, summary_id)
                    except Exception as e:
                        logger.warning(f"[WARNING] Highlight processing failed: {str(e)}")
                
                # Auto-audio generation
                if rule.auto_audio:
                    try:
                        self._process_audio(job, summary_id, summary_result)
                    except Exception as e:
                        logger.warning(f"[WARNING] Audio generation failed: {str(e)}")
            
            # Mark as completed
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.now()
            self._update_job(job)
            
            # Mark video as processed in discoveries
            self._mark_video_processed(job.video_id, summary_id)
            
            logger.info(f"[SUCCESS] Completed processing: {job.video_title}")
            self._log_event("job_completed", f"Successfully processed: {job.video_title}", job.id)
            
        except Exception as e:
            logger.error(f"[ERROR] Job processing failed: {str(e)}")
            
            # Mark as failed
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            self._update_job(job)
            
            self._log_event("job_failed", f"Processing failed: {str(e)}", job.id)
            
        finally:
            # Remove from active jobs
            with self.processing_lock:
                if job.id in self.active_jobs:
                    del self.active_jobs[job.id]
    
    def _extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        import re
        
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:v\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _process_highlights(self, job: ProcessingJob, summary_id: int):
        """Process highlights for the video"""
        try:
            from video_highlight_extractor import VideoHighlightExtractor
            
            extractor = VideoHighlightExtractor()
            result = extractor.extract_highlights(summary_id)
            
            if result and result.get('success'):
                logger.info(f"[INFO] Generated highlights for {job.video_title}")
            else:
                logger.warning(f"[WARNING] Highlight generation failed for {job.video_title}")
                
        except Exception as e:
            logger.error(f"[ERROR] Highlight processing error: {str(e)}")
            raise
    
    def _process_audio(self, job: ProcessingJob, summary_id: int, summary_text: str):
        """Process audio generation for the video"""
        try:
            from text_to_speech import TextToSpeech
            
            tts = TextToSpeech()
            # Use default voice or get from settings
            voice_id = self.settings_manager.get_setting('default_voice_id', 'default')
            
            audio_file = tts.generate_audio(summary_text, voice_id)
            
            if audio_file:
                # Update summary with audio file
                self.db.update_audio_file(summary_id, audio_file, voice_id)
                logger.info(f"[INFO] Generated audio for {job.video_title}")
            else:
                logger.warning(f"[WARNING] Audio generation failed for {job.video_title}")
                
        except Exception as e:
            logger.error(f"[ERROR] Audio processing error: {str(e)}")
            raise
    
    def _mark_video_processed(self, video_id: str, summary_id: int):
        """Mark video as processed in discovered_videos"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE discovered_videos 
                SET auto_processed = TRUE, processed_summary_id = ?
                WHERE video_id = ?
            ''', (summary_id, video_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to mark video as processed: {str(e)}")
    
    def _update_job(self, job: ProcessingJob):
        """Update job in database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE processing_jobs 
                SET status = ?, started_at = ?, completed_at = ?, error_message = ?, 
                    retry_count = ?, summary_id = ?
                WHERE id = ?
            ''', (
                job.status.value,
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message,
                job.retry_count,
                getattr(job, 'summary_id', None),
                job.id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to update job: {str(e)}")
    
    def _retry_failed_jobs(self):
        """Retry failed jobs that are eligible for retry"""
        try:
            # Get failed jobs that haven't exceeded max retries
            failed_jobs = self._get_failed_jobs_for_retry()
            
            for job in failed_jobs:
                if job.retry_count < job.max_retries:
                    job.status = ProcessingStatus.PENDING
                    job.retry_count += 1
                    job.error_message = None
                    job.scheduled_for = datetime.now() + timedelta(minutes=job.retry_count * 10)  # Exponential backoff
                    
                    self._update_job(job)
                    logger.info(f"[INFO] Scheduled retry {job.retry_count} for: {job.video_title[:50]}...")
                    
        except Exception as e:
            logger.error(f"[ERROR] Failed to retry jobs: {str(e)}")
    
    def _get_failed_jobs_for_retry(self) -> List[ProcessingJob]:
        """Get failed jobs eligible for retry"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM processing_jobs 
                WHERE status = 'failed' 
                AND retry_count < max_retries
                AND completed_at > datetime('now', '-1 hour')
                LIMIT 5
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            jobs = []
            for row in rows:
                job = ProcessingJob(
                    id=row[0],
                    video_id=row[1],
                    channel_subscription_id=row[2],
                    video_url=row[3],
                    video_title=row[4],
                    channel_name=row[5],
                    priority=ProcessingPriority(row[6]),
                    status=ProcessingStatus(row[7]),
                    created_at=datetime.fromisoformat(row[8]),
                    scheduled_for=datetime.fromisoformat(row[9]) if row[9] else None,
                    started_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    completed_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    error_message=row[12],
                    retry_count=row[13],
                    max_retries=row[14]
                )
                jobs.append(job)
            
            return jobs
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get failed jobs: {str(e)}")
            return []
    
    def _cleanup_old_jobs(self):
        """Clean up old completed/failed jobs"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Delete jobs older than 7 days
            cursor.execute('''
                DELETE FROM processing_jobs 
                WHERE completed_at < datetime('now', '-7 days')
                AND status IN ('completed', 'failed')
            ''')
            
            deleted_count = cursor.rowcount
            
            # Clean up old logs (older than 30 days)
            cursor.execute('''
                DELETE FROM automation_logs 
                WHERE created_at < datetime('now', '-30 days')
            ''')
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"[INFO] Cleaned up {deleted_count} old jobs")
                self._log_event("cleanup_completed", f"Cleaned up {deleted_count} old jobs")
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to clean up old jobs: {str(e)}")
    
    def _job_exists(self, video_id: str) -> bool:
        """Check if a job already exists for this video"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM processing_jobs 
                WHERE video_id = ? 
                AND status IN ('pending', 'processing', 'completed')
            ''', (video_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to check job existence: {str(e)}")
            return False
    
    def _get_processing_rules(self) -> List[ProcessingRule]:
        """Get processing rules from database"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM processing_rules WHERE enabled = TRUE')
            rows = cursor.fetchall()
            conn.close()
            
            rules = []
            for row in rows:
                rule = ProcessingRule(
                    id=row[0],
                    name=row[1],
                    enabled=bool(row[2]),
                    channel_filter=row[3],
                    title_keywords=json.loads(row[4]) if row[4] else None,
                    exclude_keywords=json.loads(row[5]) if row[5] else None,
                    min_duration=row[6],
                    max_duration=row[7],
                    min_views=row[8],
                    max_age_days=row[9],
                    auto_highlight=bool(row[10]),
                    auto_audio=bool(row[11]),
                    priority=ProcessingPriority(row[12])
                )
                rules.append(rule)
            
            return rules
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get processing rules: {str(e)}")
            return []
    
    def _log_event(self, event_type: str, event_message: str, job_id: Optional[str] = None):
        """Log automation event"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO automation_logs (event_type, event_message, related_job_id)
                VALUES (?, ?, ?)
            ''', (event_type, event_message, job_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to log event: {str(e)}")
    
    # Public API methods
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get current automation status"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Get job counts by status
            cursor.execute('''
                SELECT status, COUNT(*) FROM processing_jobs 
                GROUP BY status
            ''')
            status_counts = dict(cursor.fetchall())
            
            # Get recent activity
            cursor.execute('''
                SELECT * FROM automation_logs 
                ORDER BY created_at DESC 
                LIMIT 10
            ''')
            recent_logs = cursor.fetchall()
            
            # Get active jobs
            cursor.execute('''
                SELECT * FROM processing_jobs 
                WHERE status = 'processing'
                ORDER BY started_at DESC
            ''')
            active_jobs = cursor.fetchall()
            
            conn.close()
            
            return {
                'scheduler_running': self.scheduler.running,
                'status_counts': status_counts,
                'active_jobs_count': len(self.active_jobs),
                'active_jobs': [
                    {
                        'id': job[0],
                        'title': job[4][:50] + '...',
                        'channel': job[5],
                        'started_at': job[10]
                    }
                    for job in active_jobs
                ],
                'recent_activity': [
                    {
                        'type': log[1],
                        'message': log[2],
                        'time': log[4]
                    }
                    for log in recent_logs
                ]
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get job status: {str(e)}")
            return {'error': str(e)}
    
    def add_processing_rule(self, rule: ProcessingRule) -> bool:
        """Add a new processing rule"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_rules 
                (id, name, enabled, channel_filter, title_keywords, exclude_keywords,
                 min_duration, max_duration, min_views, max_age_days, auto_highlight, 
                 auto_audio, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.id, rule.name, rule.enabled, rule.channel_filter,
                json.dumps(rule.title_keywords) if rule.title_keywords else None,
                json.dumps(rule.exclude_keywords) if rule.exclude_keywords else None,
                rule.min_duration, rule.max_duration, rule.min_views, rule.max_age_days,
                rule.auto_highlight, rule.auto_audio, rule.priority.value
            ))
            
            conn.commit()
            conn.close()
            
            self._log_event("rule_added", f"Added processing rule: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to add processing rule: {str(e)}")
            return False
    
    def pause_automation(self):
        """Pause the automation scheduler"""
        try:
            self.scheduler.pause()
            self._log_event("scheduler_paused", "Automation scheduler paused")
            logger.info("[INFO] Automation scheduler paused")
        except Exception as e:
            logger.error(f"[ERROR] Failed to pause scheduler: {str(e)}")
    
    def resume_automation(self):
        """Resume the automation scheduler"""
        try:
            self.scheduler.resume()
            self._log_event("scheduler_resumed", "Automation scheduler resumed")
            logger.info("[INFO] Automation scheduler resumed")
        except Exception as e:
            logger.error(f"[ERROR] Failed to resume scheduler: {str(e)}")
    
    def force_subscription_check(self):
        """Force an immediate subscription check"""
        try:
            self._check_subscriptions_job()
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to force subscription check: {str(e)}")
            return False