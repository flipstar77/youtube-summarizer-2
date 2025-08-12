#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket Progress Manager
Real-time progress updates for video processing operations
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Lock

class ProgressTracker:
    """Track and broadcast progress updates"""
    
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_lock = Lock()
        self.socketio: Optional[SocketIO] = None
    
    def init_socketio(self, socketio: SocketIO):
        """Initialize SocketIO instance"""
        self.socketio = socketio
        print("[OK] WebSocket progress tracker initialized")
    
    def create_job(self, job_type: str, video_url: str = "", user_id: str = "default") -> str:
        """Create a new progress tracking job"""
        job_id = str(uuid.uuid4())
        
        with self.job_lock:
            self.active_jobs[job_id] = {
                'id': job_id,
                'type': job_type,
                'status': 'started',
                'progress': 0,
                'current_step': 'Initializing...',
                'total_steps': 100,
                'video_url': video_url,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'steps': [],
                'errors': []
            }
        
        self._broadcast_update(job_id)
        print(f"[INFO] Created progress job: {job_id} ({job_type})")
        return job_id
    
    def update_progress(self, job_id: str, progress: int, step: str, **kwargs):
        """Update job progress"""
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job['progress'] = progress
                job['current_step'] = step
                job['updated_at'] = datetime.now().isoformat()
                
                # Add optional data
                for key, value in kwargs.items():
                    job[key] = value
                
                # Log step
                job['steps'].append({
                    'step': step,
                    'progress': progress,
                    'timestamp': datetime.now().isoformat()
                })
        
        self._broadcast_update(job_id)
        print(f"[PROGRESS] {job_id}: {progress}% - {step}")
    
    def add_error(self, job_id: str, error: str):
        """Add error to job"""
        with self.job_lock:
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['errors'].append({
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
        
        self._broadcast_update(job_id)
        print(f"[ERROR] {job_id}: {error}")
    
    def complete_job(self, job_id: str, result: Dict[str, Any] = None):
        """Mark job as completed"""
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job['status'] = 'completed'
                job['progress'] = 100
                job['current_step'] = 'Completed successfully'
                job['completed_at'] = datetime.now().isoformat()
                
                if result:
                    job['result'] = result
        
        self._broadcast_update(job_id)
        print(f"[SUCCESS] Job completed: {job_id}")
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        with self.job_lock:
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job['status'] = 'failed'
                job['current_step'] = f'Failed: {error}'
                job['failed_at'] = datetime.now().isoformat()
                job['errors'].append({
                    'error': error,
                    'timestamp': datetime.now().isoformat()
                })
        
        self._broadcast_update(job_id)
        print(f"[FAILED] Job failed: {job_id} - {error}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        with self.job_lock:
            return self.active_jobs.get(job_id)
    
    def get_user_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all jobs for a user"""
        with self.job_lock:
            return [job for job in self.active_jobs.values() 
                   if job.get('user_id') == user_id]
    
    def cleanup_completed_jobs(self, hours: int = 24):
        """Remove completed jobs older than specified hours"""
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self.job_lock:
            to_remove = []
            for job_id, job in self.active_jobs.items():
                if job['status'] in ['completed', 'failed']:
                    job_time = datetime.fromisoformat(job.get('completed_at', job.get('failed_at', job['created_at'])))
                    if job_time < cutoff:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self.active_jobs[job_id]
        
        print(f"[INFO] Cleaned up {len(to_remove)} old jobs")
    
    def _broadcast_update(self, job_id: str):
        """Broadcast progress update via WebSocket"""
        if self.socketio and job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            self.socketio.emit('progress_update', job, room=f"job_{job_id}")
            
            # Also emit to user room
            user_id = job.get('user_id', 'default')
            self.socketio.emit('job_update', {
                'job_id': job_id,
                'progress': job['progress'],
                'step': job['current_step'],
                'status': job['status']
            }, room=f"user_{user_id}")

# Global progress tracker
progress_tracker = ProgressTracker()

# SocketIO event handlers
def register_socketio_events(socketio: SocketIO):
    """Register SocketIO event handlers"""
    progress_tracker.init_socketio(socketio)
    
    @socketio.on('connect')
    def handle_connect():
        print(f"[WEBSOCKET] Client connected")
        emit('connected', {'status': 'Connected to progress updates'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print(f"[WEBSOCKET] Client disconnected")
    
    @socketio.on('join_job')
    def handle_join_job(data):
        """Join a job room to receive updates"""
        job_id = data.get('job_id')
        if job_id:
            join_room(f"job_{job_id}")
            # Send current status
            job = progress_tracker.get_job_status(job_id)
            if job:
                emit('progress_update', job)
            print(f"[WEBSOCKET] Client joined job room: {job_id}")
    
    @socketio.on('leave_job')
    def handle_leave_job(data):
        """Leave a job room"""
        job_id = data.get('job_id')
        if job_id:
            leave_room(f"job_{job_id}")
            print(f"[WEBSOCKET] Client left job room: {job_id}")
    
    @socketio.on('join_user')
    def handle_join_user(data):
        """Join user room for general updates"""
        user_id = data.get('user_id', 'default')
        join_room(f"user_{user_id}")
        
        # Send current jobs
        jobs = progress_tracker.get_user_jobs(user_id)
        emit('user_jobs', {'jobs': jobs})
        print(f"[WEBSOCKET] Client joined user room: {user_id}")
    
    @socketio.on('get_job_status')
    def handle_get_job_status(data):
        """Get current job status"""
        job_id = data.get('job_id')
        if job_id:
            job = progress_tracker.get_job_status(job_id)
            emit('job_status', {'job': job})

# Context manager for progress tracking
class ProgressContext:
    """Context manager for easy progress tracking"""
    
    def __init__(self, job_type: str, video_url: str = "", user_id: str = "default"):
        self.job_type = job_type
        self.video_url = video_url
        self.user_id = user_id
        self.job_id = None
    
    def __enter__(self):
        self.job_id = progress_tracker.create_job(
            self.job_type, self.video_url, self.user_id
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            error_msg = str(exc_val) if exc_val else "Unknown error"
            progress_tracker.fail_job(self.job_id, error_msg)
        else:
            progress_tracker.complete_job(self.job_id)
    
    def update(self, progress: int, step: str, **kwargs):
        """Update progress"""
        if self.job_id:
            progress_tracker.update_progress(self.job_id, progress, step, **kwargs)
    
    def error(self, error: str):
        """Add error"""
        if self.job_id:
            progress_tracker.add_error(self.job_id, error)