"""
SQLite Database Interface - DEVELOPMENT FALLBACK ONLY

WARNING: This is a legacy SQLite implementation used only as a development fallback.
PRIMARY DATABASE: Supabase (see supabase_client.py)

Features NOT available in SQLite fallback:
- Vector similarity search
- Real-time subscriptions  
- Advanced indexing
- Concurrent access optimization

Use this only when Supabase is unavailable during development.
"""
import sqlite3
import json
from datetime import datetime


class Database:
    def __init__(self, db_path="summaries.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                summary_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                transcript_length INTEGER,
                audio_file TEXT,
                voice_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_summary(self, video_id, url, title, summary_type, summary, transcript_length, audio_file=None, voice_id=None):
        """Save a summary to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO summaries (video_id, url, title, summary_type, summary, transcript_length, audio_file, voice_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, url, title, summary_type, summary, transcript_length, audio_file, voice_id))
        
        summary_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return summary_id
    
    def get_all_summaries(self):
        """Get all summaries from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, video_id, url, title, summary_type, summary, 
                   transcript_length, audio_file, voice_id, created_at
            FROM summaries 
            ORDER BY created_at DESC
        ''')
        
        summaries = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return summaries
    
    def get_summary(self, summary_id):
        """Get a specific summary by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, video_id, url, title, summary_type, summary, 
                   transcript_length, audio_file, voice_id, created_at
            FROM summaries 
            WHERE id = ?
        ''', (summary_id,))
        
        summary = cursor.fetchone()
        conn.close()
        
        return dict(summary) if summary else None
    
    def delete_summary(self, summary_id):
        """Delete a summary by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM summaries WHERE id = ?', (summary_id,))
        rows_affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return rows_affected > 0
    
    def update_audio_file(self, summary_id, audio_file, voice_id):
        """Update a summary with audio file information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE summaries 
            SET audio_file = ?, voice_id = ?
            WHERE id = ?
        ''', (audio_file, voice_id, summary_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return rows_affected > 0