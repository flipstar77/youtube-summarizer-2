#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Database Manager
Handles both Supabase and SQLite with automatic fallback
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    """Unified database manager with Supabase and SQLite support"""
    
    def __init__(self):
        self.use_supabase = False
        self.supabase_client = None
        self.sqlite_path = 'summaries.db'
        
        # Try to initialize Supabase first
        if self._init_supabase():
            self.use_supabase = True
            print("[OK] Using Supabase as database backend")
        else:
            self._init_sqlite()
            print("[OK] Using SQLite as database backend")
    
    def _init_supabase(self) -> bool:
        """Initialize Supabase client"""
        try:
            from supabase import create_client, Client
            
            url = os.getenv('SUPABASE_URL')
            key = os.getenv('SUPABASE_KEY')
            
            if not url or not key:
                return False
            
            self.supabase_client = create_client(url, key)
            
            # Test connection
            response = self.supabase_client.table('summaries').select('id').limit(1).execute()
            return True
            
        except Exception as e:
            print(f"[WARNING] Supabase initialization failed: {e}")
            return False
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        url TEXT NOT NULL UNIQUE,
                        video_id TEXT,
                        summary TEXT,
                        key_points TEXT,
                        transcript TEXT,
                        metadata TEXT,
                        ai_provider TEXT DEFAULT 'openai',
                        model_used TEXT DEFAULT 'gpt-4o-mini',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                
        except Exception as e:
            print(f"[ERROR] SQLite initialization failed: {e}")
            raise
    
    def save_summary(self, summary_data: Dict[str, Any]) -> Optional[int]:
        """Save summary to database"""
        try:
            if self.use_supabase:
                return self._save_summary_supabase(summary_data)
            else:
                return self._save_summary_sqlite(summary_data)
        except Exception as e:
            print(f"[ERROR] Save summary failed: {e}")
            return None
    
    def _save_summary_supabase(self, data: Dict[str, Any]) -> Optional[int]:
        """Save to Supabase"""
        try:
            summary_record = {
                'title': data.get('title', 'Untitled'),
                'url': data.get('url', ''),
                'video_id': data.get('video_id', ''),
                'summary': data.get('summary', ''),
                'key_points': json.dumps(data.get('key_points', [])),
                'transcript': data.get('transcript', ''),
                'metadata': json.dumps(data.get('metadata', {})),
                'ai_provider': data.get('ai_provider', 'openai'),
                'model_used': data.get('model_used', 'gpt-4o-mini')
            }
            
            response = self.supabase_client.table('summaries').insert(summary_record).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0].get('id')
            return None
            
        except Exception as e:
            print(f"[ERROR] Supabase save failed: {e}")
            return None
    
    def _save_summary_sqlite(self, data: Dict[str, Any]) -> Optional[int]:
        """Save to SQLite"""
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO summaries 
                    (title, url, video_id, summary, key_points, transcript, metadata, ai_provider, model_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('title', 'Untitled'),
                    data.get('url', ''),
                    data.get('video_id', ''),
                    data.get('summary', ''),
                    json.dumps(data.get('key_points', [])),
                    data.get('transcript', ''),
                    json.dumps(data.get('metadata', {})),
                    data.get('ai_provider', 'openai'),
                    data.get('model_used', 'gpt-4o-mini')
                ))
                return cursor.lastrowid
                
        except Exception as e:
            print(f"[ERROR] SQLite save failed: {e}")
            return None
    
    def get_summaries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all summaries"""
        try:
            if self.use_supabase:
                return self._get_summaries_supabase(limit)
            else:
                return self._get_summaries_sqlite(limit)
        except Exception as e:
            print(f"[ERROR] Get summaries failed: {e}")
            return []
    
    def _get_summaries_supabase(self, limit: int) -> List[Dict[str, Any]]:
        """Get from Supabase"""
        try:
            response = self.supabase_client.table('summaries')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            summaries = []
            for row in response.data:
                summary = dict(row)
                # Parse JSON fields
                if summary.get('key_points'):
                    try:
                        summary['key_points'] = json.loads(summary['key_points'])
                    except:
                        summary['key_points'] = []
                
                if summary.get('metadata'):
                    try:
                        summary['metadata'] = json.loads(summary['metadata'])
                    except:
                        summary['metadata'] = {}
                
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            print(f"[ERROR] Supabase get summaries failed: {e}")
            return []
    
    def _get_summaries_sqlite(self, limit: int) -> List[Dict[str, Any]]:
        """Get from SQLite"""
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM summaries 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                summaries = []
                for row in cursor.fetchall():
                    summary = dict(row)
                    # Parse JSON fields
                    if summary.get('key_points'):
                        try:
                            summary['key_points'] = json.loads(summary['key_points'])
                        except:
                            summary['key_points'] = []
                    
                    if summary.get('metadata'):
                        try:
                            summary['metadata'] = json.loads(summary['metadata'])
                        except:
                            summary['metadata'] = {}
                    
                    summaries.append(summary)
                
                return summaries
                
        except Exception as e:
            print(f"[ERROR] SQLite get summaries failed: {e}")
            return []
    
    def get_summary(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """Get single summary by ID"""
        try:
            if self.use_supabase:
                response = self.supabase_client.table('summaries')\
                    .select('*')\
                    .eq('id', summary_id)\
                    .single()\
                    .execute()
                
                if response.data:
                    summary = dict(response.data)
                    # Parse JSON fields
                    if summary.get('key_points'):
                        try:
                            summary['key_points'] = json.loads(summary['key_points'])
                        except:
                            summary['key_points'] = []
                    
                    if summary.get('metadata'):
                        try:
                            summary['metadata'] = json.loads(summary['metadata'])
                        except:
                            summary['metadata'] = {}
                    
                    return summary
                
            else:
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        'SELECT * FROM summaries WHERE id = ?', 
                        (summary_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        summary = dict(row)
                        # Parse JSON fields
                        if summary.get('key_points'):
                            try:
                                summary['key_points'] = json.loads(summary['key_points'])
                            except:
                                summary['key_points'] = []
                        
                        if summary.get('metadata'):
                            try:
                                summary['metadata'] = json.loads(summary['metadata'])
                            except:
                                summary['metadata'] = {}
                        
                        return summary
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Get summary failed: {e}")
            return None
    
    def search_summaries(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search summaries by text content"""
        try:
            if self.use_supabase:
                # Use Supabase full-text search if available
                response = self.supabase_client.table('summaries')\
                    .select('*')\
                    .or_(f"title.ilike.%{query}%,summary.ilike.%{query}%")\
                    .order('created_at', desc=True)\
                    .limit(limit)\
                    .execute()
                
                return [dict(row) for row in response.data]
            
            else:
                # SQLite text search
                with sqlite3.connect(self.sqlite_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute('''
                        SELECT * FROM summaries 
                        WHERE title LIKE ? OR summary LIKE ? 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (f'%{query}%', f'%{query}%', limit))
                    
                    return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"[ERROR] Search summaries failed: {e}")
            return []