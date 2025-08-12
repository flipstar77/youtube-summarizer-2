import os
import sqlite3
import json
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from supabase_client import SupabaseDatabase
from database import Database
import logging

class ChannelSubscriptionManager:
    """
    Manages YouTube channel subscriptions via RSS feeds
    """
    
    def __init__(self):
        """Initialize the subscription manager with the same database as the main app"""
        # For now, force SQLite for subscriptions until Supabase tables are created
        self.use_supabase = False
        self.db = Database()
        print("[INFO] Using SQLite for channel subscriptions")
            
        self._init_subscription_tables()
    
    def _init_subscription_tables(self):
        """Initialize subscription tables in the database"""
        if self.use_supabase:
            # For Supabase, we'll create tables via SQL if they don't exist
            try:
                # Create subscriptions table
                self.db.client.rpc('create_subscriptions_table_if_not_exists').execute()
                # Create discovered_videos table  
                self.db.client.rpc('create_discovered_videos_table_if_not_exists').execute()
                print("[OK] Supabase subscription tables initialized")
            except Exception as e:
                print(f"[INFO] Supabase tables may already exist: {str(e)}")
        else:
            # For SQLite, create tables directly
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Channel subscriptions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS channel_subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT UNIQUE NOT NULL,
                    channel_name TEXT NOT NULL,
                    channel_url TEXT NOT NULL,
                    rss_url TEXT NOT NULL,
                    subscriber_count INTEGER,
                    last_video_date TIMESTAMP,
                    auto_process BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Discovered videos from subscriptions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS discovered_videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_subscription_id INTEGER,
                    video_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    published_date TIMESTAMP,
                    description TEXT,
                    auto_processed BOOLEAN DEFAULT FALSE,
                    processed_summary_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (channel_subscription_id) REFERENCES channel_subscriptions (id),
                    FOREIGN KEY (processed_summary_id) REFERENCES summaries (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            print("[OK] SQLite subscription tables initialized")
    
    def extract_channel_id_from_url(self, url: str) -> Optional[str]:
        """
        Extract YouTube channel ID from various URL formats
        """
        import re
        
        # Direct channel ID URLs: /channel/UCxxxxx
        channel_match = re.search(r'/channel/([a-zA-Z0-9_-]+)', url)
        if channel_match:
            return channel_match.group(1)
        
        # Handle URLs: /@username
        handle_match = re.search(r'/@([a-zA-Z0-9_-]+)', url)
        if handle_match:
            # For handles, we need to resolve to channel ID
            return self._resolve_handle_to_channel_id(handle_match.group(1))
        
        # Custom URLs: /c/username or /user/username
        custom_match = re.search(r'/(?:c|user)/([a-zA-Z0-9_-]+)', url)
        if custom_match:
            # For custom URLs, we need to resolve to channel ID
            return self._resolve_custom_url_to_channel_id(url, custom_match.group(1))
            
        return None
    
    def _resolve_handle_to_channel_id(self, handle: str) -> Optional[str]:
        """
        Resolve YouTube handle (@username) to channel ID
        """
        try:
            # Try to fetch the handle page and extract channel ID
            handle_url = f"https://www.youtube.com/@{handle}"
            response = requests.get(handle_url, timeout=10)
            
            if response.status_code == 200:
                # Look for channel ID in the page source
                import re
                channel_match = re.search(r'"channelId":"([^"]+)"', response.text)
                if channel_match:
                    return channel_match.group(1)
                    
                # Alternative: look for externalId
                external_match = re.search(r'"externalId":"([^"]+)"', response.text)
                if external_match:
                    return external_match.group(1)
                    
        except Exception as e:
            print(f"[WARNING] Could not resolve handle @{handle}: {str(e)}")
            
        return None
    
    def _resolve_custom_url_to_channel_id(self, url: str, username: str) -> Optional[str]:
        """
        Resolve custom YouTube URL to channel ID
        """
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                import re
                # Look for channel ID in the page source
                channel_match = re.search(r'"channelId":"([^"]+)"', response.text)
                if channel_match:
                    return channel_match.group(1)
                    
                # Alternative patterns
                external_match = re.search(r'"externalId":"([^"]+)"', response.text)
                if external_match:
                    return external_match.group(1)
                    
        except Exception as e:
            print(f"[WARNING] Could not resolve custom URL {url}: {str(e)}")
            
        return None
    
    def add_subscription(self, channel_url: str, auto_process: bool = False) -> Dict[str, Any]:
        """
        Add a new channel subscription
        """
        try:
            # Extract channel ID from URL
            channel_id = self.extract_channel_id_from_url(channel_url)
            if not channel_id:
                return {
                    'success': False,
                    'error': 'Could not extract channel ID from URL. Please use a valid YouTube channel URL.'
                }
            
            # Create RSS feed URL
            rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            
            # Fetch RSS feed to get channel info
            feed_data = self._fetch_rss_feed(rss_url)
            if not feed_data['success']:
                return {
                    'success': False,
                    'error': f"Could not fetch RSS feed: {feed_data['error']}"
                }
            
            feed = feed_data['feed']
            channel_name = feed.feed.get('title', 'Unknown Channel')
            
            # Get last video date if available
            last_video_date = None
            if feed.entries:
                try:
                    last_video_date = datetime.fromisoformat(
                        feed.entries[0].published.replace('Z', '+00:00').replace('+00:00', '')
                    )
                except:
                    pass
            
            # Save to database
            subscription_data = {
                'channel_id': channel_id,
                'channel_name': channel_name,
                'channel_url': channel_url,
                'rss_url': rss_url,
                'last_video_date': last_video_date,
                'auto_process': auto_process
            }
            
            if self.use_supabase:
                result = self.db.client.table('channel_subscriptions').insert(subscription_data).execute()
                subscription_id = result.data[0]['id'] if result.data else None
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO channel_subscriptions 
                    (channel_id, channel_name, channel_url, rss_url, last_video_date, auto_process)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (channel_id, channel_name, channel_url, rss_url, 
                     last_video_date.isoformat() if last_video_date else None, auto_process))
                subscription_id = cursor.lastrowid
                conn.commit()
                conn.close()
            
            # Discover initial videos
            self._discover_videos_for_subscription(subscription_id, channel_id, rss_url)
            
            return {
                'success': True,
                'subscription_id': subscription_id,
                'channel_name': channel_name,
                'channel_id': channel_id,
                'videos_discovered': len(feed.entries)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to add subscription: {str(e)}"
            }
    
    def _fetch_rss_feed(self, rss_url: str) -> Dict[str, Any]:
        """
        Fetch and parse RSS feed
        """
        try:
            response = requests.get(rss_url, timeout=15)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                return {
                    'success': False,
                    'error': f"Invalid RSS feed: {feed.bozo_exception}"
                }
            
            return {
                'success': True,
                'feed': feed
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _discover_videos_for_subscription(self, subscription_id: int, channel_id: str, rss_url: str):
        """
        Discover new videos for a subscription
        """
        try:
            # Fetch RSS feed
            feed_data = self._fetch_rss_feed(rss_url)
            if not feed_data['success']:
                print(f"[WARNING] Could not fetch RSS for channel {channel_id}: {feed_data['error']}")
                return
            
            feed = feed_data['feed']
            new_videos = 0
            
            for entry in feed.entries:
                try:
                    # Extract video info
                    video_id = entry.yt_videoid if hasattr(entry, 'yt_videoid') else None
                    if not video_id:
                        # Try to extract from link
                        import re
                        video_match = re.search(r'v=([a-zA-Z0-9_-]+)', entry.link)
                        if video_match:
                            video_id = video_match.group(1)
                    
                    if not video_id:
                        continue
                    
                    # Check if video already exists
                    if self._video_already_discovered(video_id):
                        continue
                    
                    # Parse published date
                    published_date = None
                    try:
                        published_date = datetime.fromisoformat(
                            entry.published.replace('Z', '+00:00').replace('+00:00', '')
                        )
                    except:
                        pass
                    
                    # Save discovered video
                    video_data = {
                        'channel_subscription_id': subscription_id,
                        'video_id': video_id,
                        'title': entry.title,
                        'url': entry.link,
                        'published_date': published_date,
                        'description': getattr(entry, 'summary', '')
                    }
                    
                    if self.use_supabase:
                        self.db.client.table('discovered_videos').insert(video_data).execute()
                    else:
                        conn = sqlite3.connect(self.db.db_path)
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO discovered_videos 
                            (channel_subscription_id, video_id, title, url, published_date, description)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (subscription_id, video_id, entry.title, entry.link,
                             published_date.isoformat() if published_date else None,
                             getattr(entry, 'summary', '')))
                        conn.commit()
                        conn.close()
                    
                    new_videos += 1
                    
                except Exception as e:
                    print(f"[WARNING] Error processing video entry: {str(e)}")
                    continue
            
            print(f"[INFO] Discovered {new_videos} new videos for channel {channel_id}")
            
        except Exception as e:
            print(f"[ERROR] Error discovering videos for subscription {subscription_id}: {str(e)}")
    
    def _video_already_discovered(self, video_id: str) -> bool:
        """
        Check if video is already discovered or processed
        """
        try:
            if self.use_supabase:
                # Check discovered_videos table
                result = self.db.client.table('discovered_videos').select('id').eq('video_id', video_id).execute()
                if result.data:
                    return True
                
                # Check summaries table
                result = self.db.client.table('summaries').select('id').eq('video_id', video_id).execute()
                return len(result.data) > 0
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                # Check discovered_videos table
                cursor.execute('SELECT id FROM discovered_videos WHERE video_id = ?', (video_id,))
                if cursor.fetchone():
                    conn.close()
                    return True
                
                # Check summaries table
                cursor.execute('SELECT id FROM summaries WHERE video_id = ?', (video_id,))
                result = cursor.fetchone()
                conn.close()
                return result is not None
                
        except Exception as e:
            print(f"[WARNING] Error checking if video exists: {str(e)}")
            return False
    
    def get_all_subscriptions(self) -> List[Dict[str, Any]]:
        """
        Get all channel subscriptions
        """
        try:
            if self.use_supabase:
                result = self.db.client.table('channel_subscriptions').select('*').order('created_at', desc=True).execute()
                return result.data
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM channel_subscriptions 
                    ORDER BY created_at DESC
                ''')
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                conn.close()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            print(f"[ERROR] Error getting subscriptions: {str(e)}")
            return []
    
    def get_discovered_videos(self, subscription_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get discovered videos, optionally filtered by subscription
        """
        try:
            if self.use_supabase:
                query = self.db.client.table('discovered_videos').select('''
                    *, channel_subscriptions(channel_name, channel_id)
                ''').order('published_date', desc=True)
                
                if subscription_id:
                    query = query.eq('channel_subscription_id', subscription_id)
                
                result = query.limit(limit).execute()
                return result.data
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                if subscription_id:
                    cursor.execute('''
                        SELECT dv.*, cs.channel_name, cs.channel_id
                        FROM discovered_videos dv
                        JOIN channel_subscriptions cs ON dv.channel_subscription_id = cs.id
                        WHERE dv.channel_subscription_id = ?
                        ORDER BY dv.published_date DESC
                        LIMIT ?
                    ''', (subscription_id, limit))
                else:
                    cursor.execute('''
                        SELECT dv.*, cs.channel_name, cs.channel_id
                        FROM discovered_videos dv
                        JOIN channel_subscriptions cs ON dv.channel_subscription_id = cs.id
                        ORDER BY dv.published_date DESC
                        LIMIT ?
                    ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                conn.close()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            print(f"[ERROR] Error getting discovered videos: {str(e)}")
            return []
    
    def refresh_all_subscriptions(self) -> Dict[str, Any]:
        """
        Refresh all subscriptions to discover new videos
        """
        try:
            subscriptions = self.get_all_subscriptions()
            total_new_videos = 0
            refreshed_channels = 0
            
            for sub in subscriptions:
                try:
                    old_count = len(self.get_discovered_videos(sub['id']))
                    self._discover_videos_for_subscription(sub['id'], sub['channel_id'], sub['rss_url'])
                    new_count = len(self.get_discovered_videos(sub['id']))
                    
                    channel_new_videos = new_count - old_count
                    total_new_videos += max(0, channel_new_videos)
                    refreshed_channels += 1
                    
                except Exception as e:
                    print(f"[WARNING] Error refreshing subscription {sub['id']}: {str(e)}")
                    continue
            
            return {
                'success': True,
                'refreshed_channels': refreshed_channels,
                'new_videos': total_new_videos,
                'total_subscriptions': len(subscriptions)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def delete_subscription(self, subscription_id: int) -> bool:
        """
        Delete a channel subscription and its discovered videos
        """
        try:
            if self.use_supabase:
                # Delete discovered videos first
                self.db.client.table('discovered_videos').delete().eq('channel_subscription_id', subscription_id).execute()
                # Delete subscription
                result = self.db.client.table('channel_subscriptions').delete().eq('id', subscription_id).execute()
                return len(result.data) > 0
            else:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                # Delete discovered videos first
                cursor.execute('DELETE FROM discovered_videos WHERE channel_subscription_id = ?', (subscription_id,))
                # Delete subscription
                cursor.execute('DELETE FROM channel_subscriptions WHERE id = ?', (subscription_id,))
                
                success = cursor.rowcount > 0
                conn.commit()
                conn.close()
                return success
                
        except Exception as e:
            print(f"[ERROR] Error deleting subscription: {str(e)}")
            return False