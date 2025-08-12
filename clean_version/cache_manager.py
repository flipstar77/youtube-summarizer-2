#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis Cache Manager
Provides caching for expensive operations like embeddings and AI responses
"""

import os
import json
import hashlib
import redis
from typing import Optional, Any, Dict, List
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class CacheManager:
    """Redis-based cache manager with fallback to memory cache"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.use_redis = self._init_redis()
        
        if self.use_redis:
            print("[OK] Redis cache initialized")
        else:
            print("[INFO] Using memory cache (Redis not available)")
    
    def _init_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            self.redis_client.ping()
            return True
            
        except Exception as e:
            print(f"[WARNING] Redis initialization failed: {e}")
            return False
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        hash_obj = hashlib.md5(data_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.use_redis:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            print(f"[WARNING] Cache get failed: {e}")
        
        return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration"""
        try:
            if self.use_redis:
                return self.redis_client.setex(
                    key, 
                    expire, 
                    json.dumps(value, default=str)
                )
            else:
                self.memory_cache[key] = value
                return True
                
        except Exception as e:
            print(f"[WARNING] Cache set failed: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.use_redis:
                return bool(self.redis_client.delete(key))
            else:
                return self.memory_cache.pop(key, None) is not None
        except Exception as e:
            print(f"[WARNING] Cache delete failed: {e}")
            return False
    
    def flush_all(self) -> bool:
        """Clear all cache"""
        try:
            if self.use_redis:
                return self.redis_client.flushdb()
            else:
                self.memory_cache.clear()
                return True
        except Exception as e:
            print(f"[WARNING] Cache flush failed: {e}")
            return False
    
    # Specialized cache methods for the application
    
    def cache_embedding(self, text: str, embedding: List[float], expire: int = 86400) -> bool:
        """Cache text embedding (24 hour default)"""
        key = self._generate_key("embedding", text)
        return self.set(key, embedding, expire)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._generate_key("embedding", text)
        return self.get(key)
    
    def cache_ai_response(self, prompt: str, response: str, provider: str, expire: int = 3600) -> bool:
        """Cache AI response (1 hour default)"""
        key = self._generate_key(f"ai_{provider}", prompt)
        return self.set(key, response, expire)
    
    def get_ai_response(self, prompt: str, provider: str) -> Optional[str]:
        """Get cached AI response"""
        key = self._generate_key(f"ai_{provider}", prompt)
        return self.get(key)
    
    def cache_video_metadata(self, video_id: str, metadata: Dict[str, Any], expire: int = 43200) -> bool:
        """Cache video metadata (12 hour default)"""
        key = f"video_meta:{video_id}"
        return self.set(key, metadata, expire)
    
    def get_video_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get cached video metadata"""
        key = f"video_meta:{video_id}"
        return self.get(key)
    
    def cache_transcript(self, video_id: str, transcript: str, expire: int = 604800) -> bool:
        """Cache transcript (7 days default)"""
        key = f"transcript:{video_id}"
        return self.set(key, transcript, expire)
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get cached transcript"""
        key = f"transcript:{video_id}"
        return self.get(key)
    
    def cache_search_results(self, query: str, results: List[Dict], expire: int = 1800) -> bool:
        """Cache search results (30 minutes default)"""
        key = self._generate_key("search", query)
        return self.set(key, results, expire)
    
    def get_search_results(self, query: str) -> Optional[List[Dict]]:
        """Get cached search results"""
        key = self._generate_key("search", query)
        return self.get(key)

# Global cache instance
cache = CacheManager()