"""
Supabase Database Client
Centralized database connection and client management
"""
import os
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    """Singleton Supabase client manager"""
    
    _instance: Optional['SupabaseClient'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._init_client()
    
    def _init_client(self):
        """Initialize Supabase client"""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not url or not key:
            print("[WARNING] Supabase credentials not found in environment")
            self._client = None
            return
        
        try:
            self._client = create_client(url, key)
            print("[OK] Supabase client initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Supabase client: {e}")
            self._client = None
    
    @property
    def client(self) -> Optional[Client]:
        """Get the Supabase client instance"""
        return self._client
    
    def is_available(self) -> bool:
        """Check if Supabase client is available"""
        return self._client is not None

# Global instance
supabase_client = SupabaseClient()