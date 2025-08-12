"""
Data Access Layer (DAL)
All database queries centralized here - single source of truth for DB access
"""
from typing import List, Dict, Any, Optional
from .client import supabase_client

class DatabaseQueries:
    """Centralized database query layer"""
    
    def __init__(self):
        self.client = supabase_client.client
    
    def is_available(self) -> bool:
        """Check if database is available"""
        return supabase_client.is_available()
    
    # ========== VECTOR SEARCH QUERIES ==========
    
    def find_similar_summaries(self, summary_id: int, match_count: int = 5) -> List[Dict[str, Any]]:
        """
        Find summaries similar to a specific summary using RPC
        
        Args:
            summary_id: ID of the reference summary
            match_count: Maximum number of results to return
            
        Returns:
            List of similar summaries with similarity scores
        """
        if not self.client:
            return []
        
        try:
            payload = {
                "summary_id": int(summary_id), 
                "match_count": int(match_count)
            }
            result = self.client.rpc("find_similar_summaries", payload).execute()
            return result.data or []
        except Exception as e:
            print(f"[ERROR] find_similar_summaries failed: {e}")
            return []
    
    def search_by_embedding(self, embedding: List[float], match_threshold: float = 0.75, 
                          match_count: int = 10) -> List[Dict[str, Any]]:
        """
        Search summaries by embedding vector using RPC
        
        Args:
            embedding: Vector embedding (1536 floats)
            match_threshold: Similarity threshold (0.0-1.0)
            match_count: Maximum number of results
            
        Returns:
            List of matching summaries with similarity scores
        """
        if not self.client:
            return []
        
        try:
            payload = {
                "query_embedding": embedding,
                "match_threshold": float(match_threshold),
                "match_count": int(match_count),
            }
            result = self.client.rpc("search_summaries_by_similarity", payload).execute()
            return result.data or []
        except Exception as e:
            print(f"[ERROR] search_by_embedding failed: {e}")
            return []
    
    # ========== SUMMARY CRUD OPERATIONS ==========
    
    def get_summary(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """Get a single summary by ID"""
        if not self.client:
            return None
        
        try:
            result = self.client.table("summaries").select("*").eq("id", summary_id).single().execute()
            return result.data
        except Exception as e:
            print(f"[ERROR] get_summary failed: {e}")
            return None
    
    def get_all_summaries(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all summaries with pagination"""
        if not self.client:
            return []
        
        try:
            result = self.client.table("summaries").select("*").order("created_at", desc=True).limit(limit).offset(offset).execute()
            return result.data or []
        except Exception as e:
            print(f"[ERROR] get_all_summaries failed: {e}")
            return []
    
    def insert_summary(self, summary_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert a new summary"""
        if not self.client:
            return None
        
        try:
            result = self.client.table("summaries").insert(summary_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[ERROR] insert_summary failed: {e}")
            return None
    
    def update_summary(self, summary_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing summary"""
        if not self.client:
            return None
        
        try:
            result = self.client.table("summaries").update(updates).eq("id", summary_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[ERROR] update_summary failed: {e}")
            return None
    
    def delete_summary(self, summary_id: int) -> bool:
        """Delete a summary"""
        if not self.client:
            return False
        
        try:
            result = self.client.table("summaries").delete().eq("id", summary_id).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"[ERROR] delete_summary failed: {e}")
            return False
    
    # ========== SEARCH OPERATIONS ==========
    
    def search_summaries(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search summaries by text query"""
        if not self.client:
            return []
        
        try:
            # Simple text search in title and summary fields
            result = self.client.table("summaries").select("*").or_(
                f"title.ilike.%{query}%,summary.ilike.%{query}%"
            ).order("created_at", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            print(f"[ERROR] search_summaries failed: {e}")
            return []
    
    # ========== STATISTICS & ANALYTICS ==========
    
    def get_summary_count(self) -> int:
        """Get total number of summaries"""
        if not self.client:
            return 0
        
        try:
            result = self.client.table("summaries").select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            print(f"[ERROR] get_summary_count failed: {e}")
            return 0
    
    def get_summaries_by_type(self, summary_type: str) -> List[Dict[str, Any]]:
        """Get summaries filtered by type"""
        if not self.client:
            return []
        
        try:
            result = self.client.table("summaries").select("*").eq("summary_type", summary_type).order("created_at", desc=True).execute()
            return result.data or []
        except Exception as e:
            print(f"[ERROR] get_summaries_by_type failed: {e}")
            return []
    
    # ========== HEALTH CHECK ==========
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        if not self.client:
            return {"status": "error", "message": "Client not available"}
        
        try:
            # Test basic connectivity
            result = self.client.table("summaries").select("id").limit(1).execute()
            
            # Test vector functions
            dummy_vector = [0.0] * 1536
            self.client.rpc("search_summaries_by_similarity", {
                "query_embedding": dummy_vector,
                "match_threshold": 0.0,
                "match_count": 1
            }).execute()
            
            return {
                "status": "ok", 
                "message": "Database and vector functions available",
                "summary_count": self.get_summary_count()
            }
        except Exception as e:
            return {"status": "error", "message": f"Health check failed: {e}"}

# Global DAL instance
dal = DatabaseQueries()