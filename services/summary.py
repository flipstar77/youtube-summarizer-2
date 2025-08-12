"""
Summary Service
Business logic for summary management and operations
"""
from typing import List, Dict, Any, Optional
from db.queries import dal

class SummaryService:
    """Service for summary-related business operations"""
    
    def __init__(self):
        self.dal = dal
    
    def get_summary(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """Get a single summary by ID"""
        return self.dal.get_summary(summary_id)
    
    def get_all_summaries(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all summaries with pagination"""
        return self.dal.get_all_summaries(limit, offset)
    
    def create_summary(self, summary_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new summary with validation"""
        # Basic validation
        required_fields = ['title', 'summary', 'url', 'video_id']
        for field in required_fields:
            if field not in summary_data:
                raise ValueError(f"Required field '{field}' missing")
        
        # Set defaults
        if 'summary_type' not in summary_data:
            summary_data['summary_type'] = 'standard'
        
        return self.dal.insert_summary(summary_data)
    
    def update_summary(self, summary_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing summary"""
        # Remove read-only fields
        readonly_fields = ['id', 'created_at']
        for field in readonly_fields:
            updates.pop(field, None)
        
        return self.dal.update_summary(summary_id, updates)
    
    def delete_summary(self, summary_id: int) -> bool:
        """Delete a summary"""
        return self.dal.delete_summary(summary_id)
    
    def search_summaries(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search summaries by text"""
        return self.dal.search_summaries(query, limit)
    
    def get_summaries_by_type(self, summary_type: str) -> List[Dict[str, Any]]:
        """Get summaries filtered by type"""
        return self.dal.get_summaries_by_type(summary_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_summaries": self.dal.get_summary_count(),
            "database_status": "connected" if self.dal.is_available() else "disconnected"
        }

# Global service instance
summary_service = SummaryService()