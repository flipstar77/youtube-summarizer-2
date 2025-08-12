"""
Vector Search Service
Business logic for semantic search and similarity operations
"""
from typing import List, Dict, Any, Optional
from db.queries import dal
import openai
import os

class VectorSearchService:
    """Service for vector-based search operations"""
    
    def __init__(self):
        self.dal = dal
        # Initialize OpenAI for embeddings
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if openai.api_key:
            self.client = openai.OpenAI(api_key=openai.api_key)
        else:
            self.client = None
            print("[WARNING] OpenAI API key not found - embedding generation disabled")
    
    def find_similar(self, summary_id: int, count: int = 5) -> List[Dict[str, Any]]:
        """
        Find summaries similar to a specific summary
        
        Args:
            summary_id: ID of the reference summary
            count: Number of similar summaries to return
            
        Returns:
            List of similar summaries with similarity scores
        """
        return self.dal.find_similar_summaries(summary_id, count)
    
    def search_by_text(self, query: str, threshold: float = 0.75, count: int = 10) -> List[Dict[str, Any]]:
        """
        Search summaries by text query using vector embeddings
        
        Args:
            query: Text query to search for
            threshold: Similarity threshold (0.0-1.0)
            count: Maximum number of results
            
        Returns:
            List of matching summaries with similarity scores
        """
        if not self.client:
            print("[ERROR] OpenAI client not available for embedding generation")
            return []
        
        try:
            # Generate embedding for query
            embedding = self._generate_embedding(query)
            if not embedding:
                return []
            
            # Search using embedding
            return self.dal.search_by_embedding(embedding, threshold, count)
            
        except Exception as e:
            print(f"[ERROR] Vector search failed: {e}")
            return []
    
    def search_by_embedding(self, embedding: List[float], threshold: float = 0.75, 
                          count: int = 10) -> List[Dict[str, Any]]:
        """
        Search summaries by pre-computed embedding
        
        Args:
            embedding: Vector embedding (1536 floats)
            threshold: Similarity threshold (0.0-1.0) 
            count: Maximum number of results
            
        Returns:
            List of matching summaries with similarity scores
        """
        return self.dal.search_by_embedding(embedding, threshold, count)
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate OpenAI embedding for text"""
        if not self.client:
            return None
        
        try:
            # Clean text
            cleaned_text = ' '.join(text.split())
            if len(cleaned_text) > 2000:
                cleaned_text = cleaned_text[:2000] + "..."
            
            response = self.client.embeddings.create(
                input=cleaned_text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
            
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if vector search is available"""
        return self.dal.is_available() and self.client is not None

# Global service instance
vector_search_service = VectorSearchService()