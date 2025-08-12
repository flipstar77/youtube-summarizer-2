"""
Vector Embeddings Service for YouTube Summarizer
Handles text-to-vector conversion and similarity search using Supabase pgvector
"""

import os
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import json

class VectorEmbeddingService:
    def __init__(self, embedding_method='sentence_transformer'):
        """
        Initialize the vector embedding service
        
        Args:
            embedding_method: 'openai' or 'sentence_transformer'
        """
        self.embedding_method = embedding_method
        self.embedding_dim = 384  # Dimension for sentence transformers
        
        if embedding_method == 'openai':
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.embedding_dim = 1536  # OpenAI ada-002 dimension
            self.client = openai.OpenAI(api_key=openai.api_key)
        else:
            # Use sentence transformers (free, local processing)
            print("[INFO] Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] Sentence transformer model loaded successfully")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for given text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the vector embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean and prepare text
        cleaned_text = self._clean_text(text)
        
        if self.embedding_method == 'openai':
            return self._generate_openai_embedding(cleaned_text)
        else:
            return self._generate_sentence_transformer_embedding(cleaned_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (keep first 2000 characters for embedding)
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text
    
    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI embedding generation failed: {str(e)}")
    
    def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Sentence transformer embedding generation failed: {str(e)}")
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1, higher = more similar)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            raise Exception(f"Similarity calculation failed: {str(e)}")
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"[WARNING] Failed to generate embedding for text: {str(e)}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.embedding_dim)
        
        return embeddings
    
    def create_search_embedding(self, query: str) -> List[float]:
        """
        Create embedding specifically optimized for search queries
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector optimized for search
        """
        # For search queries, we might want to add context or modify the text
        search_text = f"Search query: {query}"
        return self.generate_embedding(search_text)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration"""
        return {
            'method': self.embedding_method,
            'dimension': self.embedding_dim,
            'model': 'text-embedding-ada-002' if self.embedding_method == 'openai' else 'all-MiniLM-L6-v2'
        }

class SummaryVectorizer:
    """Handles vectorization of summaries specifically"""
    
    def __init__(self, embedding_service: VectorEmbeddingService = None):
        if embedding_service is None:
            # Default to OpenAI for 1536-dimensional embeddings (matches Supabase schema)
            self.embedding_service = VectorEmbeddingService(embedding_method='openai')
        else:
            self.embedding_service = embedding_service
        
        # Initialize Supabase client for vector operations
        self._init_supabase_client()
    
    def vectorize_summary(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create vector embedding for a summary
        
        Args:
            summary_data: Dictionary containing summary information
            
        Returns:
            Dictionary with embedding added
        """
        # Combine title and summary for better embedding
        title = summary_data.get('title', '')
        summary = summary_data.get('summary', '')
        summary_type = summary_data.get('summary_type', '')
        
        # Create comprehensive text for embedding
        text_parts = []
        if title:
            text_parts.append(f"Title: {title}")
        if summary_type:
            text_parts.append(f"Type: {summary_type}")
        if summary:
            text_parts.append(f"Content: {summary}")
        
        combined_text = " | ".join(text_parts)
        
        try:
            embedding = self.embedding_service.generate_embedding(combined_text)
            return {
                **summary_data,
                'embedding': embedding,
                'embedding_text': combined_text[:200] + "..." if len(combined_text) > 200 else combined_text
            }
        except Exception as e:
            print(f"[ERROR] Failed to vectorize summary: {str(e)}")
            return summary_data
    
    def batch_vectorize_summaries(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Vectorize multiple summaries efficiently
        
        Args:
            summaries: List of summary dictionaries
            
        Returns:
            List of summaries with embeddings added
        """
        vectorized_summaries = []
        
        for summary in summaries:
            vectorized_summary = self.vectorize_summary(summary)
            vectorized_summaries.append(vectorized_summary)
        
        return vectorized_summaries
    
    def _init_supabase_client(self):
        """Initialize Supabase client for vector operations"""
        try:
            from supabase import create_client
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            
            if url and key:
                self.supabase = create_client(url, key)
                self.use_supabase = True
                print("[OK] Vector search using Supabase")
            else:
                self.supabase = None
                self.use_supabase = False
                print("[INFO] Supabase not configured, vector search disabled")
        except ImportError:
            self.supabase = None
            self.use_supabase = False
            print("[WARNING] Supabase client not available")
    
    def search_similar_summaries(self, query_text: str, match_threshold: float = 0.75, match_count: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar summaries using vector similarity
        
        Args:
            query_text: Text to search for
            match_threshold: Similarity threshold (0.0-1.0, higher = more similar)
            match_count: Maximum number of results to return
            
        Returns:
            List of similar summaries with similarity scores
        """
        if not self.use_supabase or not self.supabase:
            print("[WARNING] Vector search not available - Supabase not configured")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.generate_embedding(query_text)
            
            # Use the new RPC function
            result = self.supabase.rpc("search_summaries_by_similarity", {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count
            }).execute()
            
            if result.data:
                print(f"[INFO] Found {len(result.data)} similar summaries")
                return result.data
            else:
                print("[INFO] No similar summaries found")
                return []
                
        except Exception as e:
            print(f"[ERROR] Vector search failed: {str(e)}")
            return []
    
    def find_similar_to_summary(self, summary_id: int, match_count: int = 5) -> List[Dict[str, Any]]:
        """
        Find summaries similar to a specific summary
        
        Args:
            summary_id: ID of the reference summary
            match_count: Maximum number of results to return
            
        Returns:
            List of similar summaries with similarity scores
        """
        if not self.use_supabase or not self.supabase:
            print("[WARNING] Vector search not available - Supabase not configured")
            return []
        
        try:
            # Use the new RPC function
            result = self.supabase.rpc("find_similar_summaries", {
                "summary_id": summary_id,
                "match_count": match_count
            }).execute()
            
            if result.data:
                print(f"[INFO] Found {len(result.data)} similar summaries to ID {summary_id}")
                return result.data
            else:
                print(f"[INFO] No similar summaries found for ID {summary_id}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Similar summaries search failed: {str(e)}")
            return []

# Factory function for easy initialization
def create_embedding_service(use_openai: bool = False) -> VectorEmbeddingService:
    """
    Create and return an embedding service instance
    
    Args:
        use_openai: Whether to use OpenAI embeddings (requires API key)
        
    Returns:
        VectorEmbeddingService instance
    """
    method = 'openai' if use_openai else 'sentence_transformer'
    return VectorEmbeddingService(embedding_method=method)