import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class SupabaseDatabase:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_SERVICE_ROLE_KEY', os.getenv('SUPABASE_ANON_KEY'))
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key are required. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) in your .env file.")
        
        self.client: Client = create_client(self.url, self.key)
    
    def save_summary(self, video_id: str, url: str, title: str, summary_type: str, 
                    summary: str, transcript_length: int, audio_file: Optional[str] = None, 
                    voice_id: Optional[str] = None, uploader: Optional[str] = None, 
                    duration: Optional[int] = None) -> int:
        """Save a summary to Supabase"""
        try:
            data = {
                'video_id': video_id,
                'url': url,
                'title': title,
                'summary_type': summary_type,
                'summary': summary,
                'transcript_length': transcript_length,
                'audio_file': audio_file,
                'voice_id': voice_id
            }
            
            # Skip new columns if they don't exist in schema (graceful handling)
            # TODO: Add uploader and duration when database schema is updated
            
            result = self.client.table('summaries').insert(data).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]['id']
            else:
                raise Exception("Failed to save summary - no data returned")
                
        except Exception as e:
            raise Exception(f"Error saving summary to Supabase: {str(e)}")
    
    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get all summaries from Supabase"""
        try:
            result = self.client.table('summaries').select('*').order('created_at', desc=True).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error retrieving summaries from Supabase: {str(e)}")
    
    def get_summary(self, summary_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific summary by ID"""
        try:
            result = self.client.table('summaries').select('*').eq('id', summary_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Error retrieving summary from Supabase: {str(e)}")
    
    def delete_summary(self, summary_id: int) -> bool:
        """Delete a summary by ID"""
        try:
            result = self.client.table('summaries').delete().eq('id', summary_id).execute()
            return len(result.data) > 0
        except Exception as e:
            raise Exception(f"Error deleting summary from Supabase: {str(e)}")
    
    def update_audio_file(self, summary_id: int, audio_file: str, voice_id: str) -> bool:
        """Update a summary with audio file information"""
        try:
            result = self.client.table('summaries').update({
                'audio_file': audio_file,
                'voice_id': voice_id
            }).eq('id', summary_id).execute()
            return len(result.data) > 0
        except Exception as e:
            raise Exception(f"Error updating audio file in Supabase: {str(e)}")
    
    def get_summaries_paginated(self, page_size: int = 20, page_offset: int = 0, 
                               summary_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get summaries with pagination"""
        try:
            query = self.client.table('summaries').select('*')
            
            if summary_type_filter:
                query = query.eq('summary_type', summary_type_filter)
            
            result = query.order('created_at', desc=True).range(page_offset, page_offset + page_size - 1).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error retrieving paginated summaries: {str(e)}")
    
    def search_summaries(self, search_query: str) -> List[Dict[str, Any]]:
        """Search summaries by text content"""
        try:
            # Simple text search - for full-text search, you'd need to set up PostgreSQL full-text search
            result = self.client.table('summaries').select('*').or_(
                f'title.ilike.%{search_query}%,summary.ilike.%{search_query}%'
            ).order('created_at', desc=True).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error searching summaries: {str(e)}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        try:
            # Get total count
            total_result = self.client.table('summaries').select('id', count='exact').execute()
            total_count = total_result.count or 0
            
            # Get summaries with audio
            audio_result = self.client.table('summaries').select('id', count='exact').not_.is_('audio_file', 'null').execute()
            audio_count = audio_result.count or 0
            
            # Get recent summaries (last 7 days)
            from datetime import datetime, timedelta
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            recent_result = self.client.table('summaries').select('id', count='exact').gte('created_at', week_ago).execute()
            recent_count = recent_result.count or 0
            
            # Get by type
            type_stats = {}
            for summary_type in ['brief', 'detailed', 'bullet']:
                type_result = self.client.table('summaries').select('id', count='exact').eq('summary_type', summary_type).execute()
                type_stats[summary_type] = type_result.count or 0
            
            return {
                'total_summaries': total_count,
                'summaries_with_audio': audio_count,
                'recent_summaries': recent_count,
                'by_type': type_stats
            }
        except Exception as e:
            raise Exception(f"Error retrieving summary statistics: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test the Supabase connection"""
        try:
            # Try to perform a simple query
            result = self.client.table('summaries').select('id').limit(1).execute()
            return True
        except Exception as e:
            print(f"Supabase connection test failed: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """Create the necessary tables (this would typically be done via migrations)"""
        try:
            # In a real scenario, you'd run the SQL schema through Supabase migrations
            # This is just for reference - the actual table creation should be done in Supabase dashboard
            print("Tables should be created via Supabase migrations or SQL editor.")
            print("Run the contents of supabase_schema.sql in your Supabase SQL editor.")
            return True
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
            return False
    
    # Vector Embedding Methods
    def save_summary_with_vector(self, video_id: str, url: str, title: str, 
                                summary_type: str, summary: str, transcript_length: int, 
                                embedding: list = None, audio_file: str = None, voice_id: str = None,
                                uploader: str = None, duration: int = None,
                                ai_provider: str = None, ai_model: str = None):
        """Save summary with vector embedding"""
        try:
            summary_data = {
                'video_id': video_id,
                'url': url,
                'title': title,
                'summary_type': summary_type,
                'summary': summary,
                'transcript_length': transcript_length,
                'audio_file': audio_file,
                'voice_id': voice_id,
                'created_at': datetime.now().isoformat()
            }
            
            # Add AI provider and model if provided
            if ai_provider:
                summary_data['ai_provider'] = ai_provider
            if ai_model:
                summary_data['ai_model'] = ai_model
            
            # Skip new columns if they don't exist in schema (graceful handling)  
            # TODO: Add uploader and duration when database schema is updated
            
            if embedding:
                summary_data['embedding'] = embedding
            
            result = self.client.table('summaries').insert(summary_data).execute()
            return result.data[0]['id']
        except Exception as e:
            raise Exception(f"Error saving summary with vector: {str(e)}")
    
    def update_summary_embedding(self, summary_id: int, embedding: list):
        """Update summary with vector embedding"""
        try:
            result = self.client.table('summaries').update({
                'embedding': embedding
            }).eq('id', summary_id).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error updating summary embedding: {str(e)}")
    
    def get_summaries_without_embeddings(self):
        """Get summaries that don't have vector embeddings yet"""
        try:
            result = self.client.table('summaries').select('*').is_('embedding', 'null').execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error getting summaries without embeddings: {str(e)}")
    
    def search_similar_summaries(self, query_embedding: list, threshold: float = 0.7, limit: int = 5):
        """Search for similar summaries using vector similarity"""
        try:
            # Use the Supabase RPC function for vector similarity search
            result = self.client.rpc('search_summaries_by_similarity', {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error searching similar summaries: {str(e)}")
    
    def find_similar_summaries(self, summary_id: int, limit: int = 5):
        """Find summaries similar to a given summary"""
        try:
            result = self.client.rpc('find_similar_summaries', {
                'summary_id': summary_id,
                'match_count': limit
            }).execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error finding similar summaries: {str(e)}")
    
    def get_all_embeddings(self):
        """Get all summaries with their embeddings"""
        try:
            result = self.client.table('summaries').select('id, embedding, title, summary').not_.is_('embedding', 'null').execute()
            return result.data
        except Exception as e:
            raise Exception(f"Error getting all embeddings: {str(e)}")