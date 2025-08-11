#!/usr/bin/env python3
"""
Test vector search functionality after fixing SQL functions
"""

import os
from dotenv import load_dotenv
from vector_embeddings import create_embedding_service
from supabase_client import SupabaseDatabase

load_dotenv()

def test_vector_search():
    """Test the fixed vector search functionality"""
    print("Testing Vector Search Fix")
    print("=" * 30)
    
    try:
        # Initialize services
        embedding_service = create_embedding_service(use_openai=True)
        db = SupabaseDatabase()
        
        # Test query
        test_query = "artificial intelligence machine learning"
        print(f"1. Testing search query: '{test_query}'")
        
        # Generate embedding
        query_embedding = embedding_service.create_search_embedding(test_query)
        print(f"   - Generated embedding with {len(query_embedding)} dimensions")
        
        # Test vector search
        print("2. Testing vector search...")
        results = db.search_similar_summaries(
            query_embedding=query_embedding,
            threshold=0.1,  # Low threshold for testing
            limit=3
        )
        
        if results:
            print(f"   [OK] Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')[:50]
                similarity = result.get('similarity', 0)
                print(f"   {i}. {title}... (similarity: {similarity:.3f})")
        else:
            print("   [WARNING] No results found - may need to check threshold")
        
        print("\n3. Testing similar summaries function...")
        # Get first summary ID to test similar function
        all_summaries = db.get_all_summaries()
        if all_summaries:
            first_id = all_summaries[0]['id']
            similar = db.find_similar_summaries(first_id, limit=2)
            print(f"   [OK] Similar to summary {first_id}: {len(similar)} results")
        
        print("\n[SUCCESS] Vector search is working correctly!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Vector search failed: {str(e)}")
        if "does not match function result type" in str(e):
            print("\nFIX NEEDED: Please run the SQL commands from 'fix_vector_functions.sql'")
            print("This will fix the bigint vs int type mismatch.")
        return False

if __name__ == "__main__":
    test_vector_search()