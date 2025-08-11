#!/usr/bin/env python3
"""
Simple script to fix vector database issues
"""

import os
from supabase_client import SupabaseDatabase
from dotenv import load_dotenv

load_dotenv()

def test_database_schema():
    """Test current database schema and identify issues"""
    print("Testing current database schema...")
    
    try:
        db = SupabaseDatabase()
        
        # Test basic connection
        print("1. Testing basic connection...")
        summaries = db.get_all_summaries()
        print(f"   - Found {len(summaries)} summaries in database")
        
        # Test embedding column
        print("2. Testing embedding column...")
        try:
            no_embeddings = db.get_summaries_without_embeddings()
            print(f"   - Embedding column exists")
            print(f"   - {len(no_embeddings)} summaries need embeddings")
        except Exception as e:
            print(f"   - ERROR: Embedding column issue: {str(e)}")
            return False
        
        # Test vector functions
        print("3. Testing vector search functions...")
        try:
            dummy_embedding = [0.0] * 384
            result = db.search_similar_summaries(dummy_embedding, threshold=0.1, limit=1)
            print(f"   - Vector search functions working")
        except Exception as e:
            print(f"   - ERROR: Vector functions issue: {str(e)}")
            return False
        
        print("All tests passed! Database schema is correct.")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def add_embeddings_to_existing_summaries():
    """Add embeddings to summaries that don't have them"""
    print("\nAdding embeddings to existing summaries...")
    
    try:
        from supabase_client import SupabaseDatabase
        from vector_embeddings import create_embedding_service, SummaryVectorizer
        
        db = SupabaseDatabase()
        
        # Check if embedding service is available
        if not os.getenv('OPENAI_API_KEY'):
            print("ERROR: OPENAI_API_KEY not found. Cannot generate embeddings.")
            return False
        
        # Initialize services
        embedding_service = create_embedding_service(use_openai=True)
        vectorizer = SummaryVectorizer(embedding_service)
        
        # Get summaries without embeddings
        summaries_to_process = db.get_summaries_without_embeddings()
        
        if not summaries_to_process:
            print("All summaries already have embeddings!")
            return True
        
        print(f"Found {len(summaries_to_process)} summaries to process")
        
        success_count = 0
        for i, summary in enumerate(summaries_to_process, 1):
            try:
                title = summary.get('title', 'No title')[:50]
                print(f"Processing {i}/{len(summaries_to_process)}: {title}...")
                
                # Generate embedding
                vectorized_data = vectorizer.vectorize_summary(summary)
                embedding = vectorized_data.get('embedding')
                
                if embedding:
                    # Update database
                    db.update_summary_embedding(summary['id'], embedding)
                    success_count += 1
                    print(f"  - Successfully added embedding to summary {summary['id']}")
                else:
                    print(f"  - Failed to generate embedding for summary {summary['id']}")
                    
            except Exception as e:
                print(f"  - ERROR processing summary {summary['id']}: {str(e)}")
                continue
        
        print(f"\nCompleted: {success_count}/{len(summaries_to_process)} summaries now have embeddings")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("Vector Database Fix Tool")
    print("=" * 30)
    
    # Step 1: Test current database
    if not test_database_schema():
        print("\nERROR: Database schema is not set up correctly.")
        print("Please run the SQL commands from 'manual_vector_setup.sql' in your Supabase dashboard.")
        exit(1)
    
    # Step 2: Add embeddings
    if add_embeddings_to_existing_summaries():
        print("\nSUCCESS: Vector database is now ready!")
    else:
        print("\nWARNING: Some issues occurred while adding embeddings.")