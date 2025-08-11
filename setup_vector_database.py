#!/usr/bin/env python3
"""
Complete Vector Database Setup Script
Generates embeddings for existing summaries after SQL schema is applied
"""

import os
from dotenv import load_dotenv
from supabase_client import SupabaseDatabase
from vector_embeddings import create_embedding_service, SummaryVectorizer

load_dotenv()

def test_schema_applied():
    """Test if the vector schema has been applied to Supabase"""
    print("Testing if vector schema is applied...")
    
    try:
        db = SupabaseDatabase()
        
        # Test if embedding column exists by trying to get summaries without embeddings
        summaries = db.get_summaries_without_embeddings()
        print(f"[OK] Vector schema detected - found {len(summaries)} summaries needing embeddings")
        return True
        
    except Exception as e:
        print(f"[ERROR] Vector schema not applied: {str(e)}")
        print("\nTo fix this, please:")
        print("1. Go to https://app.supabase.com/project/[your-project]/sql")
        print("2. Copy and paste the contents of 'manual_vector_setup.sql'")
        print("3. Run each SQL command one by one")
        print("4. Then run this script again")
        return False

def generate_embeddings_for_all():
    """Generate embeddings for all summaries that don't have them"""
    print("\nGenerating embeddings for summaries...")
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("[ERROR] OPENAI_API_KEY not found in environment")
        return False
    
    try:
        db = SupabaseDatabase()
        
        # Initialize embedding services
        embedding_service = create_embedding_service(use_openai=True)
        vectorizer = SummaryVectorizer(embedding_service)
        
        # Get summaries that need embeddings
        summaries_to_process = db.get_summaries_without_embeddings()
        
        if not summaries_to_process:
            print("[OK] All summaries already have embeddings!")
            return True
        
        print(f"Processing {len(summaries_to_process)} summaries...")
        
        success_count = 0
        for i, summary in enumerate(summaries_to_process, 1):
            try:
                title = summary.get('title', 'No title')
                if len(title) > 50:
                    title = title[:50] + "..."
                
                print(f"  [{i}/{len(summaries_to_process)}] Processing: {title}")
                
                # Generate embedding
                vectorized_data = vectorizer.vectorize_summary(summary)
                embedding = vectorized_data.get('embedding')
                
                if embedding and len(embedding) == 1536:
                    # Update database with the embedding
                    db.update_summary_embedding(summary['id'], embedding)
                    success_count += 1
                    print(f"    [OK] Added 1536-dimensional embedding")
                else:
                    print(f"    [ERROR] Failed to generate valid embedding")
                    
            except Exception as e:
                print(f"    [ERROR] {str(e)}")
                continue
        
        print(f"\nCompleted! {success_count}/{len(summaries_to_process)} summaries now have embeddings")
        
        # Test vector search
        if success_count > 0:
            print("\nTesting vector search functionality...")
            try:
                # Try to search for AI-related content
                test_query = "artificial intelligence machine learning"
                test_embedding = embedding_service.generate_embedding(test_query)
                results = db.search_similar_summaries(test_embedding, threshold=0.1, limit=3)
                print(f"[OK] Vector search working - found {len(results)} similar summaries")
                
                for result in results[:2]:
                    similarity = result.get('similarity', 0)
                    title = result.get('title', 'No title')[:60]
                    print(f"  - {title}... (similarity: {similarity:.3f})")
                    
            except Exception as e:
                print(f"[ERROR] Vector search test failed: {str(e)}")
        
        return success_count > 0
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return False

def main():
    print("Vector Database Setup Tool")
    print("=" * 40)
    
    # Step 1: Check if schema is applied
    if not test_schema_applied():
        return
    
    # Step 2: Generate embeddings
    if generate_embeddings_for_all():
        print("\n[SUCCESS] Vector database is fully set up!")
        print("\nYour YouTube summarizer now supports:")
        print("- Semantic search for similar content")
        print("- Better categorization based on content similarity")
        print("- Improved content recommendations")
        
        # Show database stats
        try:
            db = SupabaseDatabase()
            all_summaries = db.get_all_summaries()
            remaining = db.get_summaries_without_embeddings()
            
            total = len(all_summaries)
            with_embeddings = total - len(remaining)
            
            print(f"\nDatabase Status:")
            print(f"   Total summaries: {total}")
            print(f"   With embeddings: {with_embeddings}")
            print(f"   Without embeddings: {len(remaining)}")
            
        except Exception as e:
            print(f"\n[WARNING] Could not get database stats: {str(e)}")
    else:
        print("\n[ERROR] Some issues occurred. Please check the errors above.")

if __name__ == "__main__":
    main()