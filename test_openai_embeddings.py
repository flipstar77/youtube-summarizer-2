#!/usr/bin/env python3
"""
Test OpenAI embeddings functionality
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_openai_embeddings():
    """Test OpenAI embedding generation"""
    print("Testing OpenAI Embeddings")
    print("=" * 25)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment")
        return False
    
    print(f"API Key found: {api_key[:12]}...")
    
    try:
        from vector_embeddings import create_embedding_service, SummaryVectorizer
        
        print("1. Creating OpenAI embedding service...")
        embedding_service = create_embedding_service(use_openai=True)
        print("   - Service created successfully")
        
        print("2. Testing basic embedding generation...")
        test_text = "This is a test summary about artificial intelligence and machine learning"
        embedding = embedding_service.generate_embedding(test_text)
        print(f"   - Generated embedding with {len(embedding)} dimensions")
        
        if len(embedding) != 1536:
            print(f"   - WARNING: Expected 1536 dimensions, got {len(embedding)}")
            return False
        
        print("3. Testing summary vectorization...")
        vectorizer = SummaryVectorizer(embedding_service)
        
        test_summary = {
            'title': 'Test AI Video',
            'summary': 'This video explains artificial intelligence concepts and machine learning applications',
            'summary_type': 'detailed'
        }
        
        vectorized = vectorizer.vectorize_summary(test_summary)
        print(f"   - Vectorized summary successfully")
        print(f"   - Combined text: {vectorized['embedding_text']}")
        print(f"   - Embedding dimensions: {len(vectorized['embedding'])}")
        
        print("\nSUCCESS: OpenAI embeddings working correctly!")
        print(f"Ready to process {len(vectorized['embedding'])}-dimensional embeddings")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    if test_openai_embeddings():
        print("\nNext steps:")
        print("1. Apply the SQL schema in Supabase dashboard")
        print("2. Run the vectorization script to add embeddings to existing summaries")
    else:
        print("\nPlease fix the OpenAI API issues before proceeding.")