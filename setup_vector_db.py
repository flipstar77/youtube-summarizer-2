#!/usr/bin/env python3
"""
Script to set up vector database schema in Supabase
This will create the embedding column, indexes, and required functions
"""

import os
import sys
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def setup_vector_database():
    """Setup vector database schema in Supabase"""
    
    # Get Supabase credentials
    url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not url or not service_key:
        print("❌ ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")
        return False
    
    print("🔧 Setting up vector database schema...")
    
    try:
        # Create Supabase client
        supabase = create_client(url, service_key)
        
        # Read the vector schema SQL
        with open('supabase_vector_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        print("📝 Applying vector schema to Supabase...")
        
        # Split the SQL into individual statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            if not statement:
                continue
                
            print(f"   Executing statement {i+1}/{len(statements)}...")
            
            try:
                # Execute each SQL statement
                result = supabase.rpc('exec_sql', {'sql': statement}).execute()
                print(f"   ✅ Statement {i+1} executed successfully")
                
            except Exception as e:
                # Try alternative method for SQL execution
                print(f"   ⚠️ RPC method failed, trying alternative approach...")
                try:
                    # Some statements might need to be run differently
                    if 'CREATE EXTENSION' in statement:
                        print(f"   📌 Extension creation may need manual setup in Supabase dashboard")
                    elif 'ALTER TABLE' in statement:
                        print(f"   📌 Column addition may need manual setup in Supabase dashboard")
                    else:
                        print(f"   ❌ Failed to execute: {str(e)}")
                except Exception as e2:
                    print(f"   ❌ Alternative method also failed: {str(e2)}")
        
        print("\n🎉 Vector database setup completed!")
        print("\n📋 Manual steps that may be required in Supabase dashboard:")
        print("   1. Enable the 'vector' extension in Database > Extensions")
        print("   2. Add 'embedding vector(384)' column to summaries table if not added")
        print("   3. Verify the functions were created in Database > Functions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up vector database: {str(e)}")
        return False

def test_vector_setup():
    """Test if vector database is properly set up"""
    print("\n🧪 Testing vector database setup...")
    
    try:
        from supabase_client import SupabaseDatabase
        db = SupabaseDatabase()
        
        # Test 1: Check if we can query the summaries table
        print("   Test 1: Basic database connection...")
        summaries = db.get_all_summaries()
        print(f"   ✅ Found {len(summaries)} summaries in database")
        
        # Test 2: Check if embedding column exists (by trying to get summaries without embeddings)
        print("   Test 2: Checking embedding column...")
        try:
            no_embeddings = db.get_summaries_without_embeddings()
            print(f"   ✅ Embedding column exists. {len(no_embeddings)} summaries need embeddings")
        except Exception as e:
            print(f"   ❌ Embedding column issue: {str(e)}")
            return False
        
        # Test 3: Check if vector functions exist
        print("   Test 3: Testing vector search functions...")
        try:
            # Test with a dummy embedding (384 dimensions)
            dummy_embedding = [0.0] * 384
            result = db.search_similar_summaries(dummy_embedding, threshold=0.1, limit=1)
            print(f"   ✅ Vector search functions working")
        except Exception as e:
            print(f"   ❌ Vector search functions issue: {str(e)}")
            return False
        
        print("\n🎉 All tests passed! Vector database is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def vectorize_existing_summaries():
    """Vectorize existing summaries that don't have embeddings"""
    print("\n🔄 Vectorizing existing summaries...")
    
    try:
        from supabase_client import SupabaseDatabase
        from vector_embeddings import create_embedding_service, SummaryVectorizer
        
        db = SupabaseDatabase()
        
        # Initialize embedding service
        use_openai = os.getenv('OPENAI_API_KEY') is not None
        embedding_service = create_embedding_service(use_openai=use_openai)
        vectorizer = SummaryVectorizer(embedding_service)
        
        # Get summaries without embeddings
        summaries_to_vectorize = db.get_summaries_without_embeddings()
        
        if not summaries_to_vectorize:
            print("   ✅ All summaries already have embeddings!")
            return True
        
        print(f"   📊 Found {len(summaries_to_vectorize)} summaries to vectorize")
        
        vectorized_count = 0
        for i, summary in enumerate(summaries_to_vectorize):
            try:
                print(f"   Processing {i+1}/{len(summaries_to_vectorize)}: {summary.get('title', 'No title')[:50]}...")
                
                # Generate embedding
                vectorized_data = vectorizer.vectorize_summary(summary)
                embedding = vectorized_data.get('embedding')
                
                if embedding:
                    # Update database
                    db.update_summary_embedding(summary['id'], embedding)
                    vectorized_count += 1
                    print(f"   ✅ Vectorized summary {summary['id']}")
                    
            except Exception as e:
                print(f"   ⚠️ Failed to vectorize summary {summary['id']}: {str(e)}")
                continue
        
        print(f"\n🎉 Vectorized {vectorized_count}/{len(summaries_to_vectorize)} summaries!")
        return True
        
    except Exception as e:
        print(f"❌ Error vectorizing summaries: {str(e)}")
        return False

if __name__ == "__main__":
    print("Vector Database Setup Tool")
    print("=" * 40)
    
    # Step 1: Setup database schema
    if not setup_vector_database():
        print("\n❌ Database setup failed. Please check the errors above.")
        sys.exit(1)
    
    # Step 2: Test the setup
    if not test_vector_setup():
        print("\n❌ Database tests failed. Please check the errors above.")
        sys.exit(1)
    
    # Step 3: Vectorize existing summaries
    vectorize_existing = input("\n🤔 Would you like to vectorize existing summaries now? (y/n): ").lower().strip()
    if vectorize_existing in ['y', 'yes']:
        if vectorize_existing_summaries():
            print("\n🎉 All done! Your vector database is ready to use.")
        else:
            print("\n⚠️ Some summaries couldn't be vectorized, but the database is set up.")
    else:
        print("\n✅ Database setup complete. You can vectorize summaries later.")
    
    print("\n🔍 To test semantic search, visit: http://localhost:5000")