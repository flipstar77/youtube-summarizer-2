#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from supabase_client import SupabaseDatabase

load_dotenv()

def test_supabase_connection():
    """Test Supabase database connection"""
    print("Testing Supabase Database Integration")
    print("=" * 50)
    
    try:
        # Initialize Supabase client
        print("1. Initializing Supabase client...")
        db = SupabaseDatabase()
        print("   [OK] Supabase client created")
        
        # Test connection
        print("2. Testing connection...")
        connection_ok = db.test_connection()
        if connection_ok:
            print("   [OK] Connection successful")
        else:
            print("   [ERROR] Connection failed")
            return False
        
        # Test basic operations
        print("3. Testing basic operations...")
        
        # Get all summaries (should work even if empty)
        summaries = db.get_all_summaries()
        print(f"   [OK] Retrieved {len(summaries)} existing summaries")
        
        # Test stats
        stats = db.get_summary_stats()
        print(f"   [OK] Stats: {stats['total_summaries']} total summaries")
        
        print("\n[SUCCESS] All Supabase tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Supabase test failed: {str(e)}")
        print("\n[TROUBLESHOOTING]:")
        print("   1. Check your .env file has correct SUPABASE_URL and keys")
        print("   2. Make sure you've run the SQL schema in Supabase")
        print("   3. Verify your Supabase project is active")
        return False

def create_schema_instructions():
    """Show instructions for creating the database schema"""
    print("\n[SCHEMA SETUP] Database Schema Setup Instructions:")
    print("=" * 50)
    print("1. Go to your Supabase project SQL editor:")
    print("   https://supabase.com/dashboard/project/jhvnzkfqwzimvbwtjurt/sql")
    print("")
    print("2. Copy and paste the contents of 'supabase_schema.sql'")
    print("3. Run the SQL to create tables and functions")
    print("")
    print("4. Your schema should include:")
    print("   - summaries table")
    print("   - indexes for performance")
    print("   - RLS policies")
    print("   - helper functions")

def show_current_config():
    """Show current configuration"""
    print("\n[CONFIG] Current Configuration:")
    print("=" * 50)
    
    supabase_url = os.getenv('SUPABASE_URL', 'NOT SET')
    supabase_anon = os.getenv('SUPABASE_ANON_KEY', 'NOT SET')
    supabase_service = os.getenv('SUPABASE_SERVICE_ROLE_KEY', 'NOT SET')
    access_token = os.getenv('SUPABASE_ACCESS_TOKEN', 'NOT SET')
    project_ref = os.getenv('SUPABASE_PROJECT_REF', 'NOT SET')
    
    print(f"SUPABASE_URL: {supabase_url}")
    print(f"SUPABASE_ANON_KEY: {supabase_anon[:20]}..." if supabase_anon != 'NOT SET' else "SUPABASE_ANON_KEY: NOT SET")
    print(f"SUPABASE_SERVICE_ROLE_KEY: {supabase_service[:20]}..." if supabase_service != 'NOT SET' else "SUPABASE_SERVICE_ROLE_KEY: NOT SET")
    print(f"SUPABASE_ACCESS_TOKEN: {access_token[:20]}..." if access_token != 'NOT SET' else "SUPABASE_ACCESS_TOKEN: NOT SET")
    print(f"SUPABASE_PROJECT_REF: {project_ref}")

def main():
    show_current_config()
    
    if not test_supabase_connection():
        create_schema_instructions()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Ready to use Supabase with your YouTube Summarizer!")
        print("   Set USE_SUPABASE=true in your .env file to enable it.")
    else:
        print("\n[WARNING] Please fix the issues above before using Supabase.")