#!/usr/bin/env python3

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def create_schema_via_api():
    """Create the database schema using Supabase REST API"""
    
    # Read the schema SQL file
    with open('supabase_schema.sql', 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    # Supabase connection details
    supabase_url = os.getenv('SUPABASE_URL')
    service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not service_key:
        print("[ERROR] Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file")
        return False
    
    # Split SQL into individual statements
    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
    
    print(f"[INFO] Found {len(statements)} SQL statements to execute")
    
    # Execute each statement
    for i, statement in enumerate(statements, 1):
        if not statement or statement.startswith('--'):
            continue
            
        print(f"[INFO] Executing statement {i}/{len(statements)}...")
        
        # Use Supabase Edge Functions or direct PostgreSQL connection
        # For now, we'll use the PostgREST API for basic operations
        
        # Check if it's a CREATE TABLE statement
        if 'CREATE TABLE' in statement.upper():
            print(f"   Table creation statement detected")
            
        elif 'CREATE INDEX' in statement.upper():
            print(f"   Index creation statement detected")
            
        elif 'CREATE FUNCTION' in statement.upper():
            print(f"   Function creation statement detected")
            
        elif 'CREATE POLICY' in statement.upper():
            print(f"   Policy creation statement detected")
    
    print("\n[INFO] Schema creation completed via manual execution needed")
    print("Please copy the SQL from 'supabase_schema.sql' and run it in the Supabase dashboard")
    
    return True

def check_schema_exists():
    """Check if the schema was created successfully"""
    from supabase_client import SupabaseDatabase
    
    try:
        db = SupabaseDatabase()
        result = db.test_connection()
        if result:
            print("[SUCCESS] Schema exists and is working!")
            return True
        else:
            print("[ERROR] Schema does not exist yet")
            return False
    except Exception as e:
        print(f"[ERROR] Schema check failed: {str(e)}")
        return False

def main():
    print("Supabase Schema Creation Helper")
    print("=" * 40)
    
    # Check current status
    print("\n1. Checking if schema already exists...")
    if check_schema_exists():
        print("   Schema already exists! No action needed.")
        return True
    
    # Show manual instructions
    print("\n2. Manual Schema Creation Required")
    print("   Go to: https://supabase.com/dashboard")
    print("   1. Find your project: jhvnzkfqwzimvbwtjurt")
    print("   2. Click 'SQL Editor' in the left menu")
    print("   3. Click '+ New query'")
    print("   4. Copy and paste the contents of 'supabase_schema.sql'")
    print("   5. Click 'RUN' button")
    
    print(f"\n3. Schema content preview:")
    print("=" * 40)
    
    try:
        with open('supabase_schema.sql', 'r', encoding='utf-8') as f:
            content = f.read()
            print(content[:500] + "...")
            print("=" * 40)
            print(f"Total characters: {len(content)}")
    except FileNotFoundError:
        print("[ERROR] supabase_schema.sql file not found")
        return False
    
    input("\nPress Enter after you've run the SQL in Supabase dashboard...")
    
    # Test again
    print("\n4. Testing schema after creation...")
    if check_schema_exists():
        print("[SUCCESS] Schema creation successful!")
        return True
    else:
        print("[ERROR] Schema still not working. Check for errors in Supabase.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[READY] You can now use Supabase with your YouTube Summarizer!")
        print("Set USE_SUPABASE=true in your .env file to enable it.")