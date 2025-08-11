#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from supabase_client import SupabaseDatabase

load_dotenv()

def debug_database():
    """Debug the database contents and connection"""
    print("Debug YouTube Summarizer Database")
    print("=" * 40)
    
    try:
        # Connect to Supabase
        db = SupabaseDatabase()
        print("[OK] Connected to Supabase")
        
        # Get all summaries
        summaries = db.get_all_summaries()
        print(f"[INFO] Found {len(summaries)} summaries in database")
        
        if summaries:
            print("\n[DATA] Summaries:")
            for i, summary in enumerate(summaries, 1):
                print(f"  {i}. ID: {summary['id']}")
                print(f"     Video: {summary['video_id']}")
                print(f"     URL: {summary['url']}")
                print(f"     Type: {summary['summary_type']}")
                print(f"     Created: {summary['created_at']}")
                print(f"     Summary length: {len(summary['summary'])} chars")
                print(f"     Audio: {'Yes' if summary['audio_file'] else 'No'}")
                print()
        else:
            print("\n[INFO] No summaries found in database")
            
        # Test stats
        stats = db.get_summary_stats()
        print(f"[STATS] Database statistics:")
        print(f"  Total: {stats['total_summaries']}")
        print(f"  With audio: {stats['summaries_with_audio']}")
        print(f"  Recent (7 days): {stats['recent_summaries']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Database debug failed: {str(e)}")
        return False

def check_flask_config():
    """Check Flask app configuration"""
    print("\n[CONFIG] Environment Variables:")
    print("=" * 40)
    
    vars_to_check = [
        'USE_SUPABASE',
        'SUPABASE_URL', 
        'OPENAI_API_KEY',
        'ELEVENLABS_API_KEY'
    ]
    
    for var in vars_to_check:
        value = os.getenv(var, 'NOT SET')
        if 'KEY' in var or 'TOKEN' in var:
            display_value = f"{value[:20]}..." if value != 'NOT SET' else 'NOT SET'
        else:
            display_value = value
        print(f"  {var}: {display_value}")

def main():
    check_flask_config()
    
    if debug_database():
        print("\n[SUCCESS] Database is working correctly")
    else:
        print("\n[ERROR] Database has issues")

if __name__ == "__main__":
    main()