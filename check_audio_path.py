#!/usr/bin/env python3

from supabase_client import SupabaseDatabase

def check_audio_paths():
    """Check audio file paths in database"""
    db = SupabaseDatabase()
    summaries = db.get_all_summaries()
    
    for summary in summaries:
        if summary['audio_file']:
            print(f"ID {summary['id']}: audio_file = '{summary['audio_file']}'")
            
            # Check if file exists
            import os
            full_path = f"static/{summary['audio_file']}"
            exists = os.path.exists(full_path)
            print(f"  File exists at '{full_path}': {exists}")
            
            # Show what the URL should be
            correct_url = f"/static/{summary['audio_file']}"
            print(f"  Correct URL: {correct_url}")
            print()

if __name__ == "__main__":
    check_audio_paths()