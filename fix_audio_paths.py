#!/usr/bin/env python3

from supabase_client import SupabaseDatabase

def fix_audio_paths():
    """Fix audio file paths with backslashes"""
    db = SupabaseDatabase()
    summaries = db.get_all_summaries()
    
    for summary in summaries:
        if summary['audio_file'] and '\\' in summary['audio_file']:
            old_path = summary['audio_file']
            # Fix the path: remove static/ prefix and convert backslashes
            if old_path.startswith('static'):
                new_path = old_path[7:].replace('\\', '/')  # Remove 'static/' and fix slashes
            else:
                new_path = old_path.replace('\\', '/')
                
            print(f"Fixing ID {summary['id']}:")
            print(f"  Old: '{old_path}'")
            print(f"  New: '{new_path}'")
            
            # Update in database
            success = db.update_audio_file(summary['id'], new_path, summary['voice_id'])
            print(f"  Updated: {success}")
            print()

if __name__ == "__main__":
    fix_audio_paths()