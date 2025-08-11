#!/usr/bin/env python3
"""
Prüfe welche Videos in der Datenbank verfügbar sind
"""

from database import Database
from supabase_client import SupabaseDatabase
import os

def check_databases():
    print("DATENBANK CHECK")
    print("=" * 50)
    
    # SQLite Datenbank
    print("SQLite Datenbank:")
    try:
        sqlite_db = Database()
        sqlite_summaries = sqlite_db.get_all_summaries()
        print(f"   Anzahl: {len(sqlite_summaries)}")
        for s in sqlite_summaries:
            print(f"   ID: {s['id']}, Video-ID: {s['video_id']}, Titel: {s.get('title', 'N/A')[:50]}...")
    except Exception as e:
        print(f"   Fehler: {str(e)}")
    
    print()
    
    # Supabase Datenbank  
    print("Supabase Datenbank:")
    try:
        if os.getenv('USE_SUPABASE', 'false').lower() == 'true':
            supabase_db = SupabaseDatabase()
            supabase_summaries = supabase_db.get_all_summaries()
            print(f"   Anzahl: {len(supabase_summaries)}")
            for s in supabase_summaries[:5]:  # Erste 5
                print(f"   ID: {s['id']}, Video-ID: {s['video_id']}, Titel: {s.get('title', 'N/A')[:50]}...")
        else:
            print("   Nicht konfiguriert (USE_SUPABASE=false)")
    except Exception as e:
        print(f"   Fehler: {str(e)}")

if __name__ == "__main__":
    check_databases()