#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direkter Test des Highlight Extraction Systems
"""

import traceback
import os
from video_highlight_extractor import VideoHighlightExtractor
from database import Database
from supabase_client import SupabaseDatabase

def direct_test():
    """Direkter Test der Highlight Extraktion"""
    
    print("🧪 DIREKTER TEST - Highlight Extraction")
    print("=" * 60)
    
    # Test Database-ID 19
    summary_id = 19
    
    try:
        print(f"1️⃣ Teste DB-Lookup für ID {summary_id}...")
        
        # Use same database config as Flask app
        USE_SUPABASE = os.getenv('USE_SUPABASE', 'false').lower() == 'true'
        
        if USE_SUPABASE:
            try:
                db = SupabaseDatabase()
                print("[INFO] Using Supabase database")
            except Exception as e:
                print(f"[WARNING] Supabase failed, using SQLite: {str(e)}")
                db = Database()
        else:
            db = Database()
            print("[INFO] Using SQLite database")
            
        summary = db.get_summary(summary_id)
        
        if summary:
            print(f"✅ Video gefunden:")
            print(f"   DB-ID: {summary['id']}")  
            print(f"   YouTube-ID: {summary['video_id']}")
            print(f"   Titel: {summary['title']}")
            print(f"   URL: {summary['url']}")
            
            print(f"\n2️⃣ Teste Highlight Extraktor...")
            
            extractor = VideoHighlightExtractor()
            
            print(f"3️⃣ Teste SRT-Generierung mit DB-ID {summary_id}...")
            srt_content = extractor._generate_srt_from_transcript(str(summary_id))
            
            if srt_content:
                print(f"✅ SRT generiert: {len(srt_content)} Zeichen")
                print("\n📄 SRT-Vorschau:")
                print("-" * 30)
                print(srt_content[:200] + "..." if len(srt_content) > 200 else srt_content)
                print("-" * 30)
                
                print(f"\n4️⃣ Teste komplette Extraktion...")
                result = extractor.extract_highlights_from_video(
                    video_id=str(summary_id),  # Database-ID als String
                    video_url=summary['url'],
                    srt_content=None,  # Lass es generieren
                    highlight_count=3,
                    min_duration=15,
                    max_duration=45
                )
                
                print(f"\n📊 ERGEBNIS:")
                print(f"Status: {result.get('status', 'unknown')}")
                if result.get('error'):
                    print(f"❌ Fehler: {result['error']}")
                elif result.get('status') == 'success':
                    print(f"✅ Erfolg: {result.get('total_highlights', 0)} Highlights")
                    
            else:
                print("❌ SRT-Generierung fehlgeschlagen")
                
        else:
            print(f"❌ Keine Summary für DB-ID {summary_id} gefunden")
            
    except Exception as e:
        print(f"❌ FEHLER: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    direct_test()