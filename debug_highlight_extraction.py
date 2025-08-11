#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Script für Highlight Extraction Problem
"""

import sys
import traceback
from database import Database
from video_highlight_extractor import VideoHighlightExtractor

def debug_video_lookup(video_id):
    """Debug Video-Datenbank Lookup"""
    print(f"🔍 Debug Video Lookup für ID: {video_id}")
    print("=" * 50)
    
    try:
        db = Database()
        
        # Alle Videos auflisten
        summaries = db.get_all_summaries()
        print(f"Anzahl Videos in DB: {len(summaries)}")
        
        # Video-IDs anzeigen
        video_ids = [s.get('video_id') for s in summaries]
        print(f"Verfügbare Video-IDs: {video_ids[:10]}...")  # Erste 10
        
        # Spezifisches Video suchen
        target_video = None
        for summary in summaries:
            if str(summary.get('video_id')) == str(video_id):
                target_video = summary
                break
                
        if target_video:
            print(f"✅ Video gefunden:")
            print(f"   ID: {target_video['video_id']}")
            print(f"   Titel: {target_video['title']}")
            print(f"   URL: {target_video['url']}")
            print(f"   Summary-Länge: {len(target_video.get('summary', ''))}")
            return target_video
        else:
            print(f"❌ Video mit ID {video_id} nicht gefunden!")
            print(f"Verfügbare IDs: {video_ids}")
            return None
            
    except Exception as e:
        print(f"❌ Fehler bei DB-Lookup: {str(e)}")
        traceback.print_exc()
        return None

def debug_srt_generation(video_id):
    """Debug SRT-Generierung"""
    print(f"\n🔧 Debug SRT-Generierung für ID: {video_id}")
    print("=" * 50)
    
    try:
        extractor = VideoHighlightExtractor()
        
        # Direkter Aufruf der SRT-Generierung
        srt_content = extractor._generate_srt_from_transcript(video_id)
        
        if srt_content:
            print(f"✅ SRT generiert: {len(srt_content)} Zeichen")
            print("📄 Erste Zeilen:")
            print(srt_content[:300] + "..." if len(srt_content) > 300 else srt_content)
            return srt_content
        else:
            print("❌ SRT-Generierung fehlgeschlagen")
            return None
            
    except Exception as e:
        print(f"❌ Fehler bei SRT-Generierung: {str(e)}")
        traceback.print_exc()
        return None

def debug_full_extraction(video_id):
    """Debug komplette Extraktion"""
    print(f"\n🎯 Debug komplette Extraktion für ID: {video_id}")
    print("=" * 50)
    
    try:
        extractor = VideoHighlightExtractor()
        
        # Hole Video-Info
        video_info = debug_video_lookup(video_id)
        if not video_info:
            return None
            
        # Teste Extraktion
        result = extractor.extract_highlights_from_video(
            video_id=video_id,
            video_url=video_info['url'],
            srt_content=None,  # Lass es generieren
            highlight_count=3,
            min_duration=10,
            max_duration=60
        )
        
        print(f"\n📊 Ergebnis:")
        print(f"Status: {result.get('status', 'unknown')}")
        if result.get('error'):
            print(f"Fehler: {result['error']}")
        else:
            print(f"Clips: {result.get('total_highlights', 0)}")
            
        return result
        
    except Exception as e:
        print(f"❌ Fehler bei kompletter Extraktion: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Hauptfunktion"""
    video_id = "19"  # Test mit Video-ID 19
    
    print("🐛 Debug Highlight Extraction")
    print("=" * 60)
    
    # Test 1: Video Lookup
    video_info = debug_video_lookup(video_id)
    
    if video_info:
        # Test 2: SRT Generierung
        srt_content = debug_srt_generation(video_id)
        
        # Test 3: Komplette Extraktion
        result = debug_full_extraction(video_id)
    
    print("\n" + "=" * 60)
    print("Debug abgeschlossen")

if __name__ == "__main__":
    main()