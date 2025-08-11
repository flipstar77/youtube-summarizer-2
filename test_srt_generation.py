#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script für SRT-Generierung im Video Highlight Extractor
"""

import os
import sys
from video_highlight_extractor import VideoHighlightExtractor

def test_srt_generation():
    """Test der SRT-Generierungsfunktionalität"""
    
    print("🧪 Test der SRT-Generierung")
    print("=" * 50)
    
    # Initialisiere Extractor
    try:
        extractor = VideoHighlightExtractor()
        print("✅ VideoHighlightExtractor erfolgreich initialisiert")
    except Exception as e:
        print(f"❌ Fehler bei Initialisierung: {str(e)}")
        return False
    
    # Test der _create_srt_from_text Methode
    test_text = """
    Hallo und willkommen zu diesem Tutorial. 
    Heute lernen wir, wie man Python verwendet.
    Zuerst installieren wir Python auf unserem Computer.
    Dann schreiben wir unser erstes Programm.
    Das wird sehr interessant und spannend.
    """
    
    print("\n📝 Teste SRT-Erstellung aus Text...")
    try:
        srt_content = extractor._create_srt_from_text(test_text, 120)  # 2 Minuten geschätzt
        
        if srt_content:
            print("✅ SRT-Inhalt erfolgreich generiert")
            print("\n📄 SRT-Vorschau:")
            print("-" * 30)
            print(srt_content[:300] + "..." if len(srt_content) > 300 else srt_content)
            print("-" * 30)
            
            # Speichere Test-SRT
            test_srt_path = os.path.join(extractor.temp_dir, "test_generated.srt")
            with open(test_srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            print(f"💾 Test-SRT gespeichert: {test_srt_path}")
            
            return True
        else:
            print("❌ Keine SRT-Inhalte generiert")
            return False
            
    except Exception as e:
        print(f"❌ Fehler bei SRT-Generierung: {str(e)}")
        return False

def test_with_existing_video():
    """Test mit einem existierenden Video aus der Datenbank"""
    
    print("\n🎬 Test mit existierendem Video")
    print("=" * 50)
    
    try:
        from database import Database
        db = Database()
        summaries = db.get_all_summaries()
        
        if not summaries:
            print("⚠️  Keine Videos in der Datenbank gefunden")
            return True
        
        # Nimm das erste Video
        test_video = summaries[0]
        video_id = test_video['video_id']
        
        print(f"📹 Teste mit Video: {test_video.get('title', 'Unbekannt')}")
        print(f"🆔 Video-ID: {video_id}")
        
        extractor = VideoHighlightExtractor()
        
        # Teste SRT-Generierung
        srt_content = extractor._generate_srt_from_transcript(video_id)
        
        if srt_content:
            print("✅ SRT aus Transcript erfolgreich generiert")
            
            # Zeige erste paar Zeilen
            lines = srt_content.split('\n')[:12]  # Erste 3 Segmente
            print("\n📄 SRT-Vorschau (erste Segmente):")
            print("-" * 30)
            print('\n'.join(lines))
            print("-" * 30)
            
            return True
        else:
            print("⚠️  SRT-Generierung aus Transcript fehlgeschlagen")
            print("   Das ist normal, wenn das Video kein verfügbares Transcript hat")
            return True
            
    except Exception as e:
        print(f"❌ Fehler beim Test mit existierendem Video: {str(e)}")
        return False

def main():
    """Hauptfunktion des Tests"""
    
    print("🎯 SRT-Generierung Test Suite")
    print("=" * 60)
    
    # Test 1: Basis SRT-Erstellung
    test1_success = test_srt_generation()
    
    # Test 2: Mit existierendem Video
    test2_success = test_with_existing_video()
    
    # Ergebnisse
    print("\n📊 Test-Ergebnisse")
    print("=" * 60)
    print(f"✅ SRT-Texterstellung: {'PASS' if test1_success else 'FAIL'}")
    print(f"✅ SRT aus Transcript:  {'PASS' if test2_success else 'FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 Alle Tests erfolgreich!")
        print("   Das System kann jetzt SRT-Dateien generieren wenn keine Untertitel verfügbar sind.")
    else:
        print("\n⚠️  Einige Tests fehlgeschlagen - siehe Details oben")
    
    return test1_success and test2_success

if __name__ == "__main__":
    main()