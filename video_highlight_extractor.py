#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Highlight Extractor
Extrahiert automatisch Highlights aus YouTube Videos basierend auf SRT-Dateien und AI-Analyse
"""

import os
import re
import json
import subprocess
import sys

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
from video_chaptering import VideoChapteringSystem

load_dotenv()

class VideoHighlightExtractor:
    """Extrahiert Video-Highlights basierend auf Transkripten und AI-Analyse"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.output_dir = "D:/mcp/highlights"
        self.temp_dir = "D:/mcp/temp"
        self.chaptering_system = VideoChapteringSystem()
        
        # Erstelle Ordner falls nicht vorhanden
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def extract_highlights_from_video(self, video_id: str, video_url: str, 
                                    srt_content: str = None, 
                                    highlight_count: int = 5,
                                    min_duration: int = 10,
                                    max_duration: int = 60) -> Dict[str, Any]:
        """
        Extrahiert Highlights aus einem Video
        
        Args:
            video_id: YouTube Video ID
            video_url: YouTube URL
            srt_content: SRT Untertitel (optional)
            highlight_count: Anzahl Highlights
            min_duration: Minimale Highlight-Dauer (Sekunden)
            max_duration: Maximale Highlight-Dauer (Sekunden)
        """
        try:
            print(f"[INFO] Extrahiere Highlights aus Video-ID: {video_id}")
            
            # 1. Video herunterladen
            video_path = self._download_video(video_url, video_id)
            if not video_path:
                return {"error": "Video Download fehlgeschlagen"}
            
            # 2. SRT-Datei parsen oder generieren
            if not srt_content:
                srt_content = self._extract_subtitles(video_path, video_id)
            
            if not srt_content:
                return {"error": "Keine Untertitel verfügbar und SRT-Generierung fehlgeschlagen"}
            
            # 3. SRT zu Segmenten parsen
            segments = self._parse_srt_to_segments(srt_content)
            if not segments:
                return {"error": "SRT Parsing fehlgeschlagen"}
            
            # 4. AI-basierte Highlight-Analyse
            highlight_segments = self._analyze_segments_for_highlights(
                segments, highlight_count, min_duration, max_duration
            )
            
            # 5. Video-Clips extrahieren
            extracted_clips = []
            for i, segment in enumerate(highlight_segments):
                clip_path = self._extract_video_clip(
                    video_path, segment, video_id, i + 1
                )
                
                # Füge immer Clip-Info hinzu, auch wenn Extraktion fehlschlägt
                clip_info = {
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": segment["text"],
                    "score": segment.get("highlight_score", 0),
                    "reason": segment.get("highlight_reason", ""),
                    "duration": f"{segment.get('end_seconds', 0) - segment.get('start_seconds', 0):.1f}s"
                }
                
                if clip_path:
                    clip_info["clip_path"] = clip_path
                    clip_info["status"] = "extracted"
                else:
                    clip_info["clip_path"] = None
                    clip_info["status"] = "identified_only"  # Highlight identifiziert, aber Clip-Extraktion fehlgeschlagen
                    
                extracted_clips.append(clip_info)
            
            # 6. Highlight-Reel zusammenstellen (nur mit erfolgreich extrahierten Clips)
            compilation_path = None
            successful_clips = [clip for clip in extracted_clips if clip.get("status") == "extracted"]
            if len(successful_clips) > 1:
                compilation_path = self._create_highlight_compilation(
                    successful_clips, video_id
                )
            
            # 7. Generate intelligent video chapters
            chapters_result = None
            try:
                # Get original transcript for chaptering (better than SRT)
                original_transcript = self._get_original_transcript(video_id)
                if original_transcript:
                    chapters_result = self.chaptering_system.create_video_chapters(
                        video_id=video_id,
                        transcript_text=original_transcript,
                        chapter_count=6,
                        min_duration=30
                    )
                    print(f"[INFO] Generated {chapters_result.get('total_chapters', 0)} video chapters")
            except Exception as e:
                print(f"[WARNING] Chapter generation failed: {str(e)}")
            
            return {
                "status": "success",
                "video_id": video_id,
                "clips": extracted_clips,
                "compilation": compilation_path,
                "total_highlights": len(extracted_clips),
                "highlights_identified": len(extracted_clips),
                "clips_extracted": len(successful_clips),
                "clips_failed": len(extracted_clips) - len(successful_clips),
                "chapters": chapters_result.get("chapters", []) if chapters_result else [],
                "youtube_description": chapters_result.get("youtube_description", "") if chapters_result else ""
            }
            
        except Exception as e:
            return {"error": f"Highlight Extraktion fehlgeschlagen: {str(e)}"}
    
    def _download_video(self, video_url: str, video_id: str) -> Optional[str]:
        """Lädt YouTube Video mit yt-dlp herunter"""
        try:
            output_path = os.path.join(self.temp_dir, f"{video_id}.%(ext)s")
            
            # yt-dlp Befehl - Beste Qualität bis 720p für schnellere Verarbeitung
            cmd = [
                "yt-dlp",
                "--format", "best[height<=720]",
                "--output", output_path,
                video_url
            ]
            
            print(f"[INFO] Lade Video herunter...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Finde heruntergeladene Datei
                for file in os.listdir(self.temp_dir):
                    if file.startswith(video_id):
                        return os.path.join(self.temp_dir, file)
            
            print(f"[ERROR] Video Download fehlgeschlagen: {result.stderr}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Download Fehler: {str(e)}")
            return None
    
    def _extract_subtitles(self, video_path: str, video_id: str) -> Optional[str]:
        """Extrahiert Untertitel mit yt-dlp oder generiert sie aus Transcript"""
        try:
            srt_path = os.path.join(self.temp_dir, f"{video_id}.srt")
            
            # Versuche zuerst Untertitel zu extrahieren
            cmd = [
                "yt-dlp",
                "--write-subs",
                "--write-auto-subs", 
                "--sub-lang", "de,en",
                "--skip-download",
                "--output", os.path.join(self.temp_dir, f"{video_id}.%(ext)s"),
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Suche nach SRT-Datei
            for file in os.listdir(self.temp_dir):
                if video_id in file and file.endswith('.srt'):
                    with open(os.path.join(self.temp_dir, file), 'r', encoding='utf-8') as f:
                        return f.read()
            
            print("[INFO] Keine Untertitel gefunden, generiere SRT aus Transcript...")
            return self._generate_srt_from_transcript(video_id)
            
        except Exception as e:
            print(f"[ERROR] Untertitel Extraktion fehlgeschlagen: {str(e)}")
            # Fallback: Generiere SRT aus Transcript
            return self._generate_srt_from_transcript(video_id)
    
    def _generate_srt_from_transcript(self, video_id: str) -> Optional[str]:
        """Generiert SRT-Datei aus YouTube Transcript API"""
        try:
            # Importiere hier um zirkuläre Imports zu vermeiden
            from transcript_extractor import TranscriptExtractor
            from database import Database
            from supabase_client import SupabaseDatabase
            import os
            
            # Verwende dieselbe Datenbank-Konfiguration wie die Flask App
            USE_SUPABASE = os.getenv('USE_SUPABASE', 'false').lower() == 'true'
            
            if USE_SUPABASE:
                try:
                    db = SupabaseDatabase()
                    print("[INFO] Verwende Supabase-Datenbank für Highlight-Extraktion")
                except Exception as e:
                    print(f"[WARNING] Supabase-Verbindung fehlgeschlagen, verwende SQLite: {str(e)}")
                    db = Database()
            else:
                db = Database()
            
            # Prüfe ob video_id eine Datenbank-ID (Zahl) oder YouTube-ID (String) ist
            try:
                # Versuche als Datenbank-ID
                db_id = int(video_id)
                video_summary = db.get_summary(db_id)
                if video_summary:
                    print(f"[INFO] Video gefunden über DB-ID {db_id}: {video_summary.get('title', 'Unbekannt')}")
                    video_id = video_summary['video_id']  # Verwende YouTube-ID für weitere Verarbeitung
                else:
                    print(f"[WARNING] Keine Summary für DB-ID {db_id} gefunden")
                    video_summary = None
            except ValueError:
                # Fallback: Suche nach YouTube-ID
                print(f"[INFO] Suche Video mit YouTube-ID: {video_id}")
                summaries = db.get_all_summaries()
                video_summary = None
                for summary in summaries:
                    if summary.get('video_id') == video_id:
                        video_summary = summary
                        break
            
            if not video_summary:
                print(f"[WARNING] Video {video_id} nicht in Datenbank gefunden")
                return None
            
            url = video_summary.get('url', '')
            if not url:
                print(f"[WARNING] Keine URL für Video {video_id} gefunden")
                return None
                
            print(f"[INFO] Extrahiere Transcript für Video {video_id}...")
            extractor = TranscriptExtractor()
            
            # Versuche verschiedene Sprachen und Auto-Generierung
            transcript_text = None
            transcript_data = None
            
            # Sprachen in Prioritäts-Reihenfolge versuchen
            languages = ['de', 'en', 'auto']
            for language in languages:
                try:
                    print(f"[INFO] Versuche Transcript-Extraktion für Sprache: {language}")
                    transcript_data = extractor.get_transcript(url, language)
                    if transcript_data and transcript_data.get('transcript'):
                        transcript_text = transcript_data['transcript']
                        if len(transcript_text.strip()) > 50:  # Mindestens 50 Zeichen
                            print(f"[SUCCESS] Transcript extrahiert ({language}, {len(transcript_text)} Zeichen)")
                            break
                        else:
                            print(f"[WARNING] Transcript für {language} zu kurz: {len(transcript_text)} Zeichen")
                    else:
                        print(f"[WARNING] Transcript für {language} leer oder fehlerhaft")
                except Exception as e:
                    print(f"[WARNING] Transcript-Extraktion für {language} fehlgeschlagen: {str(e)}")
                    continue
            
            # Fallback: Verwende das Summary-Feld falls vorhanden
            if not transcript_text or len(transcript_text.strip()) < 50:
                summary_text = video_summary.get('summary', '')
                if summary_text and len(summary_text) > 100:
                    print(f"[INFO] Verwende Summary als Fallback für SRT-Generierung ({len(summary_text)} Zeichen)")
                    transcript_text = summary_text
                    transcript_data = {'duration': video_summary.get('duration', 300)}  # Default 5 Minuten
            
            if not transcript_text or len(transcript_text.strip()) < 20:
                print("[ERROR] Kein ausreichender Text für SRT-Generierung gefunden")
                return None
                
            # Generiere SRT aus Text - verwende geschätzte Videolänge
            estimated_duration = transcript_data.get('duration', 0)
            if not estimated_duration and video_summary.get('duration'):
                estimated_duration = video_summary['duration']
                
            srt_content = self._create_srt_from_text(transcript_text, estimated_duration)
            
            if srt_content:
                # Speichere SRT-Datei
                srt_path = os.path.join(self.temp_dir, f"{video_id}.srt")
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                    
                print(f"[SUCCESS] SRT-Datei generiert: {srt_path}")
                return srt_content
            else:
                return None
            
        except Exception as e:
            print(f"[ERROR] SRT-Generierung fehlgeschlagen: {str(e)}")
            return None
    
    def _create_srt_from_text(self, text: str, estimated_duration: int = 0) -> str:
        """Erstellt SRT-Format aus Text-Inhalt"""
        try:
            # Teile Text in Sätze für Untertitel-Segmente
            sentences = []
            for sentence in text.replace('\n', ' ').split('.'):
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Nur sinnvolle Sätze
                    sentences.append(sentence)
            
            if not sentences:
                return ""
            
            # Schätze Timing - etwa 2-4 Sekunden pro Satz
            if estimated_duration > 0:
                base_duration = max(estimated_duration // len(sentences), 3)
            else:
                base_duration = 3
                
            srt_content = []
            start_time = 0
            
            for i, sentence in enumerate(sentences, 1):
                # Berechne Dauer basierend auf Satz-Länge (Lesegeschwindigkeit ~200 WPM)
                words = len(sentence.split())
                duration = max(int(words / 3.33), 2)  # 200 WPM = 3.33 Wörter pro Sekunde
                duration = min(duration, 8)  # Max 8 Sekunden pro Segment
                
                # Formatiere Zeitstempel
                start_hours = start_time // 3600
                start_minutes = (start_time % 3600) // 60
                start_seconds = start_time % 60
                
                end_time = start_time + duration
                end_hours = end_time // 3600
                end_minutes = (end_time % 3600) // 60
                end_seconds_final = end_time % 60
                
                # SRT-Eintrag hinzufügen
                srt_content.append(f"{i}")
                srt_content.append(f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d},000 --> {end_hours:02d}:{end_minutes:02d}:{end_seconds_final:02d},000")
                srt_content.append(f"{sentence}.")
                srt_content.append("")  # Leerzeile zwischen Einträgen
                
                start_time = end_time + 1  # 1 Sekunde Pause zwischen Untertiteln
            
            return '\n'.join(srt_content)
            
        except Exception as e:
            print(f"[ERROR] SRT-Erstellung fehlgeschlagen: {str(e)}")
            return ""
    
    def _parse_srt_to_segments(self, srt_content: str) -> List[Dict[str, Any]]:
        """Parst SRT-Inhalt zu Zeitsegmenten"""
        try:
            segments = []
            
            # SRT Pattern: Nummer, Zeitstempel, Text, Leerzeile
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n*$)'
            
            matches = re.findall(pattern, srt_content, re.DOTALL)
            
            for match in matches:
                number, start_time, end_time, text = match
                
                # Bereinige Text
                text = re.sub(r'<[^>]+>', '', text)  # HTML Tags entfernen
                text = text.replace('\n', ' ').strip()
                
                if text:  # Nur wenn Text vorhanden
                    segments.append({
                        "number": int(number),
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text,
                        "start_seconds": self._time_to_seconds(start_time),
                        "end_seconds": self._time_to_seconds(end_time)
                    })
            
            print(f"[INFO] {len(segments)} SRT Segmente geparst")
            return segments
            
        except Exception as e:
            print(f"[ERROR] SRT Parsing fehlgeschlagen: {str(e)}")
            return []
    
    def _time_to_seconds(self, time_str: str) -> float:
        """Konvertiert SRT Zeit (HH:MM:SS,mmm) zu Sekunden"""
        try:
            time_part, ms_part = time_str.split(',')
            h, m, s = map(int, time_part.split(':'))
            ms = int(ms_part)
            
            return h * 3600 + m * 60 + s + ms / 1000.0
        except:
            return 0.0
    
    def _analyze_segments_for_highlights(self, segments: List[Dict], 
                                       highlight_count: int,
                                       min_duration: int,
                                       max_duration: int) -> List[Dict[str, Any]]:
        """Analysiert Segmente mit AI um Highlights zu identifizieren"""
        try:
            print(f"[INFO] Analysiere {len(segments)} Segmente...")
            
            # Gruppiere Segmente zu längeren Abschnitten für bessere Analyse
            grouped_segments = self._group_segments(segments, min_duration, max_duration)
            
            # AI-Analyse für Highlight-Bewertung
            scored_segments = []
            
            for segment_group in grouped_segments:
                combined_text = " ".join([seg["text"] for seg in segment_group])
                
                # AI-Prompt für Highlight-Bewertung (ASCII-only)
                # Bereinige Text von Unicode-Zeichen
                clean_text = combined_text.encode('ascii', 'ignore').decode('ascii')
                
                prompt = f"""Analyze this video text segment and rate how likely it is to be an interesting highlight.

TEXT: {clean_text}

Rate from 1-10 based on:
- Information density and importance
- Entertainment or educational value
- Surprising or interesting statements
- Practical tips or insights
- Emotional moments or turning points

Answer only with: SCORE: X REASON: [Short explanation]"""

                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=100,
                        temperature=0.3
                    )
                    
                    result = response.choices[0].message.content.strip()
                    
                    # Parse Antwort
                    score_match = re.search(r'SCORE:\s*(\d+)', result)
                    reason_match = re.search(r'REASON:\s*(.+)', result)
                    
                    score = int(score_match.group(1)) if score_match else 5
                    reason = reason_match.group(1) if reason_match else "AI-Analyse"
                    
                    scored_segments.append({
                        "start_time": segment_group[0]["start_time"],
                        "end_time": segment_group[-1]["end_time"], 
                        "start_seconds": segment_group[0]["start_seconds"],
                        "end_seconds": segment_group[-1]["end_seconds"],
                        "text": combined_text,
                        "highlight_score": score,
                        "highlight_reason": reason
                    })
                    
                except Exception as e:
                    print(f"[WARNING] AI-Analyse fehlgeschlagen: {str(e)}")
                    # Fallback-Score basierend auf Text-Länge und Keywords
                    score = self._calculate_fallback_score(combined_text)
                    scored_segments.append({
                        "start_time": segment_group[0]["start_time"],
                        "end_time": segment_group[-1]["end_time"],
                        "start_seconds": segment_group[0]["start_seconds"], 
                        "end_seconds": segment_group[-1]["end_seconds"],
                        "text": combined_text,
                        "highlight_score": score,
                        "highlight_reason": "Keyword-basierte Analyse"
                    })
            
            # Sortiere nach Score und nimm die besten
            scored_segments.sort(key=lambda x: x["highlight_score"], reverse=True)
            
            # Stelle sicher, dass wir immer mindestens 1 Highlight haben, auch bei niedrigen Scores
            min_highlights = min(highlight_count, max(1, len(scored_segments) // 2))  # Mindestens 1, maximal die Hälfte aller Segmente
            
            # Nimm die besten Segmente mit flexibler Schwelle
            if len(scored_segments) > 0:
                # Zuerst versuche mit normaler Schwelle (5.0 statt 6.0 für mehr Ergebnisse)
                good_highlights = [seg for seg in scored_segments if seg["highlight_score"] >= 5.0]
                
                if len(good_highlights) < min_highlights:
                    print(f"[INFO] Nur {len(good_highlights)} Segmente über Score 5.0, nehme die besten {min_highlights}")
                    top_highlights = scored_segments[:min_highlights]
                elif len(good_highlights) > highlight_count:
                    print(f"[INFO] {len(good_highlights)} Segmente über Score 5.0, nehme die besten {highlight_count}")
                    top_highlights = good_highlights[:highlight_count]
                else:
                    top_highlights = good_highlights
                    
                # Fallback: Wenn immer noch keine Highlights, nimm einfach die besten Segmente
                if len(top_highlights) == 0 and len(scored_segments) > 0:
                    print(f"[INFO] Kein Segment über Score 5.0, nehme trotzdem die besten {min(3, len(scored_segments))} Segmente")
                    top_highlights = scored_segments[:min(3, len(scored_segments))]
            else:
                top_highlights = []
            
            # Sortiere chronologisch
            top_highlights.sort(key=lambda x: x["start_seconds"])
            
            print(f"[INFO] {len(top_highlights)} Top-Highlights identifiziert (von {len(scored_segments)} analysierten Segmenten)")
            return top_highlights
            
        except Exception as e:
            print(f"[ERROR] Highlight-Analyse fehlgeschlagen: {str(e)}")
            return []
    
    def _group_segments(self, segments: List[Dict], min_duration: int, max_duration: int) -> List[List[Dict]]:
        """Gruppiert SRT-Segmente zu zusammenhängenden Abschnitten"""
        grouped = []
        current_group = []
        
        for segment in segments:
            if not current_group:
                current_group.append(segment)
                continue
            
            # Prüfe ob Segment zur aktuellen Gruppe passt
            group_start = current_group[0]["start_seconds"]
            group_end = current_group[-1]["end_seconds"] 
            segment_start = segment["start_seconds"]
            
            current_duration = group_end - group_start
            
            # Füge hinzu wenn: Gap < 3 Sekunden und Gesamtdauer < max_duration
            if (segment_start - group_end < 3 and 
                current_duration < max_duration):
                current_group.append(segment)
            else:
                # Schließe aktuelle Gruppe ab wenn mindestens min_duration
                if current_duration >= min_duration:
                    grouped.append(current_group)
                current_group = [segment]
        
        # Letzte Gruppe hinzufügen
        if current_group:
            duration = current_group[-1]["end_seconds"] - current_group[0]["start_seconds"]
            if duration >= min_duration:
                grouped.append(current_group)
        
        return grouped
    
    def _calculate_fallback_score(self, text: str) -> int:
        """Berechnet Fallback-Score basierend auf Keywords - großzügiger für mehr Highlights"""
        text_lower = text.lower()
        
        # Erweiterte wichtige Keywords für verschiedene Kategorien
        important_keywords = [
            'wichtig', 'tipp', 'trick', 'geheimnis', 'fehler', 'problem', 
            'lösung', 'am besten', 'optimal', 'perfekt', 'niemals', 'immer',
            'aufpassen', 'achtung', 'warnung', 'empfehle', 'rate',
            'tutorial', 'lernen', 'verstehen', 'erklären', 'zeigen',
            'first', 'second', 'next', 'finally', 'conclusion'
        ]
        
        excitement_keywords = [
            'wow', 'krass', 'unglaublich', 'erstaunlich', 'fantastisch',
            'genial', 'brilliant', 'überraschend', 'schock', 'amazing',
            'great', 'awesome', 'incredible', 'fantastic'
        ]
        
        # Beginne mit höherem Basis-Score für mehr Highlights
        score = 6  # Erhöhter Basis-Score
        
        # Längerer Text bekommt Bonus
        word_count = len(text.split())
        if word_count > 20:
            score += 1
        if word_count > 40:
            score += 1
        
        # +1 für jedes wichtige Keyword (max +2)
        important_count = sum(1 for kw in important_keywords if kw in text_lower)
        score += min(important_count, 2)
        
        # +1 für Excitement Keywords (max +2)  
        excitement_count = sum(1 for kw in excitement_keywords if kw in text_lower)
        score += min(excitement_count, 2)
        
        # Bonus für Fragen (oft interessant)
        if '?' in text:
            score += 1
            
        # Bonus für Zahlen (oft faktisch wichtig)
        if re.search(r'\d+', text):
            score += 1
            
        # Bonus für strukturelle Elemente
        if any(word in text_lower for word in ['step', 'schritt', 'punkt', 'erstens', 'zweitens']):
            score += 1
        
        return min(score, 10)
    
    def _get_original_transcript(self, video_id: str) -> Optional[str]:
        """Gets the original transcript text for better chaptering"""
        try:
            # Importiere hier um zirkuläre Imports zu vermeiden
            from transcript_extractor import TranscriptExtractor
            from database import Database
            from supabase_client import SupabaseDatabase
            import os
            
            # Verwende dieselbe Datenbank-Konfiguration wie die Flask App
            USE_SUPABASE = os.getenv('USE_SUPABASE', 'false').lower() == 'true'
            
            if USE_SUPABASE:
                try:
                    db = SupabaseDatabase()
                except Exception:
                    db = Database()
            else:
                db = Database()
            
            # Hole Video-Info aus der Datenbank
            try:
                db_id = int(video_id)
                video_summary = db.get_summary(db_id)
                if video_summary:
                    youtube_video_id = video_summary['video_id']
                    url = video_summary.get('url', '')
                else:
                    return None
            except ValueError:
                # Fallback: Direkte YouTube-ID
                youtube_video_id = video_id
                summaries = db.get_all_summaries()
                for summary in summaries:
                    if summary.get('video_id') == video_id:
                        url = summary.get('url', '')
                        break
                else:
                    return None
            
            if not url:
                return None
                
            # Extrahiere frisches Transcript
            extractor = TranscriptExtractor()
            
            for language in ['de', 'en', 'auto']:
                try:
                    transcript_data = extractor.get_transcript(url, language)
                    if transcript_data and transcript_data.get('transcript'):
                        transcript_text = transcript_data['transcript']
                        if len(transcript_text.strip()) > 100:
                            return transcript_text
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Original transcript extraction failed: {str(e)}")
            return None
    
    def _extract_video_clip(self, video_path: str, segment: Dict, 
                           video_id: str, clip_number: int) -> Optional[str]:
        """Extrahiert Video-Clip mit FFmpeg"""
        try:
            start_time = segment["start_seconds"]
            end_time = segment["end_seconds"]
            duration = end_time - start_time
            
            output_filename = f"{video_id}_highlight_{clip_number:02d}_{int(start_time)}s.mp4"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # FFmpeg Befehl - mit Re-encoding für bessere Kompatibilität
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-c:v", "libx264",
                "-c:a", "aac", 
                "-preset", "medium",
                "-crf", "23",
                output_path
            ]
            
            print(f"[INFO] Extrahiere Clip {clip_number}: {start_time:.1f}s - {end_time:.1f}s")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_path):
                print(f"[SUCCESS] Clip erstellt: {output_filename}")
                return output_path
            else:
                print(f"[ERROR] FFmpeg Fehler: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Clip Extraktion fehlgeschlagen: {str(e)}")
            return None
    
    def _create_highlight_compilation(self, clips: List[Dict], video_id: str) -> Optional[str]:
        """Erstellt Zusammenstellung aller Highlights"""
        try:
            if len(clips) < 2:
                return None
            
            # Erstelle Liste der Clip-Pfade
            clip_paths = [clip["clip_path"] for clip in clips if os.path.exists(clip["clip_path"])]
            
            if len(clip_paths) < 2:
                return None
            
            # Erstelle temporäre Dateiliste für FFmpeg concat
            filelist_path = os.path.join(self.temp_dir, f"{video_id}_filelist.txt")
            
            with open(filelist_path, 'w', encoding='utf-8') as f:
                for clip_path in clip_paths:
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")
            
            # Output-Pfad für Compilation
            compilation_filename = f"{video_id}_highlights_compilation.mp4"
            compilation_path = os.path.join(self.output_dir, compilation_filename)
            
            # FFmpeg concat Befehl
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", filelist_path,
                "-c", "copy",
                compilation_path
            ]
            
            print(f"[INFO] Erstelle Highlight-Compilation mit {len(clip_paths)} Clips...")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup
            if os.path.exists(filelist_path):
                os.remove(filelist_path)
            
            if result.returncode == 0 and os.path.exists(compilation_path):
                print(f"[SUCCESS] Compilation erstellt: {compilation_filename}")
                return compilation_path
            else:
                print(f"[ERROR] Compilation fehlgeschlagen: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Compilation Fehler: {str(e)}")
            return None
    
    def cleanup_temp_files(self, video_id: str):
        """Löscht temporäre Dateien nach Verarbeitung"""
        try:
            temp_patterns = [video_id]
            
            for file in os.listdir(self.temp_dir):
                if any(pattern in file for pattern in temp_patterns):
                    file_path = os.path.join(self.temp_dir, file)
                    os.remove(file_path)
                    print(f"[INFO] Temporäre Datei gelöscht: {file}")
        except Exception as e:
            print(f"[WARNING] Cleanup fehlgeschlagen: {str(e)}")

def main():
    """Test der Video Highlight Extraktion"""
    print("Video Highlight Extractor Test")
    print("=" * 40)
    
    extractor = VideoHighlightExtractor()
    
    # Test mit einem Video (Du musst Video ID und URL anpassen)
    test_video_id = "test_video"  # Ersetze mit echter Video ID
    test_video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"  # Ersetze mit echter URL
    
    # Mock SRT für Test (normalerweise aus der Datenbank)
    test_srt = """1
00:00:05,000 --> 00:00:15,000
Das ist ein sehr wichtiger Tipp für Anfänger. Ihr solltet niemals diesen Fehler machen.

2
00:00:20,000 --> 00:00:30,000
Hier zeige ich euch den besten Trick um schnell voranzukommen.

3
00:00:35,000 --> 00:00:45,000
Wow, das ist wirklich erstaunlich! Diese Strategie funktioniert perfekt."""
    
    print("Teste Highlight-Extraktion...")
    result = extractor.extract_highlights_from_video(
        video_id=test_video_id,
        video_url=test_video_url,
        srt_content=test_srt,
        highlight_count=3
    )
    
    print(f"Ergebnis: {result}")

if __name__ == "__main__":
    main()