#!/usr/bin/env python3
"""
Automatische Themen-Kategorisierung für Video-Sammlung
Organisiert Videos in thematische Gruppen für bessere Q&A Ergebnisse
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
from supabase_client import SupabaseDatabase

load_dotenv()

class TopicCategorizer:
    """Kategorisiert Videos automatisch nach Themen"""
    
    def __init__(self):
        self.db = SupabaseDatabase()
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def analyze_video_topics(self, summaries: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Analysiert Videos und gruppiert sie nach Themen
        
        Returns:
            Dict mit Themennamen als Keys und Video-Listen als Values
        """
        try:
            # Erstelle Kontext für AI-Analyse
            video_info = []
            for summary in summaries:
                title = summary.get('title', 'Unknown')
                content = summary.get('summary', '')[:300]  # Erste 300 Zeichen
                video_info.append(f"Video: {title}\nInhalt: {content}\n")
            
            context = "\n".join(video_info)
            
            prompt = f"""Analysiere diese Video-Sammlung und kategorisiere sie nach Hauptthemen. 
            
VIDEO-SAMMLUNG:
{context}

AUFGABE:
1. Identifiziere die Hauptthemen (z.B. "Gaming", "Programmierung", "Tutorial", "KI/ML", etc.)
2. Ordne jedes Video genau EINEM Hauptthema zu
3. Gib das Ergebnis als strukturierte Liste zurück

ANTWORT FORMAT (genau so):
THEMA: Gaming
- Video: [Titel 1]
- Video: [Titel 2]

THEMA: Programmierung  
- Video: [Titel 3]
- Video: [Titel 4]

WICHTIG: Verwende deutsche Themennamen und sei spezifisch (z.B. "Tower Defense Games" statt nur "Gaming")."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für die Kategorisierung von Video-Inhalten."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            categorization = response.choices[0].message.content
            return self._parse_categorization(categorization, summaries)
            
        except Exception as e:
            print(f"Fehler bei der Themen-Analyse: {str(e)}")
            return self._fallback_categorization(summaries)
    
    def _parse_categorization(self, categorization: str, summaries: List[Dict]) -> Dict[str, List[Dict]]:
        """Parst die AI-Antwort und ordnet Videos zu"""
        topics = {}
        current_topic = None
        
        # Erstelle Titel->Video Mapping für schnelle Suche
        title_to_video = {video.get('title', ''): video for video in summaries}
        
        for line in categorization.split('\n'):
            line = line.strip()
            
            if line.startswith('THEMA:'):
                current_topic = line.replace('THEMA:', '').strip()
                topics[current_topic] = []
                
            elif line.startswith('- Video:') and current_topic:
                video_title = line.replace('- Video:', '').strip()
                
                # Finde passende Videos (fuzzy matching)
                for original_title, video in title_to_video.items():
                    if self._titles_match(video_title, original_title):
                        topics[current_topic].append(video)
                        break
        
        # Füge nicht kategorisierte Videos zu "Sonstige" hinzu
        all_categorized = []
        for video_list in topics.values():
            all_categorized.extend(video_list)
        
        uncategorized = [v for v in summaries if v not in all_categorized]
        if uncategorized:
            topics['Sonstige'] = uncategorized
            
        return topics
    
    def _titles_match(self, ai_title: str, original_title: str) -> bool:
        """Überprüft ob Titel ähnlich genug sind (fuzzy matching)"""
        ai_title = ai_title.lower().strip('[]"')
        original_title = original_title.lower()
        
        # Exakte Übereinstimmung
        if ai_title == original_title:
            return True
            
        # Teilübereinstimmung (mindestens 70% der Wörter)
        ai_words = set(ai_title.split())
        original_words = set(original_title.split())
        
        if not ai_words or not original_words:
            return False
            
        intersection = ai_words.intersection(original_words)
        similarity = len(intersection) / min(len(ai_words), len(original_words))
        
        return similarity >= 0.7
    
    def _fallback_categorization(self, summaries: List[Dict]) -> Dict[str, List[Dict]]:
        """Fallback-Kategorisierung basierend auf Schlüsselwörtern"""
        topics = {
            'Gaming': [],
            'Programmierung': [],
            'KI/ML': [],
            'Tutorial': [],
            'Sonstige': []
        }
        
        for video in summaries:
            title = video.get('title', '').lower()
            summary = video.get('summary', '').lower()
            content = title + ' ' + summary
            
            categorized = False
            
            # Gaming Keywords
            gaming_keywords = ['tower', 'game', 'gaming', 'mobile', 'strategy', 'defense']
            if any(keyword in content for keyword in gaming_keywords):
                topics['Gaming'].append(video)
                categorized = True
            
            # Programming Keywords
            elif any(keyword in content for keyword in ['python', 'javascript', 'code', 'programming', 'api', 'database']):
                topics['Programmierung'].append(video)
                categorized = True
                
            # AI/ML Keywords  
            elif any(keyword in content for keyword in ['ai', 'machine learning', 'neural', 'embedding', 'vector']):
                topics['KI/ML'].append(video)
                categorized = True
                
            # Tutorial Keywords
            elif any(keyword in content for keyword in ['tutorial', 'guide', 'how to', 'beginner']):
                topics['Tutorial'].append(video)
                categorized = True
            
            if not categorized:
                topics['Sonstige'].append(video)
        
        # Entferne leere Kategorien
        return {k: v for k, v in topics.items() if v}
    
    def create_topic_databases(self, topics: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        Erstellt separate 'virtuelle' Datenbanken für jedes Thema
        Returns: Dict mit Thema -> Info über Anzahl Videos
        """
        result = {}
        
        for topic_name, videos in topics.items():
            if videos:  # Nur wenn Videos vorhanden
                result[topic_name] = f"{len(videos)} Videos"
                
                # Speichere Topic-Zuordnung in der Datenbank (optional)
                self._save_topic_assignments(topic_name, videos)
        
        return result
    
    def _save_topic_assignments(self, topic_name: str, videos: List[Dict]):
        """Speichert Themen-Zuordnungen (optional für spätere Verwendung)"""
        try:
            for video in videos:
                video_id = video.get('id')
                if video_id:
                    # Hier könntest du eine topics Spalte in der Datenbank hinzufügen
                    # self.db.update_video_topic(video_id, topic_name)
                    pass
        except Exception as e:
            print(f"Warnung: Konnte Topic-Zuordnungen nicht speichern: {str(e)}")

def main():
    """Test der Themen-Kategorisierung"""
    print("Automatische Themen-Kategorisierung")
    print("=" * 40)
    
    categorizer = TopicCategorizer()
    
    # Hole alle Videos
    summaries = categorizer.db.get_all_summaries()
    print(f"Analysiere {len(summaries)} Videos...")
    
    # Kategorisiere nach Themen
    topics = categorizer.analyze_video_topics(summaries)
    
    print(f"\nGefundene Themen:")
    for topic_name, videos in topics.items():
        print(f"\n{topic_name} ({len(videos)} Videos):")
        for video in videos[:3]:  # Zeige nur erste 3
            title = video.get('title', 'Unknown')[:60]
            print(f"  - {title}...")
        if len(videos) > 3:
            print(f"  ... und {len(videos) - 3} weitere")
    
    # Erstelle Topic-Datenbanken
    topic_dbs = categorizer.create_topic_databases(topics)
    
    print(f"\nThemen-Datenbanken erstellt:")
    for topic, info in topic_dbs.items():
        print(f"  - {topic}: {info}")
    
    print(f"\nKategorisierung abgeschlossen!")
    return topics

if __name__ == "__main__":
    main()