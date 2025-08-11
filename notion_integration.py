#!/usr/bin/env python3
"""
Notion Integration für YouTube Summarizer
Synchronisiert Videos automatisch mit Notion-Datenbank
"""

import os
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from supabase_client import SupabaseDatabase

load_dotenv()

class NotionIntegration:
    """Integriert YouTube Videos mit Notion"""
    
    def __init__(self):
        self.notion_token = os.getenv('NOTION_TOKEN')
        self.database_id = os.getenv('NOTION_DATABASE_ID')
        self.base_url = "https://api.notion.com/v1"
        
        if not self.notion_token:
            print("[WARNING] NOTION_TOKEN nicht gefunden in .env")
        if not self.database_id:
            print("[WARNING] NOTION_DATABASE_ID nicht gefunden in .env")
        
        self.headers = {
            "Authorization": f"Bearer {self.notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        
        self.db = SupabaseDatabase()
    
    def setup_notion_database(self) -> Dict[str, Any]:
        """
        Erstellt oder überprüft Notion-Datenbank für Videos
        
        Returns:
            Dictionary mit Datenbank-Informationen
        """
        try:
            if not self.database_id:
                print("Keine Database ID konfiguriert - erstelle neue Datenbank...")
                return self._create_video_database()
            else:
                print(f"Überprüfe vorhandene Datenbank: {self.database_id}")
                return self._get_database_info()
                
        except Exception as e:
            return {"error": f"Database Setup fehlgeschlagen: {str(e)}"}
    
    def _create_video_database(self) -> Dict[str, Any]:
        """Erstellt neue Notion-Datenbank für Videos"""
        # Dieser Code würde eine neue Datenbank in einem Parent-Page erstellen
        # Für Einfachheit nehmen wir an, dass du manuell eine Datenbank erstellst
        return {
            "message": "Bitte erstelle manuell eine Notion-Datenbank mit den folgenden Spalten:",
            "required_columns": {
                "Titel": "title",
                "YouTube URL": "url", 
                "Video ID": "rich_text",
                "Zusammenfassung": "rich_text",
                "Themen": "multi_select",
                "Erstellt": "created_time",
                "Länge": "number",
                "Notizen": "rich_text"
            }
        }
    
    def _get_database_info(self) -> Dict[str, Any]:
        """Holt Informationen über vorhandene Datenbank"""
        try:
            response = requests.get(
                f"{self.base_url}/databases/{self.database_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data.get("title", [{}])[0].get("plain_text", "Unknown"),
                    "id": data.get("id"),
                    "properties": list(data.get("properties", {}).keys())
                }
            else:
                return {"error": f"Datenbank nicht gefunden: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Fehler beim Abrufen der Datenbank: {str(e)}"}
    
    def sync_video_to_notion(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronisiert ein Video mit Notion
        
        Args:
            video_data: Video-Daten aus Supabase
            
        Returns:
            Notion-Seiten-Informationen oder Fehler
        """
        try:
            # Überprüfe ob Video bereits in Notion existiert
            existing_page = self._find_video_in_notion(video_data.get('video_id'))
            
            if existing_page:
                print(f"Video bereits in Notion: {video_data.get('title')}")
                return {"status": "exists", "page_id": existing_page["id"]}
            
            # Erstelle neue Notion-Seite
            page_data = self._create_notion_page_data(video_data)
            
            response = requests.post(
                f"{self.base_url}/pages",
                headers=self.headers,
                json=page_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Video in Notion erstellt: {video_data.get('title')}")
                return {
                    "status": "created",
                    "page_id": result.get("id"),
                    "url": result.get("url")
                }
            else:
                return {"error": f"Notion API Fehler: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Sync Fehler: {str(e)}"}
    
    def _find_video_in_notion(self, video_id: str) -> Optional[Dict]:
        """Sucht Video in Notion-Datenbank"""
        try:
            query_data = {
                "filter": {
                    "property": "Video ID",
                    "rich_text": {
                        "equals": video_id
                    }
                }
            }
            
            response = requests.post(
                f"{self.base_url}/databases/{self.database_id}/query",
                headers=self.headers,
                json=query_data
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                return results[0] if results else None
            
            return None
            
        except Exception:
            return None
    
    def _create_notion_page_data(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Notion-Seiten-Daten aus Video-Informationen"""
        
        # Erkenne Themen basierend auf Titel und Summary
        topics = self._detect_video_topics(video_data)
        
        return {
            "parent": {"database_id": self.database_id},
            "properties": {
                "Titel": {
                    "title": [{"text": {"content": video_data.get('title', 'Unknown')}}]
                },
                "YouTube URL": {
                    "url": video_data.get('url', '')
                },
                "Video ID": {
                    "rich_text": [{"text": {"content": video_data.get('video_id', '')}}]
                },
                "Zusammenfassung": {
                    "rich_text": [{"text": {"content": video_data.get('summary', '')[:2000]}}]
                },
                "Themen": {
                    "multi_select": [{"name": topic} for topic in topics]
                },
                "Länge": {
                    "number": video_data.get('transcript_length', 0)
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": f"📺 Video automatisch synchronisiert am {datetime.now().strftime('%d.%m.%Y %H:%M')}"}
                            }
                        ]
                    }
                }
            ]
        }
    
    def _detect_video_topics(self, video_data: Dict[str, Any]) -> List[str]:
        """Erkennt Themen für Notion Multi-Select"""
        title = video_data.get('title', '').lower()
        summary = video_data.get('summary', '').lower()
        content = title + ' ' + summary
        
        topics = []
        
        # Gaming
        if any(keyword in content for keyword in ['tower', 'game', 'gaming', 'mobile', 'strategy']):
            topics.append('Gaming')
        
        # Programming
        if any(keyword in content for keyword in ['code', 'programming', 'python', 'javascript', 'api', 'database']):
            topics.append('Programmierung')
            
        # AI/ML
        if any(keyword in content for keyword in ['ai', 'machine learning', 'neural', 'embedding', 'vector']):
            topics.append('KI/ML')
            
        # Tutorial
        if any(keyword in content for keyword in ['tutorial', 'guide', 'anleitung', 'beginner']):
            topics.append('Tutorial')
        
        return topics if topics else ['Sonstige']
    
    def sync_all_videos(self) -> Dict[str, Any]:
        """Synchronisiert alle Videos mit Notion"""
        try:
            videos = self.db.get_all_summaries()
            results = {
                "total": len(videos),
                "created": 0,
                "exists": 0,
                "errors": []
            }
            
            print(f"Synchronisiere {len(videos)} Videos mit Notion...")
            
            for i, video in enumerate(videos, 1):
                print(f"[{i}/{len(videos)}] {video.get('title', 'Unknown')[:50]}...")
                
                result = self.sync_video_to_notion(video)
                
                if result.get("status") == "created":
                    results["created"] += 1
                elif result.get("status") == "exists":
                    results["exists"] += 1
                else:
                    results["errors"].append({
                        "video": video.get('title'),
                        "error": result.get("error")
                    })
            
            return results
            
        except Exception as e:
            return {"error": f"Bulk Sync Fehler: {str(e)}"}

# Konfiguration für .env Datei
NOTION_SETUP_GUIDE = """
# Notion Integration Setup

1. Erstelle eine Notion Integration:
   - Gehe zu https://www.notion.so/my-integrations
   - Klicke "New Integration"
   - Gib einen Namen ein: "YouTube Summarizer"
   - Kopiere den "Internal Integration Token"

2. Erstelle eine Notion-Datenbank:
   - Erstelle eine neue Seite in Notion
   - Füge eine Datenbank hinzu
   - Benenne sie: "YouTube Videos"
   - Erstelle folgende Spalten:
     * Titel (Title)
     * YouTube URL (URL)
     * Video ID (Text)
     * Zusammenfassung (Text)
     * Themen (Multi-select)
     * Länge (Number)
     * Notizen (Text)

3. Verbinde Integration mit Datenbank:
   - Gehe zur Datenbank-Seite
   - Klicke "..." → "Connections" → "YouTube Summarizer"

4. Füge zur .env Datei hinzu:
   NOTION_TOKEN=your_integration_token_here
   NOTION_DATABASE_ID=your_database_id_here
"""

def main():
    """Test der Notion Integration"""
    print("Notion Integration für YouTube Summarizer")
    print("=" * 50)
    
    # Setup Guide anzeigen falls nicht konfiguriert
    if not os.getenv('NOTION_TOKEN'):
        print(NOTION_SETUP_GUIDE)
        return
    
    notion = NotionIntegration()
    
    # Database Setup testen
    print("1. Überprüfe Notion-Datenbank...")
    db_info = notion.setup_notion_database()
    print(f"Datenbank: {db_info}")
    
    if not db_info.get("error"):
        print(f"\n2. Starte Video-Synchronisation...")
        sync_results = notion.sync_all_videos()
        
        print(f"\n📊 Sync-Ergebnisse:")
        print(f"  - Gesamt: {sync_results.get('total', 0)} Videos")
        print(f"  - Neu erstellt: {sync_results.get('created', 0)}")
        print(f"  - Bereits vorhanden: {sync_results.get('exists', 0)}")
        print(f"  - Fehler: {len(sync_results.get('errors', []))}")
        
        if sync_results.get("errors"):
            print("Fehler-Details:")
            for error in sync_results.get("errors", [])[:3]:
                print(f"  - {error}")

if __name__ == "__main__":
    main()