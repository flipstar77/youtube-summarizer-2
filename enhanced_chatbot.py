#!/usr/bin/env python3
"""
Enhanced Chatbot with Topic-Specific Search
Verbesserte Q&A mit thematischen Datenbanken für präzisere Antworten
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
from supabase_client import SupabaseDatabase
from vector_embeddings import create_embedding_service
from topic_categorizer import TopicCategorizer

load_dotenv()

class EnhancedVideoQAChatbot:
    """Verbesserter Chatbot mit themen-spezifischer Suche"""
    
    def __init__(self):
        self.db = SupabaseDatabase()
        self.embedding_service = create_embedding_service(use_openai=True)
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.categorizer = TopicCategorizer()
        
        # Lade und kategorisiere Videos beim Start
        self._initialize_topics()
    
    def _initialize_topics(self):
        """Initialisiert Themen-Kategorien"""
        try:
            summaries = self.db.get_all_summaries()
            self.topics = self.categorizer.analyze_video_topics(summaries)
            print(f"[INFO] {len(self.topics)} Themen-Kategorien geladen:")
            for topic, videos in self.topics.items():
                print(f"  - {topic}: {len(videos)} Videos")
        except Exception as e:
            print(f"[WARNING] Themen-Kategorisierung fehlgeschlagen: {str(e)}")
            # Fallback: alle Videos in eine Kategorie
            self.topics = {"Alle Videos": self.db.get_all_summaries()}
    
    def ask_question_smart(self, question: str, max_videos: int = 5) -> Dict[str, Any]:
        """
        Intelligente Fragenbeantwortung mit automatischer Themen-Erkennung
        """
        try:
            # 1. Erkenne relevantes Thema für die Frage
            relevant_topic = self._detect_question_topic(question)
            
            # 2. Durchsuche nur Videos des relevanten Themas
            if relevant_topic and relevant_topic in self.topics:
                search_videos = self.topics[relevant_topic]
                topic_info = f" (Thema: {relevant_topic})"
            else:
                # Fallback: alle Videos durchsuchen
                search_videos = []
                for videos in self.topics.values():
                    search_videos.extend(videos)
                topic_info = " (Alle Themen)"
            
            print(f"[INFO] Durchsuche {len(search_videos)} Videos{topic_info}")
            
            # 3. Semantic Search innerhalb des Themas
            relevant_videos = self._search_in_video_list(question, search_videos, max_videos)
            
            if not relevant_videos:
                return {
                    'answer': f'Ich konnte keine relevanten Videos zu "{question}" finden{topic_info}.',
                    'sources': [],
                    'topic_used': relevant_topic,
                    'question': question
                }
            
            # 4. Generiere fokussierte Antwort
            answer = self._generate_focused_answer(question, relevant_videos, relevant_topic)
            
            # 5. Bereite Quellen auf
            sources = []
            for video in relevant_videos:
                sources.append({
                    'title': video.get('title', 'Unknown'),
                    'url': video.get('url', ''),
                    'video_id': video.get('video_id', ''),
                    'similarity': video.get('similarity', 0),
                    'summary': video.get('summary', '')[:200] + '...' if len(video.get('summary', '')) > 200 else video.get('summary', ''),
                    'topic': relevant_topic
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'topic_used': relevant_topic,
                'videos_searched': len(search_videos),
                'question': question
            }
            
        except Exception as e:
            return {
                'answer': f'Fehler beim Verarbeiten der Frage: {str(e)}',
                'sources': [],
                'topic_used': None,
                'question': question
            }
    
    def _detect_question_topic(self, question: str) -> Optional[str]:
        """Erkennt das relevante Thema für eine Frage"""
        try:
            # Erstelle Themen-Beschreibungen
            topic_descriptions = []
            for topic_name, videos in self.topics.items():
                sample_titles = [v.get('title', '') for v in videos[:3]]
                topic_descriptions.append(f"{topic_name}: {', '.join(sample_titles)}")
            
            topics_context = "\n".join(topic_descriptions)
            
            prompt = f"""Analysiere diese Frage und bestimme das relevanteste Thema aus der verfügbaren Liste.

FRAGE: {question}

VERFÜGBARE THEMEN:
{topics_context}

Antworte nur mit dem exakten Themennamen oder 'None' falls kein Thema passt.
Beispiel-Antworten: "Gaming", "Programmierung", "None"
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Du bist ein Experte für die Zuordnung von Fragen zu Themenkategorien."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            detected_topic = response.choices[0].message.content.strip()
            
            # Validiere dass das Thema existiert
            if detected_topic in self.topics:
                return detected_topic
            else:
                return None
                
        except Exception as e:
            print(f"[WARNING] Themen-Erkennung fehlgeschlagen: {str(e)}")
            return None
    
    def _search_in_video_list(self, question: str, videos: List[Dict], max_results: int) -> List[Dict]:
        """Führt semantische Suche in einer spezifischen Video-Liste durch"""
        if not videos:
            return []
        
        try:
            # Generiere Embedding für die Frage
            query_embedding = self.embedding_service.create_search_embedding(question)
            
            # Berechne Ähnlichkeit für jedes Video in der Liste
            video_similarities = []
            for video in videos:
                video_id = video.get('id')
                
                # Hole Embedding aus der Datenbank (falls vorhanden)
                try:
                    # Simuliere Ähnlichkeitssuche (vereinfacht)
                    all_similar = self.db.search_similar_summaries(
                        query_embedding=query_embedding,
                        threshold=0.1,
                        limit=100  # Viele holen, dann filtern
                    )
                    
                    # Filtere nur Videos aus unserer Themen-Liste
                    video_ids_in_topic = {v.get('id') for v in videos}
                    filtered_similar = [v for v in all_similar if v.get('id') in video_ids_in_topic]
                    
                    return filtered_similar[:max_results]
                    
                except Exception:
                    continue
            
            return []
            
        except Exception as e:
            print(f"[WARNING] Suche in Video-Liste fehlgeschlagen: {str(e)}")
            return videos[:max_results]  # Fallback
    
    def _generate_focused_answer(self, question: str, videos: List[Dict], topic: str) -> str:
        """Generiert fokussierte Antwort basierend auf Thema und Videos"""
        try:
            # Bereite Kontext vor
            context_parts = []
            for i, video in enumerate(videos, 1):
                title = video.get('title', 'Unknown')
                summary = video.get('summary', '')
                similarity = video.get('similarity', 0)
                
                context_parts.append(f"""
Video {i}: {title}
Relevanz: {similarity:.1%}
Inhalt: {summary[:400]}...
""")
            
            context = "\n".join(context_parts)
            
            # Themen-spezifisches System Prompt
            system_prompt = f"""Du bist ein Experte für das Thema "{topic}" und hilfst bei Fragen zu einer Video-Sammlung.

ANWEISUNGEN:
- Beantworte die Frage spezifisch basierend auf den bereitgestellten Videos
- Da alle Videos zum Thema "{topic}" gehören, fokussiere dich auf die Details
- Erwähne konkrete Video-Inhalte wenn relevant (z.B. "In Video 1 wird erklärt...")
- Falls die Antwort nicht vollständig in den Videos zu finden ist, sage das ehrlich
- Antworte auf Deutsch und strukturiert
- Sei präzise und hilfreich für das spezifische Thema "{topic}"
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"Frage: {question}\n\nRelevante {topic}-Videos:\n{context}\n\nBitte beantworte die Frage spezifisch basierend auf diesen Videos."
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Fehler bei der Antwort-Generierung: {str(e)}"
    
    def get_available_topics(self) -> Dict[str, int]:
        """Gibt verfügbare Themen und Anzahl Videos zurück"""
        return {topic: len(videos) for topic, videos in self.topics.items()}
    
    def search_by_topic(self, topic: str, question: str = None) -> Dict[str, Any]:
        """Durchsucht nur ein spezifisches Thema"""
        if topic not in self.topics:
            return {
                'error': f'Thema "{topic}" nicht gefunden',
                'available_topics': list(self.topics.keys())
            }
        
        videos = self.topics[topic]
        
        if question:
            # Spezifische Frage im Thema
            return self.ask_question_smart(question)
        else:
            # Überblick über das Thema
            return {
                'topic': topic,
                'video_count': len(videos),
                'videos': [{'title': v.get('title'), 'summary': v.get('summary', '')[:100] + '...'} for v in videos[:5]]
            }

def main():
    """Test des erweiterten Chatbots"""
    print("Enhanced Topic-Based Chatbot Test")
    print("=" * 40)
    
    chatbot = EnhancedVideoQAChatbot()
    
    # Zeige verfügbare Themen
    topics = chatbot.get_available_topics()
    print(f"Verfügbare Themen: {topics}")
    
    # Test eine Gaming-Frage
    print(f"\nTest: Gaming-spezifische Frage...")
    response = chatbot.ask_question_smart("Was sind typische Anfängerfehler bei Tower Defense?")
    
    print(f"Erkanntes Thema: {response['topic_used']}")
    print(f"Videos durchsucht: {response.get('videos_searched', 'Unknown')}")
    print(f"Relevante Videos: {len(response['sources'])}")
    print(f"Antwort: {response['answer'][:200]}...")
    
    print("\nEnhanced Chatbot bereit!")

if __name__ == "__main__":
    main()