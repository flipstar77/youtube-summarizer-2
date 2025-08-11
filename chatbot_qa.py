#!/usr/bin/env python3
"""
Chatbot Q&A System for YouTube Video Collection
Chat with your entire video library or individual transcripts
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
from supabase_client import SupabaseDatabase
from vector_embeddings import create_embedding_service

load_dotenv()

class VideoQAChatbot:
    """Chatbot that answers questions based on your video collection"""
    
    def __init__(self):
        self.db = SupabaseDatabase()
        self.embedding_service = create_embedding_service(use_openai=True)
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def detect_topic_keywords(self, question: str) -> List[str]:
        """Erkennt Themen-Keywords in der Frage für bessere Filterung"""
        question_lower = question.lower()
        
        topic_keywords = {
            'gaming': ['tower', 'game', 'gaming', 'mobile', 'strategy', 'defense', 'spielen', 'level', 'upgrade'],
            'programming': ['code', 'programming', 'python', 'javascript', 'api', 'database', 'programmieren', 'entwicklung'],
            'ai_ml': ['ai', 'machine learning', 'neural', 'embedding', 'vector', 'künstliche intelligenz', 'ki'],
            'tutorial': ['tutorial', 'guide', 'anleitung', 'lernen', 'beginner', 'anfänger']
        }
        
        detected = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                detected.append(topic)
        
        return detected

    def ask_question(self, question: str, max_videos: int = 5) -> Dict[str, Any]:
        """
        Ask a question about your video collection
        
        Args:
            question: The question to ask
            max_videos: Maximum number of videos to analyze
            
        Returns:
            Response with answer and source videos
        """
        try:
            # 1. Detect topic keywords for smarter filtering
            detected_topics = self.detect_topic_keywords(question)
            topic_info = f" (Erkannte Themen: {', '.join(detected_topics)})" if detected_topics else ""
            
            # 2. Find relevant videos using semantic search
            query_embedding = self.embedding_service.create_search_embedding(question)
            
            # Adjust threshold based on detected topics - be more selective for specific topics
            threshold = 0.4 if detected_topics else 0.3
            
            relevant_videos = self.db.search_similar_summaries(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=max_videos * 2  # Get more results for filtering
            )
            
            # 3. Filter videos based on detected topics (if any)
            if detected_topics and relevant_videos:
                filtered_videos = []
                for video in relevant_videos:
                    title = video.get('title', '').lower()
                    summary = video.get('summary', '').lower()
                    video_content = title + ' ' + summary
                    
                    # Check if video matches detected topics
                    for topic in detected_topics:
                        topic_keywords = {
                            'gaming': ['tower', 'game', 'gaming', 'mobile', 'strategy', 'defense'],
                            'programming': ['code', 'programming', 'python', 'javascript', 'api', 'database'],
                            'ai_ml': ['ai', 'machine learning', 'neural', 'embedding', 'vector'],
                            'tutorial': ['tutorial', 'guide', 'anleitung', 'lernen', 'beginner', 'anfänger']
                        }.get(topic, [])
                        
                        if any(keyword in video_content for keyword in topic_keywords):
                            filtered_videos.append(video)
                            break
                
                # Use filtered results if we have enough, otherwise use original
                if len(filtered_videos) >= 2:
                    relevant_videos = filtered_videos[:max_videos]
            
            if not relevant_videos:
                return {
                    'answer': f'Ich konnte keine relevanten Videos zu dieser Frage finden{topic_info}.',
                    'sources': [],
                    'question': question,
                    'detected_topics': detected_topics
                }
            
            # 4. Prepare context from relevant videos
            context_parts = []
            source_info = []
            
            for i, video in enumerate(relevant_videos, 1):
                similarity = video.get('similarity', 0)
                title = video.get('title', 'Unknown')
                summary = video.get('summary', '')
                video_id = video.get('video_id', '')
                url = video.get('url', '')
                
                # Add to context
                context_parts.append(f"""
Video {i}: {title}
Similarity: {similarity:.1%}
Summary: {summary[:500]}...
""")
                
                # Add to sources
                source_info.append({
                    'title': title,
                    'url': url,
                    'video_id': video_id,
                    'similarity': similarity,
                    'summary': summary[:200] + '...' if len(summary) > 200 else summary
                })
            
            # 5. Generate focused answer using AI
            context = "\\n\\n".join(context_parts)
            
            # Create topic-aware system prompt
            topic_context = f"Die Videos gehören hauptsächlich zu folgenden Themen: {', '.join(detected_topics)}. " if detected_topics else ""
            
            system_prompt = f"""Du bist ein hilfsreicher Assistent, der Fragen zu einer YouTube-Video-Sammlung beantwortet.

{topic_context}Basierend auf den folgenden Videos, beantworte die Frage des Nutzers:

VIDEOS:
{context}

ANWEISUNGEN:
- Beantworte die Frage spezifisch basierend auf den bereitgestellten Video-Inhalten
- Erwähne konkrete Videos wenn relevant (z.B. "In Video 1 über...")
- {f"Fokussiere dich auf {', '.join(detected_topics)}-spezifische Details" if detected_topics else ""}
- Falls die Antwort nicht vollständig in den Videos zu finden ist, sage das ehrlich
- Antworte auf Deutsch und strukturiert
- Sei präzise und hilfreich"""

            # Generate answer using direct OpenAI API for better control
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"Frage: {question}\n\nKontext aus den Videos:\n{context}\n\nBitte beantworte die Frage spezifisch basierend auf den bereitgestellten Video-Inhalten. Antworte auf Deutsch und sei präzise."
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': source_info,
                'question': question,
                'videos_analyzed': len(relevant_videos),
                'detected_topics': detected_topics,
                'topic_filtered': len(detected_topics) > 0
            }
            
        except Exception as e:
            return {
                'answer': f'Fehler beim Verarbeiten der Frage: {str(e)}',
                'sources': [],
                'question': question
            }
    
    def chat_with_transcript(self, video_id: str, question: str) -> Dict[str, Any]:
        """
        Chat directly with a specific video transcript
        
        Args:
            video_id: The video ID to chat with
            question: The question about this specific video
            
        Returns:
            Response based on the specific video
        """
        try:
            # Get the specific video
            summary_data = None
            all_summaries = self.db.get_all_summaries()
            
            for summary in all_summaries:
                if summary.get('video_id') == video_id:
                    summary_data = summary
                    break
            
            if not summary_data:
                return {
                    'answer': f'Video mit ID {video_id} nicht gefunden.',
                    'video_info': None,
                    'question': question
                }
            
            # Prepare context from this specific video
            title = summary_data.get('title', 'Unknown')
            summary = summary_data.get('summary', '')
            transcript = summary_data.get('transcript', '')
            
            # Use both summary and transcript if available
            content = f"Titel: {title}\\n\\nZusammenfassung: {summary}"
            if transcript:
                content += f"\\n\\nTranskript: {transcript[:2000]}..."  # Limit transcript length
            
            system_prompt = f"""Du bist ein hilfsreicher Assistent, der Fragen zu einem spezifischen YouTube-Video beantwortet.

VIDEO INFORMATIONEN:
{content}

ANWEISUNGEN:
- Beantworte die Frage basierend nur auf diesem Video
- Nutze sowohl die Zusammenfassung als auch das Transkript
- Falls die Antwort nicht im Video zu finden ist, sage das ehrlich
- Antworte auf Deutsch
- Sei präzise und beziehe dich auf spezifische Teile des Videos"""

            # Generate answer using direct OpenAI API
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Frage: {question}\n\nVideo Kontext:\n{content}\n\nBitte beantworte die Frage spezifisch basierend nur auf diesem Video. Antworte auf Deutsch."
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'video_info': {
                    'title': title,
                    'video_id': video_id,
                    'url': summary_data.get('url', ''),
                    'summary': summary[:200] + '...' if len(summary) > 200 else summary
                },
                'question': question
            }
            
        except Exception as e:
            return {
                'answer': f'Fehler beim Chat mit Video {video_id}: {str(e)}',
                'video_info': None,
                'question': question
            }

def main():
    """Test the chatbot functionality"""
    print("Video Q&A Chatbot Test")
    print("=" * 40)
    
    chatbot = VideoQAChatbot()
    
    # Test collection-wide question
    print("1. Testing collection-wide question...")
    question = "Wie funktioniert künstliche Intelligenz?"
    response = chatbot.ask_question(question)
    
    print(f"Frage: {response['question']}")
    print(f"Antwort: {response['answer'][:300]}...")
    print(f"Quellen: {len(response['sources'])} Videos")
    
    for i, source in enumerate(response['sources'][:2], 1):
        print(f"  {i}. {source['title']} ({source['similarity']:.1%} match)")
    
    print("\\n" + "="*40)
    
    # Test specific video chat
    print("2. Testing specific video chat...")
    all_videos = chatbot.db.get_all_summaries()
    if all_videos:
        test_video_id = all_videos[0].get('video_id')
        video_response = chatbot.chat_with_transcript(test_video_id, "Worum geht es in diesem Video?")
        
        print(f"Video: {video_response['video_info']['title'] if video_response['video_info'] else 'Not found'}")
        print(f"Antwort: {video_response['answer'][:300]}...")
    
    print("\\nChatbot functionality ready!")

if __name__ == "__main__":
    main()