#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Categorization & Auto-Tagging System
Intelligent categorization and tagging of video content using AI
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

from cache_manager import cache

load_dotenv()

@dataclass
class ContentCategory:
    """Content category definition"""
    name: str
    description: str
    keywords: List[str]
    confidence_threshold: float = 0.7
    color: str = "#3498db"
    icon: str = "📁"

@dataclass
class TagResult:
    """Tagging result"""
    tags: List[str]
    categories: List[str]
    confidence_scores: Dict[str, float]
    topics: List[str]
    sentiment: str
    difficulty_level: str
    target_audience: str
    
class ContentCategorizer:
    """AI-powered content categorization and tagging"""
    
    def __init__(self, ai_client=None):
        self.ai_client = ai_client
        self.categories = self._load_predefined_categories()
        self.tag_cache = {}
        
    def _load_predefined_categories(self) -> Dict[str, ContentCategory]:
        """Load predefined content categories"""
        return {
            'education': ContentCategory(
                name='Education',
                description='Educational and tutorial content',
                keywords=['tutorial', 'learn', 'guide', 'how-to', 'course', 'lesson', 'teach', 'explain', 'education'],
                color='#2ecc71',
                icon='🎓'
            ),
            'technology': ContentCategory(
                name='Technology',
                description='Technology, programming, and software content',
                keywords=['code', 'programming', 'software', 'tech', 'developer', 'coding', 'computer', 'AI', 'machine learning'],
                color='#3498db',
                icon='💻'
            ),
            'business': ContentCategory(
                name='Business',
                description='Business, entrepreneurship, and finance content',
                keywords=['business', 'entrepreneur', 'startup', 'finance', 'investment', 'marketing', 'sales', 'strategy'],
                color='#e67e22',
                icon='📈'
            ),
            'science': ContentCategory(
                name='Science',
                description='Scientific research and discoveries',
                keywords=['science', 'research', 'discovery', 'experiment', 'study', 'analysis', 'biology', 'physics', 'chemistry'],
                color='#9b59b6',
                icon='🔬'
            ),
            'entertainment': ContentCategory(
                name='Entertainment',
                description='Entertainment, gaming, and lifestyle content',
                keywords=['game', 'gaming', 'entertainment', 'fun', 'comedy', 'music', 'movie', 'review', 'lifestyle'],
                color='#e74c3c',
                icon='🎮'
            ),
            'health': ContentCategory(
                name='Health & Fitness',
                description='Health, fitness, and wellness content',
                keywords=['health', 'fitness', 'workout', 'exercise', 'nutrition', 'wellness', 'medical', 'diet'],
                color='#1abc9c',
                icon='🏥'
            ),
            'news': ContentCategory(
                name='News & Politics',
                description='News, current events, and political content',
                keywords=['news', 'politics', 'current events', 'government', 'policy', 'election', 'breaking'],
                color='#34495e',
                icon='📰'
            ),
            'creative': ContentCategory(
                name='Creative & Arts',
                description='Creative content, arts, and design',
                keywords=['art', 'design', 'creative', 'painting', 'drawing', 'photography', 'craft', 'DIY'],
                color='#f39c12',
                icon='🎨'
            )
        }
    
    def categorize_content(self, title: str, summary: str, transcript: str = "", 
                          use_ai: bool = True) -> TagResult:
        """Categorize and tag content using multiple approaches"""
        
        # Check cache first
        cache_key = f"categorize_{hash(title + summary[:100])}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return TagResult(**cached_result)
        
        print(f"[CATEGORIZER] Analyzing content: {title[:50]}...")
        
        # Combine content for analysis
        content_text = f"{title}\n\n{summary}"
        if transcript and len(transcript) > 100:
            # Use first 2000 characters of transcript for analysis
            content_text += f"\n\n{transcript[:2000]}"
        
        # Rule-based categorization
        rule_based_result = self._rule_based_categorization(content_text)
        
        # AI-powered analysis (if available and enabled)
        ai_result = None
        if use_ai and self.ai_client:
            try:
                ai_result = self._ai_categorization(content_text)
            except Exception as e:
                print(f"[WARNING] AI categorization failed: {e}")
        
        # Combine results
        final_result = self._combine_results(rule_based_result, ai_result)
        
        # Cache the result
        cache.set(cache_key, final_result.__dict__, expire=86400)  # 24 hours
        
        print(f"[CATEGORIZER] Categories: {final_result.categories}, Tags: {final_result.tags[:5]}")
        
        return final_result
    
    def _rule_based_categorization(self, content: str) -> TagResult:
        """Rule-based categorization using keyword matching"""
        
        content_lower = content.lower()
        
        # Category detection
        category_scores = {}
        for cat_id, category in self.categories.items():
            score = 0
            for keyword in category.keywords:
                # Count keyword occurrences with different weights
                title_matches = content_lower[:100].count(keyword.lower()) * 3  # Title has higher weight
                content_matches = content_lower.count(keyword.lower())
                score += title_matches + content_matches
            
            if score > 0:
                # Normalize score
                category_scores[cat_id] = min(score / 10, 1.0)
        
        # Select categories above threshold
        selected_categories = [
            cat_id for cat_id, score in category_scores.items() 
            if score >= self.categories[cat_id].confidence_threshold
        ]
        
        # If no categories meet threshold, pick the highest scoring one
        if not selected_categories and category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            selected_categories = [best_category[0]]
        
        # Extract tags using keyword matching and NLP techniques
        tags = self._extract_tags_from_text(content)
        
        # Determine difficulty level
        difficulty = self._assess_difficulty(content)
        
        # Determine target audience
        audience = self._determine_audience(content, selected_categories)
        
        # Basic sentiment analysis
        sentiment = self._analyze_sentiment(content)
        
        # Extract topics
        topics = self._extract_topics(content)
        
        return TagResult(
            tags=tags,
            categories=[self.categories[cat].name for cat in selected_categories],
            confidence_scores=category_scores,
            topics=topics,
            sentiment=sentiment,
            difficulty_level=difficulty,
            target_audience=audience
        )
    
    def _ai_categorization(self, content: str) -> TagResult:
        """AI-powered categorization using language models"""
        
        prompt = f"""
        Analyze the following video content and provide categorization:
        
        Content: {content[:3000]}
        
        Please provide a JSON response with:
        1. categories: List of relevant categories from [Education, Technology, Business, Science, Entertainment, Health, News, Creative]
        2. tags: 10-15 specific tags describing the content
        3. topics: 3-5 main topics covered
        4. sentiment: Overall sentiment (positive, neutral, negative)
        5. difficulty_level: Content difficulty (beginner, intermediate, advanced)
        6. target_audience: Primary audience (general, professionals, students, enthusiasts)
        7. confidence: Your confidence in this categorization (0-1)
        
        Respond with valid JSON only.
        """
        
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_data = json.loads(json_match.group())
                
                return TagResult(
                    tags=result_data.get('tags', []),
                    categories=result_data.get('categories', []),
                    confidence_scores={'ai_confidence': result_data.get('confidence', 0.8)},
                    topics=result_data.get('topics', []),
                    sentiment=result_data.get('sentiment', 'neutral'),
                    difficulty_level=result_data.get('difficulty_level', 'intermediate'),
                    target_audience=result_data.get('target_audience', 'general')
                )
                
        except Exception as e:
            print(f"[ERROR] AI categorization failed: {e}")
        
        return None
    
    def _combine_results(self, rule_based: TagResult, ai_result: Optional[TagResult]) -> TagResult:
        """Combine rule-based and AI results intelligently"""
        
        if not ai_result:
            return rule_based
        
        # Combine tags (remove duplicates)
        combined_tags = list(set(rule_based.tags + ai_result.tags))
        
        # Combine categories (prefer AI if confident)
        if ai_result.confidence_scores.get('ai_confidence', 0) > 0.8:
            combined_categories = list(set(rule_based.categories + ai_result.categories))
        else:
            combined_categories = rule_based.categories
        
        # Combine topics
        combined_topics = list(set(rule_based.topics + ai_result.topics))
        
        # Prefer AI results for subjective measures
        final_sentiment = ai_result.sentiment if ai_result.sentiment != 'neutral' else rule_based.sentiment
        final_difficulty = ai_result.difficulty_level if ai_result.difficulty_level else rule_based.difficulty_level
        final_audience = ai_result.target_audience if ai_result.target_audience != 'general' else rule_based.target_audience
        
        # Combine confidence scores
        combined_confidence = {**rule_based.confidence_scores, **ai_result.confidence_scores}
        
        return TagResult(
            tags=combined_tags[:15],  # Limit to 15 tags
            categories=combined_categories[:3],  # Limit to 3 categories
            confidence_scores=combined_confidence,
            topics=combined_topics[:5],  # Limit to 5 topics
            sentiment=final_sentiment,
            difficulty_level=final_difficulty,
            target_audience=final_audience
        )
    
    def _extract_tags_from_text(self, text: str) -> List[str]:
        """Extract relevant tags from text using NLP techniques"""
        
        # Common technical terms and concepts
        tech_terms = [
            'python', 'javascript', 'react', 'nodejs', 'api', 'database', 'aws', 'docker',
            'kubernetes', 'microservices', 'frontend', 'backend', 'fullstack', 'devops',
            'machine learning', 'deep learning', 'neural network', 'algorithm', 'data science'
        ]
        
        business_terms = [
            'startup', 'revenue', 'growth', 'marketing', 'sales', 'customer', 'product',
            'strategy', 'leadership', 'management', 'entrepreneurship', 'investment'
        ]
        
        educational_terms = [
            'tutorial', 'course', 'lesson', 'guide', 'explanation', 'example', 'practice',
            'beginner', 'advanced', 'step-by-step', 'walkthrough'
        ]
        
        all_terms = tech_terms + business_terms + educational_terms
        
        text_lower = text.lower()
        found_tags = []
        
        for term in all_terms:
            if term in text_lower:
                found_tags.append(term.title())
        
        # Extract hashtag-like terms
        hashtag_pattern = r'#(\w+)'
        hashtags = re.findall(hashtag_pattern, text)
        found_tags.extend(hashtags)
        
        # Extract capitalized terms (potential proper nouns)
        proper_noun_pattern = r'\b[A-Z][a-z]+\b'
        proper_nouns = re.findall(proper_noun_pattern, text)
        # Filter out common words
        common_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By'}
        proper_nouns = [noun for noun in proper_nouns if noun not in common_words and len(noun) > 3]
        found_tags.extend(proper_nouns[:5])  # Limit proper nouns
        
        # Remove duplicates and return
        return list(set(found_tags))[:10]
    
    def _assess_difficulty(self, content: str) -> str:
        """Assess content difficulty level"""
        
        text_lower = content.lower()
        
        beginner_indicators = ['beginner', 'introduction', 'basic', 'simple', 'easy', 'start', 'first']
        advanced_indicators = ['advanced', 'expert', 'complex', 'deep dive', 'sophisticated', 'professional']
        
        beginner_score = sum(1 for indicator in beginner_indicators if indicator in text_lower)
        advanced_score = sum(1 for indicator in advanced_indicators if indicator in text_lower)
        
        # Also consider sentence complexity (rough estimate)
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        if beginner_score > advanced_score or avg_sentence_length < 15:
            return 'beginner'
        elif advanced_score > beginner_score or avg_sentence_length > 25:
            return 'advanced'
        else:
            return 'intermediate'
    
    def _determine_audience(self, content: str, categories: List[str]) -> str:
        """Determine target audience based on content and categories"""
        
        text_lower = content.lower()
        
        # Audience indicators
        professional_indicators = ['professional', 'enterprise', 'business', 'corporate', 'industry']
        student_indicators = ['student', 'university', 'college', 'academic', 'research', 'study']
        enthusiast_indicators = ['hobby', 'enthusiast', 'fan', 'community', 'passion']
        
        prof_score = sum(1 for indicator in professional_indicators if indicator in text_lower)
        student_score = sum(1 for indicator in student_indicators if indicator in text_lower)
        enthusiast_score = sum(1 for indicator in enthusiast_indicators if indicator in text_lower)
        
        # Consider categories
        if 'education' in [cat.lower() for cat in categories]:
            student_score += 2
        if 'business' in [cat.lower() for cat in categories]:
            prof_score += 2
        
        if prof_score > student_score and prof_score > enthusiast_score:
            return 'professionals'
        elif student_score > enthusiast_score:
            return 'students'
        elif enthusiast_score > 0:
            return 'enthusiasts'
        else:
            return 'general'
    
    def _analyze_sentiment(self, content: str) -> str:
        """Basic sentiment analysis"""
        
        positive_words = ['great', 'excellent', 'amazing', 'awesome', 'fantastic', 'love', 'perfect', 'best']
        negative_words = ['terrible', 'awful', 'hate', 'worst', 'bad', 'horrible', 'disappointing']
        
        text_lower = content.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count + 1:
            return 'positive'
        elif negative_count > positive_count + 1:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content"""
        
        # Simple topic extraction based on frequent meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Remove common stop words
        stop_words = {
            'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know', 'want', 'been',
            'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like',
            'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 4]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words as topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return [topic[0].title() for topic in topics]
    
    def get_category_stats(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get categorization statistics for a set of summaries"""
        
        category_counts = {}
        tag_counts = {}
        total_videos = len(summaries)
        
        for summary in summaries:
            # Get or generate tags for this summary
            if 'tags' not in summary:
                result = self.categorize_content(
                    summary.get('title', ''),
                    summary.get('summary', ''),
                    summary.get('transcript', '')
                )
                categories = result.categories
                tags = result.tags
            else:
                categories = summary.get('categories', [])
                tags = summary.get('tags', [])
            
            # Count categories
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count tags
            for tag in tags[:5]:  # Top 5 tags only
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Calculate percentages
        category_stats = {
            cat: {'count': count, 'percentage': round((count / total_videos) * 100, 1)}
            for cat, count in category_counts.items()
        }
        
        # Get top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'total_videos': total_videos,
            'category_distribution': category_stats,
            'top_tags': top_tags,
            'categories_used': len(category_counts),
            'unique_tags': len(tag_counts)
        }