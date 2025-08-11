"""
AI-Powered Highlight Detection Module

This module uses OpenAI's API to analyze video transcript segments and score them
for "highlight" value. It identifies the most engaging, informative, or important
parts of videos for extraction as highlights.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HighlightScore:
    """Represents a highlight score for a text segment"""
    segment_id: str
    start_time: float
    end_time: float
    text: str
    overall_score: float
    engagement_score: float
    information_score: float
    uniqueness_score: float
    emotional_score: float
    reasoning: str
    key_topics: List[str]
    sentiment: str
    recommended_action: str  # "extract", "maybe", "skip"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class HighlightAnalyzer:
    """AI-powered highlight detection and scoring system"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the highlight analyzer
        
        Args:
            api_key: OpenAI API key (if None, loads from environment)
            model: OpenAI model to use for analysis
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.analysis_cache = {}  # Cache for avoiding duplicate analyses
        
    def analyze_segments(self, segments: List[Dict], video_context: Optional[Dict] = None,
                        highlight_criteria: Optional[Dict] = None) -> List[HighlightScore]:
        """
        Analyze multiple segments and return highlight scores
        
        Args:
            segments: List of text segments with timing information
            video_context: Optional context about the video (title, topic, etc.)
            highlight_criteria: Custom criteria for what makes a good highlight
            
        Returns:
            List of HighlightScore objects sorted by overall score
        """
        results = []
        
        # Set default criteria if none provided
        if not highlight_criteria:
            highlight_criteria = self._get_default_criteria()
        
        logger.info(f"Analyzing {len(segments)} segments for highlights...")
        
        # Process segments in batches to optimize API usage
        batch_size = 5
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_results = self._analyze_segment_batch(batch, video_context, highlight_criteria)
            results.extend(batch_results)
            
            # Log progress
            processed = min(i + batch_size, len(segments))
            logger.info(f"Processed {processed}/{len(segments)} segments")
        
        # Sort by overall score (highest first)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return results
    
    def _analyze_segment_batch(self, segments: List[Dict], video_context: Optional[Dict],
                             highlight_criteria: Dict) -> List[HighlightScore]:
        """Analyze a batch of segments together for efficiency"""
        # Create cache key for this batch
        cache_key = self._create_cache_key(segments, video_context, highlight_criteria)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Prepare the analysis prompt
        prompt = self._create_analysis_prompt(segments, video_context, highlight_criteria)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent scoring
                max_tokens=2000
            )
            
            # Parse the response
            results = self._parse_analysis_response(response.choices[0].message.content, segments)
            
            # Cache the results
            self.analysis_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing segments: {str(e)}")
            # Return default scores as fallback
            return self._create_fallback_scores(segments)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for highlight analysis"""
        return """You are an expert video content analyzer specializing in identifying highlight-worthy segments.
Your task is to evaluate transcript segments and score them based on multiple criteria to identify the most engaging and valuable parts of videos.

For each segment, you should consider:
1. ENGAGEMENT: How likely is this to capture and hold viewer attention?
2. INFORMATION: How much valuable/useful information does this contain?
3. UNIQUENESS: How unique or surprising is this content?
4. EMOTIONAL IMPACT: Does this evoke strong emotions or reactions?

Score each aspect from 0-10, then provide an overall score (0-10) and recommend an action.

Respond in valid JSON format with an array of segment analyses."""
    
    def _create_analysis_prompt(self, segments: List[Dict], video_context: Optional[Dict],
                               highlight_criteria: Dict) -> str:
        """Create the analysis prompt for a batch of segments"""
        prompt = "Analyze the following video transcript segments for highlight potential:\n\n"
        
        # Add video context if available
        if video_context:
            prompt += f"VIDEO CONTEXT:\n"
            for key, value in video_context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        # Add highlight criteria
        prompt += f"HIGHLIGHT CRITERIA:\n"
        for key, value in highlight_criteria.items():
            prompt += f"- {key}: {value}\n"
        prompt += "\n"
        
        # Add segments to analyze
        prompt += "SEGMENTS TO ANALYZE:\n"
        for i, segment in enumerate(segments):
            prompt += f"\nSegment {i+1}:\n"
            prompt += f"Time: {segment.get('start_time_formatted', 'N/A')} - {segment.get('end_time_formatted', 'N/A')}\n"
            prompt += f"Duration: {segment.get('duration', 0):.1f} seconds\n"
            prompt += f"Text: {segment.get('text', '')}\n"
        
        prompt += """\n\nFor each segment, provide a JSON response with this structure:
{
  "segments": [
    {
      "segment_id": "1",
      "overall_score": 8.5,
      "engagement_score": 9.0,
      "information_score": 8.0,
      "uniqueness_score": 7.5,
      "emotional_score": 8.5,
      "reasoning": "Explanation of why this segment scored as it did",
      "key_topics": ["topic1", "topic2"],
      "sentiment": "positive/negative/neutral",
      "recommended_action": "extract/maybe/skip"
    }
  ]
}

Recommended actions:
- "extract": Overall score > 7.0, definitely worth highlighting
- "maybe": Overall score 5.0-7.0, consider for highlights depending on available slots
- "skip": Overall score < 5.0, not suitable for highlights

Be critical but fair in your scoring. Not every segment needs to be a highlight."""
        
        return prompt
    
    def _parse_analysis_response(self, response: str, segments: List[Dict]) -> List[HighlightScore]:
        """Parse the AI response and create HighlightScore objects"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = json.loads(response)
            
            results = []
            segment_analyses = response_data.get('segments', [])
            
            for i, analysis in enumerate(segment_analyses):
                if i < len(segments):
                    segment = segments[i]
                    
                    # Create HighlightScore object
                    score = HighlightScore(
                        segment_id=analysis.get('segment_id', str(i+1)),
                        start_time=segment.get('start_time', 0),
                        end_time=segment.get('end_time', 0),
                        text=segment.get('text', ''),
                        overall_score=float(analysis.get('overall_score', 0)),
                        engagement_score=float(analysis.get('engagement_score', 0)),
                        information_score=float(analysis.get('information_score', 0)),
                        uniqueness_score=float(analysis.get('uniqueness_score', 0)),
                        emotional_score=float(analysis.get('emotional_score', 0)),
                        reasoning=analysis.get('reasoning', ''),
                        key_topics=analysis.get('key_topics', []),
                        sentiment=analysis.get('sentiment', 'neutral'),
                        recommended_action=analysis.get('recommended_action', 'skip')
                    )
                    results.append(score)
            
            return results
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return self._create_fallback_scores(segments)
    
    def _create_fallback_scores(self, segments: List[Dict]) -> List[HighlightScore]:
        """Create fallback scores if AI analysis fails"""
        results = []
        
        for i, segment in enumerate(segments):
            # Simple heuristic scoring based on text length and keywords
            text = segment.get('text', '')
            duration = segment.get('duration', 0)
            
            # Basic scoring heuristics
            engagement_score = min(10, len(text) / 100 * 5)  # Longer text = more engagement
            information_score = self._calculate_information_score(text)
            uniqueness_score = 5.0  # Default neutral
            emotional_score = self._calculate_emotional_score(text)
            overall_score = (engagement_score + information_score + uniqueness_score + emotional_score) / 4
            
            score = HighlightScore(
                segment_id=str(i+1),
                start_time=segment.get('start_time', 0),
                end_time=segment.get('end_time', 0),
                text=text,
                overall_score=overall_score,
                engagement_score=engagement_score,
                information_score=information_score,
                uniqueness_score=uniqueness_score,
                emotional_score=emotional_score,
                reasoning="Fallback heuristic scoring (AI analysis unavailable)",
                key_topics=self._extract_simple_topics(text),
                sentiment="neutral",
                recommended_action="maybe" if overall_score >= 5.0 else "skip"
            )
            results.append(score)
        
        return results
    
    def _calculate_information_score(self, text: str) -> float:
        """Calculate information score based on text content"""
        # Look for informational keywords
        info_keywords = [
            'how', 'why', 'what', 'when', 'where', 'because', 'therefore',
            'important', 'key', 'main', 'first', 'second', 'step', 'process',
            'method', 'technique', 'strategy', 'tip', 'advice', 'remember'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in info_keywords if keyword in text_lower)
        
        # Score based on keyword density and text length
        base_score = min(10, keyword_count * 2)
        length_bonus = min(2, len(text) / 200)
        
        return min(10, base_score + length_bonus)
    
    def _calculate_emotional_score(self, text: str) -> float:
        """Calculate emotional impact score"""
        # Look for emotional keywords
        positive_words = ['amazing', 'great', 'excellent', 'fantastic', 'love', 'excited']
        negative_words = ['problem', 'issue', 'difficult', 'challenge', 'mistake', 'error']
        emphasis_words = ['really', 'very', 'extremely', 'incredibly', 'absolutely']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        emphasis_count = sum(1 for word in emphasis_words if word in text_lower)
        
        # Emotional punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        total_emotional_markers = positive_count + negative_count + emphasis_count + exclamation_count + question_count
        
        return min(10, total_emotional_markers * 1.5)
    
    def _extract_simple_topics(self, text: str) -> List[str]:
        """Extract simple topics from text using keyword analysis"""
        # Common topic keywords
        tech_words = ['python', 'code', 'programming', 'software', 'api', 'database']
        business_words = ['marketing', 'sales', 'business', 'strategy', 'growth']
        education_words = ['learn', 'tutorial', 'course', 'lesson', 'teach', 'explain']
        
        topics = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in tech_words):
            topics.append('technology')
        if any(word in text_lower for word in business_words):
            topics.append('business')
        if any(word in text_lower for word in education_words):
            topics.append('education')
        
        return topics or ['general']
    
    def _get_default_criteria(self) -> Dict:
        """Get default highlight criteria"""
        return {
            'engagement_weight': 0.3,
            'information_weight': 0.3,
            'uniqueness_weight': 0.2,
            'emotional_weight': 0.2,
            'min_duration': 10,  # Minimum seconds for a highlight
            'max_duration': 120,  # Maximum seconds for a highlight
            'preferred_topics': ['key_insights', 'actionable_tips', 'interesting_facts'],
            'avoid_topics': ['filler_content', 'repetitive_information']
        }
    
    def _create_cache_key(self, segments: List[Dict], video_context: Optional[Dict],
                         criteria: Dict) -> str:
        """Create a cache key for the analysis"""
        # Create a simple hash based on segment texts and criteria
        segment_texts = [seg.get('text', '')[:50] for seg in segments]  # First 50 chars
        cache_data = {
            'segments': segment_texts,
            'context': video_context,
            'criteria': criteria
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def filter_highlights(self, scores: List[HighlightScore], 
                         max_highlights: int = 10,
                         min_score: float = 6.0,
                         max_total_duration: float = 300.0) -> List[HighlightScore]:
        """
        Filter and select the best highlights based on constraints
        
        Args:
            scores: List of all highlight scores
            max_highlights: Maximum number of highlights to return
            min_score: Minimum overall score to consider
            max_total_duration: Maximum total duration of all highlights in seconds
            
        Returns:
            Filtered list of highlights
        """
        # Filter by minimum score
        candidates = [score for score in scores if score.overall_score >= min_score]
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Apply constraints
        selected = []
        total_duration = 0.0
        
        for candidate in candidates:
            if len(selected) >= max_highlights:
                break
            
            duration = candidate.end_time - candidate.start_time
            if total_duration + duration <= max_total_duration:
                selected.append(candidate)
                total_duration += duration
        
        return selected
    
    def create_highlight_report(self, scores: List[HighlightScore], 
                               output_path: Optional[str] = None) -> Dict:
        """
        Create a comprehensive report of highlight analysis
        
        Args:
            scores: List of highlight scores
            output_path: Optional path to save report as JSON
            
        Returns:
            Report dictionary
        """
        if not scores:
            return {'error': 'No scores provided'}
        
        # Calculate statistics
        overall_scores = [score.overall_score for score in scores]
        recommended_extracts = [score for score in scores if score.recommended_action == 'extract']
        
        report = {
            'analysis_summary': {
                'total_segments_analyzed': len(scores),
                'recommended_highlights': len(recommended_extracts),
                'average_score': round(sum(overall_scores) / len(overall_scores), 2),
                'highest_score': max(overall_scores),
                'lowest_score': min(overall_scores),
                'total_highlight_duration': sum(
                    score.end_time - score.start_time 
                    for score in recommended_extracts
                ),
            },
            'top_highlights': [
                score.to_dict() for score in 
                sorted(scores, key=lambda x: x.overall_score, reverse=True)[:10]
            ],
            'score_distribution': {
                'extract_count': len([s for s in scores if s.recommended_action == 'extract']),
                'maybe_count': len([s for s in scores if s.recommended_action == 'maybe']),
                'skip_count': len([s for s in scores if s.recommended_action == 'skip']),
            },
            'common_topics': self._analyze_common_topics(scores),
            'sentiment_distribution': self._analyze_sentiment_distribution(scores),
            'timestamp': os.popen('date').read().strip()
        }
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _analyze_common_topics(self, scores: List[HighlightScore]) -> Dict:
        """Analyze common topics across all scores"""
        topic_counts = {}
        
        for score in scores:
            for topic in score.key_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'most_common_topics': sorted_topics[:10],
            'total_unique_topics': len(topic_counts)
        }
    
    def _analyze_sentiment_distribution(self, scores: List[HighlightScore]) -> Dict:
        """Analyze sentiment distribution across scores"""
        sentiment_counts = {}
        
        for score in scores:
            sentiment = score.sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        total = len(scores)
        return {
            sentiment: {
                'count': count,
                'percentage': round((count / total) * 100, 1)
            }
            for sentiment, count in sentiment_counts.items()
        }


if __name__ == "__main__":
    # Example usage
    analyzer = HighlightAnalyzer()
    
    # Sample segments for testing
    test_segments = [
        {
            'id': 1,
            'start_time': 0,
            'end_time': 30,
            'duration': 30,
            'text': 'Welcome to this comprehensive tutorial on Python programming. Today we will learn some amazing techniques that will revolutionize how you write code!',
            'start_time_formatted': '00:00:00,000',
            'end_time_formatted': '00:00:30,000'
        },
        {
            'id': 2,
            'start_time': 30,
            'end_time': 60,
            'duration': 30,
            'text': 'So, um, let me just quickly show you this basic example. This is just a simple print statement, nothing too exciting here.',
            'start_time_formatted': '00:00:30,000',
            'end_time_formatted': '00:01:00,000'
        }
    ]
    
    # Analyze segments
    scores = analyzer.analyze_segments(test_segments)
    
    # Print results
    for score in scores:
        print(f"Segment {score.segment_id}: {score.overall_score}/10 - {score.recommended_action}")
        print(f"  Reasoning: {score.reasoning}")
        print()