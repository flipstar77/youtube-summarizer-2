"""
AI Council Judge System
Evaluates agent responses for groundedness, coherence, and novelty
Implements Tree-of-Thought pruning through scoring
"""

import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from .schemas import AgentResponse, JudgeScore, ScoreBreakdown, Evidence, CouncilConfig
from enhanced_summarizer import EnhancedSummarizer

logger = logging.getLogger(__name__)


class CouncilJudge:
    """Evaluates and scores agent responses in council debates"""
    
    def __init__(self, enhanced_summarizer: EnhancedSummarizer):
        self.summarizer = enhanced_summarizer
        self.embedding_model = None  # Initialize lazily
        
        # Scoring weights
        self.score_weights = {
            'grounded': 0.4,    # Most important - evidence-based
            'coherence': 0.35,  # Logical consistency
            'novelty': 0.25     # New insights
        }
        
        logger.info("[JUDGE] Initialized council judge system")
    
    def _get_embedding_model(self):
        """Lazy initialization of embedding model"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("[JUDGE] Loaded embedding model for similarity scoring")
            except Exception as e:
                logger.warning(f"[JUDGE] Could not load embedding model: {e}")
                self.embedding_model = False  # Mark as unavailable
        return self.embedding_model
    
    async def score_response(self, 
                           response: AgentResponse,
                           retrieval_context: List[Dict],
                           config: CouncilConfig,
                           previous_responses: List[AgentResponse] = None) -> JudgeScore:
        """Score an agent response across all dimensions"""
        
        try:
            # Score each dimension
            grounded_score = self._score_groundedness(response, retrieval_context, config)
            coherence_score = self._score_coherence(response)
            novelty_score = self._score_novelty(response, previous_responses or [])
            
            scores = ScoreBreakdown(
                grounded=grounded_score,
                coherence=coherence_score,
                novelty=novelty_score
            )
            
            # Generate explanatory notes
            notes = self._generate_score_notes(response, scores, retrieval_context)
            
            # Score individual claims
            claim_scores = self._score_claims(response.claims, retrieval_context, config)
            
            judge_score = JudgeScore(
                agent=response.agent_name,
                scores=scores,
                notes=notes,
                claim_scores=claim_scores
            )
            
            logger.debug(f"[JUDGE] Scored {response.agent_name}: {scores.total}/30")
            return judge_score
            
        except Exception as e:
            logger.error(f"[JUDGE] Error scoring response: {e}")
            # Return minimal failing score
            return JudgeScore(
                agent=response.agent_name,
                scores=ScoreBreakdown(grounded=1, coherence=1, novelty=1),
                notes=f"Scoring failed: {str(e)}",
                claim_scores=[]
            )
    
    def _score_groundedness(self, 
                           response: AgentResponse, 
                           retrieval_context: List[Dict], 
                           config: CouncilConfig) -> int:
        """Score how well response is grounded in evidence (0-10)"""
        
        if not response.claims:
            return 0
        
        total_score = 0
        embedding_model = self._get_embedding_model()
        
        for claim in response.claims:
            claim_score = 0
            
            # Check if claim has evidence
            if not claim.evidence:
                if config.require_citations:
                    continue  # Skip ungrounded claims
                else:
                    claim_score = 2  # Minimal score for unsupported claims
            else:
                # Score evidence quality
                evidence_scores = []
                
                for evidence in claim.evidence:
                    # Check if evidence exists in retrieval context
                    evidence_found = False
                    for context_item in retrieval_context:
                        if evidence.video_id == context_item.get('video_id'):
                            evidence_found = True
                            
                            # Use similarity score if available
                            if evidence.relevance_score is not None:
                                evidence_scores.append(evidence.relevance_score * 10)
                            else:
                                # Calculate similarity if embedding model available
                                if embedding_model and embedding_model != False:
                                    similarity = self._calculate_similarity(
                                        claim.text, 
                                        context_item.get('summary', '')
                                    )
                                    evidence_scores.append(similarity * 10)
                                else:
                                    # Basic scoring based on citation format
                                    if evidence.start_s is not None and evidence.end_s is not None:
                                        evidence_scores.append(7)  # Has timestamps
                                    else:
                                        evidence_scores.append(5)  # Basic citation
                            break
                    
                    if not evidence_found:
                        evidence_scores.append(0)  # Citation not found
                
                # Average evidence scores for this claim
                if evidence_scores:
                    claim_score = sum(evidence_scores) / len(evidence_scores)
                else:
                    claim_score = 0
            
            total_score += claim_score
        
        if not response.claims:
            return 0
        
        # Average across claims and normalize to 0-10
        final_score = min(10, max(0, int(total_score / len(response.claims))))
        return final_score
    
    def _score_coherence(self, response: AgentResponse) -> int:
        """Score logical coherence and consistency (0-10)"""
        
        score = 5  # Start with baseline
        
        # Check response structure
        if len(response.answer) < 50:
            score -= 2  # Too brief
        elif len(response.answer) > 1000:
            score -= 1  # Too verbose
        
        # Check claim consistency
        if response.claims:
            # Claims should support the main answer
            if len(response.claims) < 2:
                score -= 1  # Too few claims
            elif len(response.claims) > 5:
                score -= 1  # Too many claims
            
            # Check for contradictions (simple keyword analysis)
            claim_texts = [claim.text.lower() for claim in response.claims]
            negative_words = ['not', 'never', 'no', 'false', 'incorrect', 'wrong']
            positive_words = ['yes', 'true', 'correct', 'always', 'definitely']
            
            has_negative = any(any(word in text for word in negative_words) for text in claim_texts)
            has_positive = any(any(word in text for word in positive_words) for text in claim_texts)
            
            if has_negative and has_positive:
                # Check if this is reasonable nuance vs contradiction
                if response.uncertainty < 0.3:  # High confidence with contradictions
                    score -= 2
        
        # Check uncertainty calibration
        if response.uncertainty < 0.1 and len(response.assumptions) == 0:
            score -= 1  # Overconfident without acknowledging assumptions
        elif response.uncertainty > 0.8 and len(response.claims) > 2:
            score -= 1  # Many claims but very uncertain
        
        return min(10, max(0, score))
    
    def _score_novelty(self, response: AgentResponse, previous_responses: List[AgentResponse]) -> int:
        """Score novelty and unique insights (0-10)"""
        
        if not previous_responses:
            return 7  # First response gets good novelty score
        
        embedding_model = self._get_embedding_model()
        
        # Extract key phrases from current response
        current_text = response.answer + " " + " ".join([claim.text for claim in response.claims])
        
        # Compare with previous responses
        similarities = []
        for prev_response in previous_responses:
            if prev_response.agent_name == response.agent_name:
                continue  # Don't compare with self
            
            prev_text = prev_response.answer + " " + " ".join([claim.text for claim in prev_response.claims])
            
            if embedding_model and embedding_model != False:
                similarity = self._calculate_similarity(current_text, prev_text)
                similarities.append(similarity)
            else:
                # Simple word overlap analysis
                current_words = set(current_text.lower().split())
                prev_words = set(prev_text.lower().split())
                overlap = len(current_words & prev_words) / len(current_words | prev_words)
                similarities.append(overlap)
        
        if not similarities:
            return 7
        
        # Higher similarity means lower novelty
        max_similarity = max(similarities)
        novelty_score = int(10 * (1 - max_similarity))
        
        # Bonus for asking good follow-up questions
        if response.next_questions and len(response.next_questions) > 0:
            novelty_score += 1
        
        # Bonus for identifying new assumptions
        if response.assumptions and len(response.assumptions) > 0:
            novelty_score += 1
        
        return min(10, max(0, novelty_score))
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        try:
            embedding_model = self._get_embedding_model()
            if not embedding_model or embedding_model == False:
                return 0.0
            
            embeddings = embedding_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"[JUDGE] Similarity calculation failed: {e}")
            return 0.0
    
    def _score_claims(self, 
                     claims: List, 
                     retrieval_context: List[Dict], 
                     config: CouncilConfig) -> List[Dict]:
        """Score individual claims"""
        
        claim_scores = []
        
        for i, claim in enumerate(claims):
            claim_analysis = {
                'claim_index': i,
                'text_preview': claim.text[:100] + "..." if len(claim.text) > 100 else claim.text,
                'evidence_count': len(claim.evidence) if hasattr(claim, 'evidence') else 0,
                'confidence': getattr(claim, 'confidence', 0.5),
                'grounding_score': 0,
                'notes': []
            }
            
            # Score evidence grounding
            if hasattr(claim, 'evidence') and claim.evidence:
                evidence_scores = []
                for evidence in claim.evidence:
                    if any(ctx.get('video_id') == evidence.video_id for ctx in retrieval_context):
                        evidence_scores.append(8)
                        claim_analysis['notes'].append(f"Found evidence in {evidence.video_id}")
                    else:
                        evidence_scores.append(2)
                        claim_analysis['notes'].append(f"Evidence {evidence.video_id} not in context")
                
                claim_analysis['grounding_score'] = int(np.mean(evidence_scores)) if evidence_scores else 0
            else:
                claim_analysis['notes'].append("No evidence provided")
                if config.require_citations:
                    claim_analysis['grounding_score'] = 0
                else:
                    claim_analysis['grounding_score'] = 3
            
            claim_scores.append(claim_analysis)
        
        return claim_scores
    
    def _generate_score_notes(self, 
                            response: AgentResponse, 
                            scores: ScoreBreakdown, 
                            retrieval_context: List[Dict]) -> str:
        """Generate explanatory notes for scores"""
        
        notes = []
        
        # Groundedness notes
        if scores.grounded >= 8:
            notes.append("🟢 Well-grounded with strong evidence citations.")
        elif scores.grounded >= 5:
            notes.append("🟡 Moderately grounded, some evidence provided.")
        else:
            notes.append("🔴 Poorly grounded, lacking sufficient evidence.")
        
        # Coherence notes
        if scores.coherence >= 8:
            notes.append("🟢 Highly coherent and logically consistent.")
        elif scores.coherence >= 5:
            notes.append("🟡 Generally coherent with minor issues.")
        else:
            notes.append("🔴 Lacks coherence or has logical inconsistencies.")
        
        # Novelty notes
        if scores.novelty >= 8:
            notes.append("🟢 Provides novel insights and unique perspectives.")
        elif scores.novelty >= 5:
            notes.append("🟡 Some new insights, builds on existing ideas.")
        else:
            notes.append("🔴 Limited novelty, mostly repeats existing points.")
        
        # Additional specific feedback
        if response.uncertainty > 0.7:
            notes.append("⚠️ High uncertainty acknowledged.")
        if len(response.assumptions) > 2:
            notes.append("📋 Good identification of key assumptions.")
        if len(response.next_questions) > 0:
            notes.append("❓ Suggested valuable follow-up questions.")
        
        return " ".join(notes)
    
    async def rank_responses(self, 
                           responses: List[AgentResponse], 
                           scores: List[JudgeScore]) -> List[Tuple[AgentResponse, JudgeScore]]:
        """Rank responses by total score for Tree-of-Thought pruning"""
        
        if len(responses) != len(scores):
            logger.error("[JUDGE] Mismatch between responses and scores")
            return list(zip(responses, scores))
        
        # Combine responses with scores
        combined = list(zip(responses, scores))
        
        # Sort by total score (descending)
        ranked = sorted(combined, key=lambda x: x[1].scores.total, reverse=True)
        
        logger.debug(f"[JUDGE] Ranked {len(responses)} responses")
        return ranked
    
    def get_judge_stats(self, scores: List[JudgeScore]) -> Dict:
        """Get statistics across judge scores"""
        
        if not scores:
            return {'total_scores': 0, 'averages': {}}
        
        stats = {
            'total_scores': len(scores),
            'averages': {
                'grounded': np.mean([s.scores.grounded for s in scores]),
                'coherence': np.mean([s.scores.coherence for s in scores]), 
                'novelty': np.mean([s.scores.novelty for s in scores]),
                'total': np.mean([s.scores.total for s in scores])
            },
            'best_agent': max(scores, key=lambda s: s.scores.total).agent,
            'score_distribution': {
                'excellent': len([s for s in scores if s.scores.total >= 24]),  # 80%+
                'good': len([s for s in scores if 18 <= s.scores.total < 24]),   # 60-80%
                'fair': len([s for s in scores if 12 <= s.scores.total < 18]),   # 40-60%
                'poor': len([s for s in scores if s.scores.total < 12])          # <40%
            }
        }
        
        return stats