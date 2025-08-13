"""
AI Council Moderator
Orchestrates Tree-of-Thought debates with structured phases and pruning
"""

import json
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, AsyncGenerator
from datetime import datetime
import uuid

from .schemas import (
    CouncilSession, CouncilRound, CouncilConfig, DebatePhase,
    AgentResponse, JudgeScore, CouncilConsensus, AgentPersona,
    RetrievalQuery
)
from .router import ModelRouter, create_agent_personas
from .judge import CouncilJudge
from .agent import CouncilAgent
from enhanced_summarizer import EnhancedSummarizer

logger = logging.getLogger(__name__)


class CouncilModerator:
    """Orchestrates AI Council debates with Tree-of-Thought methodology"""
    
    def __init__(self, enhanced_summarizer: EnhancedSummarizer, vector_search=None):
        self.summarizer = enhanced_summarizer
        self.vector_search = vector_search
        self.router = ModelRouter(enhanced_summarizer)
        self.judge = CouncilJudge(enhanced_summarizer)
        
        # Load agent personas
        self.personas = {p.name: p for p in create_agent_personas()}
        
        # Session management
        self.active_sessions = {}
        self.session_history = []
        
        logger.info(f"[MODERATOR] Initialized with {len(self.personas)} personas")
    
    async def start_session(self, 
                          question: str, 
                          context: str = "",
                          config: CouncilConfig = None) -> str:
        """Start a new council session"""
        
        if config is None:
            config = CouncilConfig()
        
        session_id = str(uuid.uuid4())
        
        session = CouncilSession(
            session_id=session_id,
            question=question,
            context=context,
            config=config.dict(),
            participants=list(self.personas.keys())
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"[MODERATOR] Started session {session_id[:8]}: {question[:100]}")
        return session_id
    
    async def run_session(self, session_id: str) -> AsyncGenerator[Dict, None]:
        """Run complete council session with streaming updates"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        config = CouncilConfig(**session.config)
        
        try:
            # Initial knowledge retrieval
            yield {"type": "status", "message": "Retrieving relevant knowledge...", "progress": 10}
            retrieval_results = await self._retrieve_knowledge(session.question, session.context)
            
            # Phase 1: Framing
            yield {"type": "status", "message": "Phase 1: Framing the question...", "progress": 20}
            framing_round = await self._run_debate_round(
                session, DebatePhase.FRAMING, retrieval_results, config
            )
            session.rounds.append(framing_round)
            yield {"type": "round_complete", "round": framing_round.dict()}
            
            # Phase 2: Arguments (multiple rounds possible)
            for round_num in range(config.max_rounds - 1):
                progress = 20 + (40 * (round_num + 1) / config.max_rounds)
                yield {"type": "status", "message": f"Phase 2: Arguments (Round {round_num + 1})...", "progress": progress}
                
                argument_round = await self._run_debate_round(
                    session, DebatePhase.ARGUMENTS, retrieval_results, config, round_num + 1
                )
                session.rounds.append(argument_round)
                yield {"type": "round_complete", "round": argument_round.dict()}
                
                # Early termination if scores are very high
                if self._should_terminate_early(argument_round, config):
                    logger.info(f"[MODERATOR] Early termination - high quality responses achieved")
                    break
            
            # Phase 3: Rebuttal
            yield {"type": "status", "message": "Phase 3: Rebuttals and counter-arguments...", "progress": 70}
            rebuttal_round = await self._run_debate_round(
                session, DebatePhase.REBUTTAL, retrieval_results, config, 
                len(session.rounds) + 1
            )
            session.rounds.append(rebuttal_round)
            yield {"type": "round_complete", "round": rebuttal_round.dict()}
            
            # Phase 4: Final Synthesis
            yield {"type": "status", "message": "Phase 4: Building consensus...", "progress": 85}
            consensus = await self._build_final_consensus(session, retrieval_results)
            session.final_consensus = consensus
            
            # Complete session
            session.completed_at = datetime.now()
            session.status = "completed"
            
            # Move to history
            self.session_history.append(session)
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            yield {"type": "session_complete", "session": session.dict(), "progress": 100}
            
        except Exception as e:
            logger.error(f"[MODERATOR] Session {session_id} failed: {e}")
            session.status = "failed"
            session.completed_at = datetime.now()
            yield {"type": "error", "message": str(e)}
    
    async def _retrieve_knowledge(self, question: str, context: str) -> List[Dict]:
        """Retrieve relevant knowledge for the debate"""
        
        if not self.vector_search:
            logger.warning("[MODERATOR] No vector search available")
            return []
        
        try:
            # Enhanced retrieval with multiple queries
            queries = [
                question,
                f"{question} {context}".strip()
            ]
            
            # Add query variations for different perspectives
            perspective_queries = [
                f"What are the arguments for {question}",
                f"What are the arguments against {question}",
                f"Examples and evidence about {question}"
            ]
            queries.extend(perspective_queries)
            
            all_results = []
            seen_ids = set()
            
            for query in queries:
                try:
                    results = await self.vector_search.semantic_search(
                        query=query,
                        limit=5,
                        similarity_threshold=0.6
                    )
                    
                    # Deduplicate and add to results
                    for result in results:
                        result_id = result.get('id')
                        if result_id not in seen_ids:
                            seen_ids.add(result_id)
                            all_results.append(result)
                            
                except Exception as e:
                    logger.warning(f"[MODERATOR] Query '{query}' failed: {e}")
                    continue
            
            logger.info(f"[MODERATOR] Retrieved {len(all_results)} knowledge chunks")
            return all_results[:20]  # Limit total results
            
        except Exception as e:
            logger.error(f"[MODERATOR] Knowledge retrieval failed: {e}")
            return []
    
    async def _run_debate_round(self, 
                              session: CouncilSession, 
                              phase: DebatePhase,
                              retrieval_results: List[Dict],
                              config: CouncilConfig,
                              round_number: int = 1) -> CouncilRound:
        """Run a single debate round with Tree-of-Thought branching"""
        
        round_obj = CouncilRound(
            phase=phase,
            round_number=round_number,
            question=session.question,
            context=session.context,
            retrieval_results=retrieval_results
        )
        
        # Get agent responses (potentially with branching)
        all_responses = []
        
        if config.parallel_agents:
            # Run agents in parallel
            tasks = []
            for persona_name in session.participants:
                persona = self.personas[persona_name]
                task = self._get_agent_responses(
                    persona, phase, session.question, session.context,
                    retrieval_results, session.rounds, config
                )
                tasks.append(task)
            
            agent_response_groups = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten responses and handle exceptions
            for response_group in agent_response_groups:
                if isinstance(response_group, Exception):
                    logger.error(f"[MODERATOR] Agent failed: {response_group}")
                    continue
                all_responses.extend(response_group)
        else:
            # Run agents sequentially
            for persona_name in session.participants:
                persona = self.personas[persona_name]
                try:
                    responses = await self._get_agent_responses(
                        persona, phase, session.question, session.context,
                        retrieval_results, session.rounds, config
                    )
                    all_responses.extend(responses)
                except Exception as e:
                    logger.error(f"[MODERATOR] Agent {persona_name} failed: {e}")
                    continue
        
        # Score all responses
        scoring_tasks = []
        previous_responses = self._get_previous_responses(session.rounds)
        
        for response in all_responses:
            task = self.judge.score_response(response, retrieval_results, config, previous_responses)
            scoring_tasks.append(task)
        
        judge_scores = await asyncio.gather(*scoring_tasks, return_exceptions=True)
        valid_scores = [s for s in judge_scores if not isinstance(s, Exception)]
        
        # Tree-of-Thought pruning: keep top responses per agent
        if config.branching_factor > 1:
            pruned_responses, pruned_scores = self._prune_responses(
                all_responses, valid_scores, config
            )
            round_obj.agent_responses = pruned_responses
            round_obj.judge_scores = pruned_scores
        else:
            round_obj.agent_responses = all_responses
            round_obj.judge_scores = valid_scores
        
        round_obj.completed_at = datetime.now()
        
        logger.info(f"[MODERATOR] Round {round_number} ({phase.value}) completed with {len(round_obj.agent_responses)} responses")
        return round_obj
    
    async def _get_agent_responses(self, 
                                 persona: AgentPersona,
                                 phase: DebatePhase,
                                 question: str,
                                 context: str,
                                 retrieval_results: List[Dict],
                                 previous_rounds: List[CouncilRound],
                                 config: CouncilConfig) -> List[AgentResponse]:
        """Get response(s) from an agent with potential branching"""
        
        agent = CouncilAgent(persona, self.summarizer, self.router)
        
        responses = []
        
        # Generate multiple responses for Tree-of-Thought branching
        for variant in range(config.branching_factor):
            try:
                # Vary temperature for diversity
                temp_adjustment = variant * 0.1
                adjusted_temp = min(2.0, config.temperature + temp_adjustment)
                
                response = await agent.generate_response(
                    question=question,
                    context=context,
                    phase=phase,
                    retrieval_context=retrieval_results,
                    previous_rounds=previous_rounds,
                    temperature=adjusted_temp,
                    max_tokens=config.max_tokens_per_agent
                )
                
                if response:
                    responses.append(response)
                    
            except Exception as e:
                logger.error(f"[MODERATOR] Agent {persona.name} variant {variant} failed: {e}")
                continue
        
        return responses
    
    def _prune_responses(self, 
                        responses: List[AgentResponse], 
                        scores: List[JudgeScore],
                        config: CouncilConfig) -> Tuple[List[AgentResponse], List[JudgeScore]]:
        """Prune responses using Tree-of-Thought scoring"""
        
        # Group by agent
        agent_groups = {}
        for response, score in zip(responses, scores):
            agent_name = response.agent_name
            if agent_name not in agent_groups:
                agent_groups[agent_name] = []
            agent_groups[agent_name].append((response, score))
        
        # Keep top response(s) per agent
        pruned_responses = []
        pruned_scores = []
        
        keep_per_agent = 1  # In most cases, keep only the best response per agent
        
        for agent_name, response_score_pairs in agent_groups.items():
            # Sort by total score
            sorted_pairs = sorted(response_score_pairs, 
                                key=lambda x: x[1].scores.total, reverse=True)
            
            # Keep top responses
            for response, score in sorted_pairs[:keep_per_agent]:
                pruned_responses.append(response)
                pruned_scores.append(score)
        
        logger.debug(f"[MODERATOR] Pruned from {len(responses)} to {len(pruned_responses)} responses")
        return pruned_responses, pruned_scores
    
    def _get_previous_responses(self, rounds: List[CouncilRound]) -> List[AgentResponse]:
        """Get all previous agent responses for novelty scoring"""
        all_responses = []
        for round_obj in rounds:
            all_responses.extend(round_obj.agent_responses)
        return all_responses
    
    def _should_terminate_early(self, round_obj: CouncilRound, config: CouncilConfig) -> bool:
        """Determine if debate should terminate early due to high quality"""
        
        if not round_obj.judge_scores:
            return False
        
        # Check if all agents scored well
        high_quality_threshold = 24  # 80% of max score (30)
        high_quality_responses = [s for s in round_obj.judge_scores if s.scores.total >= high_quality_threshold]
        
        # Terminate early if most responses are high quality
        return len(high_quality_responses) >= len(round_obj.judge_scores) * 0.8
    
    async def _build_final_consensus(self, 
                                   session: CouncilSession, 
                                   retrieval_results: List[Dict]) -> CouncilConsensus:
        """Build final consensus from all debate rounds"""
        
        try:
            # Collect all agent responses and scores
            all_responses = []
            all_scores = []
            
            for round_obj in session.rounds:
                all_responses.extend(round_obj.agent_responses)
                all_scores.extend(round_obj.judge_scores)
            
            # Prepare synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(
                session.question, session.context, all_responses, all_scores
            )
            
            # Use the best-performing provider for synthesis
            synthesis_provider = self.router.choose_provider(
                task="synthesis",
                budget="high",
                latency="balanced"
            )
            
            # Generate consensus
            consensus_text = self.summarizer.summarize(
                text=synthesis_prompt,
                summary_type="custom",
                provider=synthesis_provider,
                custom_prompt="Generate a structured consensus as specified",
                max_tokens=800
            )
            
            # Parse consensus (attempt JSON, fallback to text parsing)
            consensus = self._parse_consensus(consensus_text, all_responses)
            
            logger.info(f"[MODERATOR] Built consensus for session {session.session_id[:8]}")
            return consensus
            
        except Exception as e:
            logger.error(f"[MODERATOR] Consensus building failed: {e}")
            
            # Fallback consensus
            return CouncilConsensus(
                consensus=["Unable to build full consensus due to technical issues"],
                disagreements=["Synthesis process encountered errors"],
                actionables=["Review session logs and retry analysis"],
                confidence=0.3,
                evidence_summary=[]
            )
    
    def _create_synthesis_prompt(self, 
                               question: str,
                               context: str, 
                               responses: List[AgentResponse],
                               scores: List[JudgeScore]) -> str:
        """Create prompt for final synthesis"""
        
        # Organize responses by agent and score
        agent_summaries = []
        for response in responses:
            # Find corresponding score
            score = next((s for s in scores if s.agent == response.agent_name), None)
            score_text = f"(Score: {score.scores.total}/30)" if score else ""
            
            agent_summary = f"""
**{response.agent_name}** {score_text}:
- Answer: {response.answer[:200]}...
- Key Claims: {'; '.join([c.text[:100] for c in response.claims[:2]])}
- Uncertainty: {response.uncertainty:.2f}
- Assumptions: {'; '.join(response.assumptions[:2])}
"""
            agent_summaries.append(agent_summary)
        
        return f"""As the MODERATOR of this AI Council debate, synthesize the discussion into a structured consensus.

ORIGINAL QUESTION: {question}
CONTEXT: {context}

DEBATE PARTICIPANTS AND RESPONSES:
{''.join(agent_summaries)}

Generate a JSON response with exactly this structure:
{{
    "consensus": ["point 1", "point 2", "point 3"],
    "disagreements": ["disagreement 1", "disagreement 2"], 
    "actionables": ["action 1", "action 2", "action 3"],
    "confidence": 0.85,
    "evidence_summary": [
        {{"video_id": "example", "snippet": "key evidence"}}
    ]
}}

Focus on:
- Consensus: What do the agents generally agree on? (3 bullets max)
- Disagreements: What are the key areas of disagreement? (2 bullets max) 
- Actionables: What concrete steps should be taken? (3 bullets max)
- Confidence: Overall confidence in the consensus (0.0-1.0)
- Evidence: Key pieces of evidence cited across responses"""
    
    def _parse_consensus(self, consensus_text: str, responses: List[AgentResponse]) -> CouncilConsensus:
        """Parse consensus from AI response"""
        
        try:
            # Try JSON parsing first
            if consensus_text.strip().startswith('{'):
                parsed = json.loads(consensus_text)
                return CouncilConsensus(**parsed)
        except:
            pass
        
        # Fallback to text parsing
        lines = consensus_text.split('\n')
        
        # Extract key evidence from responses
        evidence_summary = []
        for response in responses[:3]:  # Top 3 responses
            for claim in response.claims[:1]:  # Top claim per response
                if claim.evidence:
                    evidence = claim.evidence[0]
                    evidence_summary.append(evidence)
        
        return CouncilConsensus(
            consensus=[
                "Multiple perspectives were considered in the analysis",
                "Evidence from the knowledge base was referenced",
                "Areas of agreement were identified among AI agents"
            ],
            disagreements=[
                "Some agents had different interpretations",
                "Confidence levels varied across responses"
            ],
            actionables=[
                "Review the detailed agent responses",
                "Consider additional research on key topics",
                "Apply insights to decision-making process"
            ],
            confidence=0.7,
            evidence_summary=evidence_summary[:3]
        )
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of a session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                "session_id": session_id,
                "status": session.status,
                "started_at": session.started_at.isoformat(),
                "rounds_completed": len(session.rounds),
                "participants": session.participants,
                "question": session.question
            }
        
        # Check history
        for session in self.session_history:
            if session.session_id == session_id:
                return {
                    "session_id": session_id,
                    "status": session.status,
                    "started_at": session.started_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "rounds_completed": len(session.rounds),
                    "participants": session.participants,
                    "question": session.question
                }
        
        return None
    
    def get_moderator_stats(self) -> Dict:
        """Get moderator statistics"""
        total_sessions = len(self.session_history) + len(self.active_sessions)
        
        completed_sessions = [s for s in self.session_history if s.status == "completed"]
        
        if completed_sessions:
            avg_rounds = sum(len(s.rounds) for s in completed_sessions) / len(completed_sessions)
            avg_duration = sum(s.duration_seconds or 0 for s in completed_sessions) / len(completed_sessions)
        else:
            avg_rounds = 0
            avg_duration = 0
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(completed_sessions),
            "average_rounds": round(avg_rounds, 2),
            "average_duration_seconds": round(avg_duration, 2),
            "available_personas": len(self.personas)
        }