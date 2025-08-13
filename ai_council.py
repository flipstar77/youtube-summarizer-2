#!/usr/bin/env python3
"""
AI Council System - Multi-AI Consensus Building
Orchestrates discussions between multiple AI models to reach informed consensus
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from dotenv import load_dotenv

# Import AI providers that we have available
from enhanced_summarizer import EnhancedSummarizer

load_dotenv()

@dataclass
class CouncilMember:
    """Represents an AI council member with specific role and capabilities"""
    name: str
    provider: str
    role: str
    expertise: str
    personality: str
    model: str

@dataclass
class CouncilResponse:
    """Response from a council member"""
    member: str
    role: str
    response: str
    timestamp: datetime
    reasoning: str

@dataclass
class CouncilSession:
    """Complete council session results"""
    question: str
    members: List[CouncilMember]
    discussion_rounds: List[List[CouncilResponse]]
    consensus: str
    final_answer: str
    session_summary: str
    started_at: datetime
    completed_at: Optional[datetime] = None

class AICouncil:
    """Orchestrates multi-AI discussions and consensus building"""
    
    def __init__(self):
        """Initialize AI Council with available providers"""
        self.enhanced_summarizer = EnhancedSummarizer()
        self.council_members = self._initialize_council_members()
        self.session_history = []
        
    def _initialize_council_members(self) -> List[CouncilMember]:
        """Define council members with their roles and personalities"""
        members = []
        
        # Available providers from enhanced_summarizer
        available_providers = self.enhanced_summarizer.providers
        
        # Define council structure
        council_structure = [
            {
                "name": "Dr. Analysis",
                "provider": "openai",
                "role": "Chief Analyst",
                "expertise": "Data analysis, logical reasoning, structured thinking",
                "personality": "Methodical, thorough, fact-focused",
                "model": "gpt-4o"
            },
            {
                "name": "Claude Critic", 
                "provider": "claude",
                "role": "Critical Evaluator",
                "expertise": "Critical thinking, finding flaws, alternative perspectives",
                "personality": "Skeptical, questioning, detail-oriented",
                "model": "claude-3-5-sonnet-20241022"
            },
            {
                "name": "Gemini Synthesizer",
                "provider": "gemini", 
                "role": "Synthesis Specialist",
                "expertise": "Pattern recognition, creative connections, holistic view",
                "personality": "Creative, integrative, big-picture thinking",
                "model": "gemini-1.5-pro"
            },
            {
                "name": "Perplexity Researcher",
                "provider": "perplexity",
                "role": "Research Specialist", 
                "expertise": "Current information, fact-checking, external knowledge",
                "personality": "Curious, thorough, evidence-based",
                "model": "llama-3.1-sonar-large-128k-online"
            },
            {
                "name": "Grok Challenger",
                "provider": "grok",
                "role": "Devil's Advocate",
                "expertise": "Challenging assumptions, unconventional thinking",
                "personality": "Provocative, unconventional, boundary-pushing",
                "model": "grok-beta"
            },
            {
                "name": "DeepSeek Strategist",
                "provider": "deepseek",
                "role": "Strategic Planner",
                "expertise": "Long-term thinking, strategic planning, implementation roadmaps",
                "personality": "Strategic, forward-thinking, practical",
                "model": "deepseek-chat"
            }
        ]
        
        # Only include members whose providers are available
        for member_config in council_structure:
            if member_config["provider"] in available_providers:
                members.append(CouncilMember(**member_config))
                
        return members
    
    async def hold_council_session(self, question: str, context: str = "", 
                                  discussion_rounds: int = 2) -> CouncilSession:
        """
        Orchestrate a full council session with multiple discussion rounds
        
        Args:
            question: The question to discuss
            context: Additional context for the discussion
            discussion_rounds: Number of discussion rounds (default: 2)
            
        Returns:
            Complete council session results
        """
        session = CouncilSession(
            question=question,
            members=self.council_members,
            discussion_rounds=[],
            consensus="",
            final_answer="",
            session_summary="",
            started_at=datetime.now()
        )
        
        print(f"[AI COUNCIL] Starting session with {len(self.council_members)} members")
        print(f"[AI COUNCIL] Question: {question}")
        
        try:
            # Initial round - each member provides their perspective
            print(f"[AI COUNCIL] Round 1: Initial perspectives")
            initial_responses = await self._conduct_discussion_round(
                question, context, [], session.members, round_number=1
            )
            session.discussion_rounds.append(initial_responses)
            
            # Additional discussion rounds
            for round_num in range(2, discussion_rounds + 1):
                print(f"[AI COUNCIL] Round {round_num}: Building on previous responses")
                previous_discussion = self._format_previous_discussion(session.discussion_rounds)
                
                round_responses = await self._conduct_discussion_round(
                    question, context, previous_discussion, session.members, round_num
                )
                session.discussion_rounds.append(round_responses)
            
            # Build consensus
            print(f"[AI COUNCIL] Building consensus...")
            session.consensus = await self._build_consensus(session)
            
            # Generate final answer
            print(f"[AI COUNCIL] Generating final answer...")
            session.final_answer = await self._generate_final_answer(session)
            
            # Create session summary
            session.session_summary = await self._create_session_summary(session)
            session.completed_at = datetime.now()
            
            # Store session
            self.session_history.append(session)
            
            print(f"[AI COUNCIL] Session completed successfully")
            return session
            
        except Exception as e:
            print(f"[AI COUNCIL ERROR] Session failed: {str(e)}")
            session.consensus = f"Council session failed: {str(e)}"
            session.final_answer = f"Error during discussion: {str(e)}"
            session.completed_at = datetime.now()
            return session
    
    async def _conduct_discussion_round(self, question: str, context: str, 
                                      previous_discussion: List[str], 
                                      members: List[CouncilMember],
                                      round_number: int) -> List[CouncilResponse]:
        """Conduct one round of discussion among council members"""
        responses = []
        
        for member in members:
            try:
                print(f"[AI COUNCIL] Getting response from {member.name} ({member.role})")
                
                # Create member-specific prompt
                prompt = self._create_member_prompt(
                    member, question, context, previous_discussion, round_number
                )
                
                # Get response from this AI provider
                response_text = await self._get_member_response(member, prompt)
                
                # Extract reasoning if possible
                reasoning = self._extract_reasoning(response_text)
                
                response = CouncilResponse(
                    member=member.name,
                    role=member.role,
                    response=response_text,
                    timestamp=datetime.now(),
                    reasoning=reasoning
                )
                
                responses.append(response)
                
            except Exception as e:
                print(f"[AI COUNCIL WARNING] Failed to get response from {member.name}: {str(e)}")
                # Add error response
                error_response = CouncilResponse(
                    member=member.name,
                    role=member.role,
                    response=f"Unable to participate in this round: {str(e)}",
                    timestamp=datetime.now(),
                    reasoning="Error occurred"
                )
                responses.append(error_response)
        
        return responses
    
    def _create_member_prompt(self, member: CouncilMember, question: str, 
                            context: str, previous_discussion: List[str], 
                            round_number: int) -> str:
        """Create a specialized prompt for each council member"""
        
        base_prompt = f"""You are {member.name}, serving as the {member.role} in an AI Council discussion.

YOUR ROLE: {member.role}
YOUR EXPERTISE: {member.expertise}
YOUR PERSONALITY: {member.personality}

QUESTION TO DISCUSS: {question}"""

        if context:
            base_prompt += f"\n\nCONTEXT: {context}"
        
        if round_number == 1:
            base_prompt += f"""

INSTRUCTIONS FOR ROUND 1:
- Provide your initial thoughts as part of a panel discussion
- Speak directly and conversationally, as if addressing fellow experts
- Focus on your area of expertise: {member.expertise}
- Be authentic to your personality: {member.personality}
- Keep it focused and engaging (max 150 words)
- Respond as if you're in a live discussion format

Begin with a conversational opener like "I think..." or "From my perspective..." or "What strikes me is..."
Avoid formal formatting - speak naturally as you would in a panel discussion."""

        else:
            previous_text = "\n".join(previous_discussion)
            base_prompt += f"""

PREVIOUS DISCUSSION:
{previous_text}

INSTRUCTIONS FOR ROUND {round_number}:
- Continue the panel discussion naturally
- Reference or respond to specific points made by other members
- Maintain your role as {member.role} and your personality
- You can agree, disagree, build upon, or challenge what was said
- Keep it conversational and engaging (max 150 words)
- Use phrases like "I agree with [Name] that..." or "However, I think..." or "Building on what was said..."

Respond naturally as if you're in a live panel discussion with experts."""
        
        return base_prompt
    
    async def _get_member_response(self, member: CouncilMember, prompt: str) -> str:
        """Get response from specific AI provider"""
        try:
            # Use enhanced_summarizer to get response from specific provider
            response = self.enhanced_summarizer.summarize(
                text=prompt,
                summary_type='custom',
                provider=member.provider,
                custom_prompt="Respond as the specified council member with their personality and expertise",
                max_tokens=300
            )
            
            # Ensure we never return None - fallback to a default response
            if response is None or response.strip() == "":
                print(f"[AI COUNCIL WARNING] Got empty response from {member.name}, using fallback")
                return f"I apologize, but I'm experiencing technical difficulties and cannot contribute to this discussion round."
            
            return response
            
        except Exception as e:
            print(f"[AI COUNCIL ERROR] Exception from {member.name}: {str(e)}")
            return f"Unable to participate: {str(e)}"
    
    def _extract_reasoning(self, response_text: str) -> str:
        """Extract reasoning section from response"""
        if response_text and "**Reasoning:**" in response_text:
            parts = response_text.split("**Reasoning:**")
            if len(parts) > 1:
                reasoning_part = parts[1].split("**")[0].strip()
                return reasoning_part
        
        # Fallback: try to extract some reasoning from the response
        if response_text and len(response_text) > 50:
            return response_text[:100] + "..."
        return "No reasoning provided"
    
    def _format_previous_discussion(self, discussion_rounds: List[List[CouncilResponse]]) -> List[str]:
        """Format previous discussion rounds for context"""
        formatted = []
        
        for round_num, responses in enumerate(discussion_rounds, 1):
            formatted.append(f"=== ROUND {round_num} ===")
            for response in responses:
                formatted.append(f"\n{response.member} ({response.role}):")
                formatted.append(response.response)
                formatted.append("")
        
        return formatted
    
    async def _build_consensus(self, session: CouncilSession) -> str:
        """Build consensus from all discussion rounds"""
        try:
            # Compile all responses
            all_responses = []
            for round_responses in session.discussion_rounds:
                all_responses.extend(round_responses)
            
            # Create consensus prompt - filter out empty responses
            discussion_text = "\n\n".join([
                f"{resp.member} ({resp.role}): {resp.response if resp.response else 'No response provided'}" 
                for resp in all_responses if resp.response and resp.response.strip()
            ])
            
            consensus_prompt = f"""Based on this AI Council discussion about: "{session.question}"

FULL DISCUSSION:
{discussion_text}

As a neutral facilitator, analyze this discussion and identify:
1. Areas of agreement between council members
2. Key disagreements or tensions
3. Most valuable insights that emerged
4. Common themes and patterns

Synthesize this into a clear consensus statement that captures:
- What the council generally agrees on
- Where there are productive disagreements
- The most important insights and recommendations

Keep the consensus balanced and acknowledgment different perspectives."""

            consensus = self.enhanced_summarizer.summarize(
                text=consensus_prompt,
                summary_type='custom',
                provider='openai',  # Use OpenAI for neutral consensus building
                custom_prompt="Create a balanced consensus from the AI council discussion",
                max_tokens=400
            )
            
            return consensus
            
        except Exception as e:
            return f"Error building consensus: {str(e)}"
    
    async def _generate_final_answer(self, session: CouncilSession) -> str:
        """Generate final comprehensive answer based on consensus"""
        try:
            final_prompt = f"""Question: {session.question}

Consensus from AI Council: {session.consensus}

Based on the comprehensive discussion and consensus, provide a final, actionable answer that:
1. Directly addresses the original question
2. Incorporates the best insights from the council discussion
3. Acknowledges different perspectives where relevant
4. Provides clear, practical guidance
5. Is well-structured and easy to understand

Make this a definitive answer that benefits from the collective intelligence of the AI council."""

            final_answer = self.enhanced_summarizer.summarize(
                text=final_prompt,
                summary_type='custom',
                provider='claude',  # Use Claude for final synthesis
                custom_prompt="Create a comprehensive final answer based on AI council consensus",
                max_tokens=500
            )
            
            return final_answer
            
        except Exception as e:
            return f"Error generating final answer: {str(e)}"
    
    async def _create_session_summary(self, session: CouncilSession) -> str:
        """Create a brief summary of the session"""
        member_names = [member.name for member in session.members]
        rounds_count = len(session.discussion_rounds)
        duration = (session.completed_at - session.started_at).total_seconds() if session.completed_at else 0
        
        return f"""AI Council Session Summary:
- Question: {session.question}
- Participants: {', '.join(member_names)}
- Discussion Rounds: {rounds_count}
- Duration: {duration:.1f} seconds
- Status: {'Completed' if session.completed_at else 'In Progress'}"""

    def get_session_history(self) -> List[CouncilSession]:
        """Get all previous council sessions"""
        return self.session_history
    
    def get_available_members(self) -> List[CouncilMember]:
        """Get list of available council members"""
        return self.council_members

# Test the system
if __name__ == "__main__":
    async def test_council():
        council = AICouncil()
        
        print("Available Council Members:")
        for member in council.get_available_members():
            print(f"- {member.name} ({member.role}) - {member.provider}")
        
        # Test session
        session = await council.hold_council_session(
            question="What are the most important considerations when implementing AI in a business?",
            context="Focus on practical, actionable advice for small to medium businesses."
        )
        
        print(f"\nFinal Answer: {session.final_answer}")
    
    # Run test
    asyncio.run(test_council())