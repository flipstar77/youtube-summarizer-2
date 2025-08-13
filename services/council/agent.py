"""
AI Council Agent
Individual agent that participates in council debates with structured responses
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .schemas import (
    AgentResponse, AgentPersona, DebatePhase, CouncilRound, 
    Claim, Evidence, TaskType
)
from .router import ModelRouter
from enhanced_summarizer import EnhancedSummarizer

logger = logging.getLogger(__name__)


class CouncilAgent:
    """Individual AI agent participating in council debates"""
    
    def __init__(self, persona: AgentPersona, summarizer: EnhancedSummarizer, router: ModelRouter):
        self.persona = persona
        self.summarizer = summarizer
        self.router = router
        
        logger.debug(f"[AGENT] Initialized {persona.name} ({persona.role})")
    
    async def generate_response(self,
                              question: str,
                              context: str,
                              phase: DebatePhase,
                              retrieval_context: List[Dict],
                              previous_rounds: List[CouncilRound] = None,
                              temperature: float = 0.7,
                              max_tokens: int = 500) -> Optional[AgentResponse]:
        """Generate structured response for debate phase"""
        
        try:
            # Select optimal provider for this agent/task
            task_type = self._determine_task_type(phase)
            provider = self.router.choose_provider(
                task=task_type,
                agent_persona=self.persona,
                budget="balanced",
                latency="balanced"
            )
            
            # Build phase-specific prompt
            prompt = self._build_prompt(
                question, context, phase, retrieval_context, previous_rounds
            )
            
            # Generate response
            raw_response = self.summarizer.summarize(
                text=prompt,
                summary_type="custom",
                provider=provider,
                custom_prompt="Respond as the specified agent with structured JSON output",
                max_tokens=max_tokens
            )
            
            # Parse structured response
            parsed_response = self._parse_response(raw_response, provider)
            
            logger.debug(f"[AGENT] {self.persona.name} generated response for {phase.value}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"[AGENT] {self.persona.name} failed to generate response: {e}")
            return self._create_error_response(str(e))
    
    def _determine_task_type(self, phase: DebatePhase) -> TaskType:
        """Determine task type based on debate phase and agent expertise"""
        
        phase_task_map = {
            DebatePhase.FRAMING: TaskType.STRUCTURING,
            DebatePhase.ARGUMENTS: TaskType.SYNTHESIS,
            DebatePhase.REBUTTAL: TaskType.CRITIQUE,
            DebatePhase.SYNTHESIS: TaskType.SYNTHESIS
        }
        
        base_task = phase_task_map.get(phase, TaskType.SYNTHESIS)
        
        # Override with agent's preferred task types
        if self.persona.task_types and base_task not in self.persona.task_types:
            return self.persona.task_types[0]
        
        return base_task
    
    def _build_prompt(self,
                     question: str,
                     context: str,
                     phase: DebatePhase,
                     retrieval_context: List[Dict],
                     previous_rounds: List[CouncilRound]) -> str:
        """Build comprehensive prompt for the agent"""
        
        # Base persona prompt
        base_prompt = self.persona.get_full_prompt(question, context, phase)
        
        # Add retrieval context
        context_section = self._format_retrieval_context(retrieval_context)
        
        # Add previous discussion context
        discussion_section = self._format_previous_discussion(previous_rounds, phase)
        
        # Phase-specific instructions
        phase_instructions = self._get_phase_instructions(phase, previous_rounds)
        
        # JSON output requirements
        json_schema = self._get_json_schema()
        
        full_prompt = f"""{base_prompt}

KNOWLEDGE BASE CONTEXT:
{context_section}

{discussion_section}

PHASE-SPECIFIC INSTRUCTIONS:
{phase_instructions}

OUTPUT FORMAT (REQUIRED):
You MUST respond with valid JSON matching this exact schema:
{json_schema}

All claims must include specific evidence citations with video_id and timestamps where available.
Provide thoughtful uncertainty estimates and identify key assumptions.
Suggest valuable follow-up questions that could deepen the analysis."""
        
        return full_prompt
    
    def _format_retrieval_context(self, retrieval_context: List[Dict]) -> str:
        """Format retrieval results for prompt"""
        
        if not retrieval_context:
            return "No specific knowledge base context available."
        
        context_items = []
        for i, item in enumerate(retrieval_context[:10]):  # Limit to top 10
            video_id = item.get('video_id', 'unknown')
            title = item.get('title', 'Untitled')
            summary = item.get('summary', item.get('content', ''))[:200]
            similarity = item.get('similarity', 0.0)
            
            context_items.append(f"""
[{i+1}] Video ID: {video_id}
Title: {title}
Content: {summary}...
Relevance: {similarity:.2f}""")
        
        return "\n".join(context_items)
    
    def _format_previous_discussion(self, previous_rounds: List[CouncilRound], current_phase: DebatePhase) -> str:
        """Format previous discussion for context"""
        
        if not previous_rounds:
            return ""
        
        if current_phase == DebatePhase.FRAMING:
            return ""  # No prior discussion to reference
        
        discussion_parts = ["PREVIOUS DISCUSSION:"]
        
        for round_obj in previous_rounds[-2:]:  # Last 2 rounds for context
            discussion_parts.append(f"\n--- {round_obj.phase.value.upper()} ROUND ---")
            
            for response in round_obj.agent_responses:
                if response.agent_name != self.persona.name:  # Don't include own responses
                    discussion_parts.append(f"""
{response.agent_name} ({response.role}):
- {response.answer[:150]}...
- Key claims: {'; '.join([c.text[:100] for c in response.claims[:2]])}
- Uncertainty: {response.uncertainty:.2f}""")
        
        return "\n".join(discussion_parts)
    
    def _get_phase_instructions(self, phase: DebatePhase, previous_rounds: List[CouncilRound]) -> str:
        """Get specific instructions for debate phase"""
        
        instructions = {
            DebatePhase.FRAMING: f"""
As {self.persona.name}, frame this question from your perspective as {self.persona.role}.
- Identify the key dimensions and constraints of the problem
- Highlight aspects most relevant to your expertise: {self.persona.expertise}
- Define any important terms or concepts
- Set boundaries for what can be reasonably answered with available evidence
- Provide 2-3 framing claims with citations""",
            
            DebatePhase.ARGUMENTS: f"""
As {self.persona.name}, provide your substantive arguments as {self.persona.role}.
- Present 2-3 discrete, evidence-backed claims
- Each claim must have specific citations (video_id + timestamps if available)
- Draw on your expertise in: {self.persona.expertise}
- Acknowledge uncertainty and key assumptions
- Build on or differentiate from previous arguments""",
            
            DebatePhase.REBUTTAL: f"""
As {self.persona.name}, engage critically with the strongest opposing argument.
- Identify the most challenging point raised by other agents
- Provide a thoughtful counter-argument or refinement
- Use your perspective as {self.persona.role} to offer unique insights
- Maintain intellectual honesty about the strengths of opposing views
- Suggest synthesis opportunities where possible""",
            
            DebatePhase.SYNTHESIS: f"""
As {self.persona.name}, help synthesize the discussion from your {self.persona.role} perspective.
- Integrate insights from the debate
- Identify areas of consensus and productive disagreement
- Highlight remaining uncertainties and research gaps
- Suggest practical next steps or applications"""
        }
        
        base_instruction = instructions.get(phase, "Participate in the debate with your expertise.")
        
        # Add rebuttal targets for rebuttal phase
        if phase == DebatePhase.REBUTTAL and previous_rounds:
            opposing_claims = self._identify_opposing_claims(previous_rounds)
            if opposing_claims:
                base_instruction += f"""

SPECIFIC CLAIMS TO ADDRESS:
{opposing_claims}

Choose the strongest opposing claim and provide a substantive response."""
        
        return base_instruction
    
    def _identify_opposing_claims(self, previous_rounds: List[CouncilRound]) -> str:
        """Identify claims from other agents to address in rebuttal"""
        
        claims = []
        for round_obj in previous_rounds:
            for response in round_obj.agent_responses:
                if response.agent_name != self.persona.name:
                    for claim in response.claims[:1]:  # Top claim from each agent
                        claims.append(f"- {response.agent_name}: {claim.text[:150]}")
        
        return "\n".join(claims[:3])  # Top 3 claims to consider
    
    def _get_json_schema(self) -> str:
        """Get required JSON schema for responses"""
        return """{
  "agent_name": "string",
  "role": "string", 
  "answer": "string (main response, 100-300 words)",
  "claims": [
    {
      "text": "string (specific claim statement)",
      "evidence": [
        {
          "video_id": "string",
          "start_s": number or null,
          "end_s": number or null,
          "snippet": "string (optional context)"
        }
      ],
      "confidence": number (0.0-1.0)
    }
  ],
  "uncertainty": number (0.0-1.0),
  "assumptions": ["string (key assumptions made)"],
  "next_questions": ["string (valuable follow-up questions)"]
}"""
    
    def _parse_response(self, raw_response: str, provider: str) -> AgentResponse:
        """Parse raw AI response into structured format"""
        
        try:
            # Try direct JSON parsing
            if raw_response.strip().startswith('{'):
                parsed = json.loads(raw_response)
                return self._create_structured_response(parsed, provider)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from text
        try:
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = raw_response[json_start:json_end]
                parsed = json.loads(json_text)
                return self._create_structured_response(parsed, provider)
        except:
            pass
        
        # Fallback to text parsing
        return self._fallback_parse_response(raw_response, provider)
    
    def _create_structured_response(self, parsed_data: Dict, provider: str) -> AgentResponse:
        """Create AgentResponse from parsed JSON"""
        
        # Process claims
        claims = []
        for claim_data in parsed_data.get('claims', []):
            evidence_list = []
            for evidence_data in claim_data.get('evidence', []):
                evidence = Evidence(
                    video_id=evidence_data.get('video_id', 'unknown'),
                    start_s=evidence_data.get('start_s'),
                    end_s=evidence_data.get('end_s'),
                    snippet=evidence_data.get('snippet')
                )
                evidence_list.append(evidence)
            
            if evidence_list:  # Only add claims with evidence
                claim = Claim(
                    text=claim_data.get('text', ''),
                    evidence=evidence_list,
                    confidence=claim_data.get('confidence', 0.5)
                )
                claims.append(claim)
        
        return AgentResponse(
            agent_name=self.persona.name,
            role=self.persona.role,
            answer=parsed_data.get('answer', ''),
            claims=claims,
            uncertainty=parsed_data.get('uncertainty', 0.5),
            assumptions=parsed_data.get('assumptions', []),
            next_questions=parsed_data.get('next_questions', []),
            provider=provider
        )
    
    def _fallback_parse_response(self, raw_response: str, provider: str) -> AgentResponse:
        """Fallback parsing when JSON parsing fails"""
        
        logger.warning(f"[AGENT] {self.persona.name} using fallback parsing")
        
        # Extract main answer (first paragraph or up to first bullet)
        lines = raw_response.split('\n')
        answer_lines = []
        for line in lines:
            if line.strip() and not line.startswith('-') and not line.startswith('*'):
                answer_lines.append(line.strip())
            else:
                break
        
        answer = ' '.join(answer_lines) if answer_lines else raw_response[:200]
        
        # Create minimal claim from the response
        claims = [Claim(
            text=answer[:100] + "..." if len(answer) > 100 else answer,
            evidence=[Evidence(video_id="context", snippet="Unable to parse structured evidence")],
            confidence=0.5
        )]
        
        return AgentResponse(
            agent_name=self.persona.name,
            role=self.persona.role,
            answer=answer,
            claims=claims,
            uncertainty=0.7,  # High uncertainty for unparsed responses
            assumptions=["Response parsing was incomplete"],
            next_questions=["What specific evidence supports this analysis?"],
            provider=provider
        )
    
    def _create_error_response(self, error_msg: str) -> AgentResponse:
        """Create error response when agent fails"""
        
        return AgentResponse(
            agent_name=self.persona.name,
            role=self.persona.role,
            answer=f"I apologize, but I encountered a technical issue: {error_msg[:100]}",
            claims=[Claim(
                text="Unable to provide structured analysis due to technical difficulties",
                evidence=[Evidence(video_id="error", snippet=error_msg[:100])],
                confidence=0.0
            )],
            uncertainty=1.0,
            assumptions=["Technical systems are functioning properly"],
            next_questions=["How can we resolve the technical issues preventing full analysis?"],
            provider="error"
        )
    
    def get_agent_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "name": self.persona.name,
            "role": self.persona.role,
            "expertise": self.persona.expertise,
            "personality": self.persona.personality,
            "preferred_provider": self.persona.preferred_provider,
            "task_types": [str(task) for task in self.persona.task_types]
        }