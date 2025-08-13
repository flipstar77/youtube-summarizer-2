"""
AI Council Schemas - Pydantic models for Tree-of-Thought debate system
Defines structured JSON contracts for agents, judges, and moderators
"""

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    """Types of tasks for multi-model routing"""
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    DECOMPOSE = "decompose" 
    MATH = "math"
    CODING = "coding"
    STRUCTURING = "structuring"
    FRESHNESS = "freshness_check"


class Evidence(BaseModel):
    """Citation evidence from knowledge base"""
    video_id: str = Field(..., description="Video ID from knowledge base")
    start_s: Optional[float] = Field(None, description="Start timestamp in seconds")
    end_s: Optional[float] = Field(None, description="End timestamp in seconds")
    snippet: Optional[str] = Field(None, description="Text snippet for context")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score")


class Claim(BaseModel):
    """Individual claim with supporting evidence"""
    text: str = Field(..., description="The claim statement")
    evidence: List[Evidence] = Field(default_factory=list, description="Supporting citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in claim")
    
    @validator('evidence')
    def evidence_required(cls, v, values):
        """Enforce that claims must have evidence"""
        if not v:
            raise ValueError("Claims must include at least one evidence citation")
        return v


class AgentResponse(BaseModel):
    """Structured response from council agent"""
    agent_name: str = Field(..., description="Name of responding agent")
    role: str = Field(..., description="Agent's role/expertise")
    answer: str = Field(..., description="Main response text")
    claims: List[Claim] = Field(default_factory=list, description="Discrete claims with evidence")
    uncertainty: float = Field(..., ge=0.0, le=1.0, description="Overall uncertainty level")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions made")
    next_questions: List[str] = Field(default_factory=list, description="Follow-up questions to explore")
    provider: str = Field(..., description="AI provider used for this response")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('claims')
    def minimum_claims(cls, v):
        """Ensure agents provide meaningful claims"""
        if len(v) < 1:
            raise ValueError("Agents must provide at least 1 claim")
        return v


class ScoreBreakdown(BaseModel):
    """Detailed scoring for agent response"""
    grounded: int = Field(..., ge=0, le=10, description="Groundedness in evidence")
    coherence: int = Field(..., ge=0, le=10, description="Logical coherence")
    novelty: int = Field(..., ge=0, le=10, description="Novel insights provided")
    total: Optional[int] = Field(None, description="Computed total score")
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.total is None:
            self.total = self.grounded + self.coherence + self.novelty


class JudgeScore(BaseModel):
    """Judge evaluation of agent response"""
    agent: str = Field(..., description="Agent being scored")
    scores: ScoreBreakdown = Field(..., description="Detailed scores")
    notes: str = Field(..., description="Reasoning for scores")
    claim_scores: List[Dict[str, Union[int, str]]] = Field(default_factory=list, description="Per-claim analysis")
    timestamp: datetime = Field(default_factory=datetime.now)


class DebatePhase(str, Enum):
    """Phases of the council debate"""
    FRAMING = "framing"
    ARGUMENTS = "arguments" 
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"


class CouncilRound(BaseModel):
    """Single round of council debate"""
    phase: DebatePhase = Field(..., description="Current debate phase")
    round_number: int = Field(..., ge=1, description="Round number")
    question: str = Field(..., description="Original question")
    context: Optional[str] = Field(None, description="Additional context")
    retrieval_results: List[Dict] = Field(default_factory=list, description="Retrieved knowledge chunks")
    agent_responses: List[AgentResponse] = Field(default_factory=list)
    judge_scores: List[JudgeScore] = Field(default_factory=list)
    moderator_notes: Optional[str] = Field(None, description="Moderator observations")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class CouncilConsensus(BaseModel):
    """Final consensus output"""
    consensus: List[str] = Field(..., max_items=3, description="Areas of agreement")
    disagreements: List[str] = Field(..., max_items=2, description="Key disagreements")
    actionables: List[str] = Field(..., max_items=3, description="Actionable next steps")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence level")
    evidence_summary: List[Evidence] = Field(default_factory=list, description="Key supporting evidence")
    
    @validator('consensus', 'disagreements', 'actionables')
    def non_empty_lists(cls, v):
        if not v:
            raise ValueError("Consensus sections cannot be empty")
        return v


class CouncilSession(BaseModel):
    """Complete council debate session"""
    session_id: str = Field(..., description="Unique session identifier")
    question: str = Field(..., description="Original question")
    context: Optional[str] = Field(None, description="Additional context provided")
    config: Dict = Field(default_factory=dict, description="Session configuration")
    rounds: List[CouncilRound] = Field(default_factory=list)
    final_consensus: Optional[CouncilConsensus] = None
    participants: List[str] = Field(default_factory=list, description="Agent names participating")
    total_tokens: Optional[int] = Field(None, description="Total tokens used")
    total_cost: Optional[float] = Field(None, description="Estimated cost")
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = Field(default="running", description="Session status")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate session duration"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class CouncilConfig(BaseModel):
    """Configuration for council session"""
    max_rounds: int = Field(default=3, ge=1, le=5, description="Maximum debate rounds")
    branching_factor: int = Field(default=2, ge=1, le=4, description="Response variants per agent")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Response creativity")
    max_tokens_per_agent: int = Field(default=500, ge=100, le=1000, description="Token limit per response")
    max_total_tokens: int = Field(default=10000, ge=1000, le=50000, description="Session token limit")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Evidence grounding threshold")
    require_citations: bool = Field(default=True, description="Enforce evidence citations")
    parallel_agents: bool = Field(default=True, description="Run agents in parallel")
    save_session: bool = Field(default=True, description="Save to session history")
    enable_tts: bool = Field(default=False, description="Enable text-to-speech")
    stream_responses: bool = Field(default=True, description="Stream live responses")


class RetrievalQuery(BaseModel):
    """Query for knowledge base retrieval"""
    query: str = Field(..., description="Search query")
    agent_name: Optional[str] = Field(None, description="Agent making the query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[Dict] = Field(None, description="Additional filters")


class AgentPersona(BaseModel):
    """Agent persona definition"""
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Primary role")
    expertise: str = Field(..., description="Area of expertise")
    personality: str = Field(..., description="Personality traits")
    preferred_provider: str = Field(..., description="Preferred AI provider")
    task_types: List[TaskType] = Field(default_factory=list, description="Suitable task types")
    system_prompt: str = Field(..., description="Base system prompt")
    voice_settings: Optional[Dict] = Field(None, description="TTS voice configuration")
    
    def get_full_prompt(self, question: str, context: str, phase: DebatePhase) -> str:
        """Generate complete prompt for agent"""
        phase_instructions = {
            DebatePhase.FRAMING: "Frame the question and identify key constraints.",
            DebatePhase.ARGUMENTS: "Provide 2-3 discrete claims with citations.",
            DebatePhase.REBUTTAL: "Address the strongest opposing claim.",
            DebatePhase.SYNTHESIS: "Synthesize insights across perspectives."
        }
        
        return f"""{self.system_prompt}

DEBATE PHASE: {phase.value.upper()}
TASK: {phase_instructions.get(phase, "Participate in debate")}

QUESTION: {question}
CONTEXT: {context}

You must output valid JSON with: answer, claims, uncertainty, assumptions, next_questions.
All claims require specific evidence citations with video_id and timestamps."""