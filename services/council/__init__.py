"""
AI Council - Tree-of-Thought Multi-Model Debate System

A sophisticated AI debate system that orchestrates multiple AI models
to engage in structured, evidence-based discussions using Tree-of-Thought methodology.

Key Features:
- Multi-model routing for optimal provider selection
- Structured debate phases (Framing → Arguments → Rebuttal → Synthesis)
- Evidence-based grounding with citation requirements
- Tree-of-Thought branching and pruning
- Real-time scoring and quality assessment
- Streaming session management

Usage:
    from services.council import get_orchestrator, CouncilConfig
    
    orchestrator = get_orchestrator(enhanced_summarizer, vector_search)
    
    async for update in orchestrator.start_council_session(
        question="Your complex question here",
        context="Additional context",
        config_dict={"max_rounds": 3, "branching_factor": 2}
    ):
        print(update)
"""

from .schemas import (
    CouncilConfig,
    CouncilSession, 
    CouncilRound,
    CouncilConsensus,
    AgentResponse,
    AgentPersona,
    JudgeScore,
    DebatePhase,
    TaskType
)

from .orchestrator import (
    CouncilOrchestrator,
    get_orchestrator,
    reset_orchestrator
)

from .moderator import CouncilModerator
from .agent import CouncilAgent
from .judge import CouncilJudge
from .router import ModelRouter, create_agent_personas

__version__ = "1.0.0"
__author__ = "AI Council Development Team"

__all__ = [
    # Main interfaces
    'get_orchestrator',
    'CouncilOrchestrator',
    'CouncilConfig',
    
    # Core components
    'CouncilModerator',
    'CouncilAgent', 
    'CouncilJudge',
    'ModelRouter',
    
    # Data models
    'CouncilSession',
    'CouncilRound',
    'CouncilConsensus',
    'AgentResponse',
    'AgentPersona', 
    'JudgeScore',
    
    # Enums
    'DebatePhase',
    'TaskType',
    
    # Utilities
    'create_agent_personas',
    'reset_orchestrator'
]