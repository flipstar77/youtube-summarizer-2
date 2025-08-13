"""
Multi-Model Router for AI Council
Routes tasks to optimal AI providers based on task type and capabilities
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from .schemas import TaskType, AgentPersona
from enhanced_summarizer import EnhancedSummarizer

logger = logging.getLogger(__name__)


class ProviderCapability(str, Enum):
    """Provider capabilities for routing decisions"""
    LONG_CONTEXT = "long_context"
    REASONING = "reasoning" 
    CREATIVITY = "creativity"
    CODING = "coding"
    STRUCTURED_OUTPUT = "structured_output"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    REAL_TIME = "real_time"
    COST_EFFICIENT = "cost_efficient"


class ModelRouter:
    """Routes tasks to optimal AI providers"""
    
    def __init__(self, enhanced_summarizer: EnhancedSummarizer):
        self.summarizer = enhanced_summarizer
        self.available_providers = list(enhanced_summarizer.providers.keys())
        
        # Provider capabilities mapping
        self.capabilities = {
            'claude': [
                ProviderCapability.LONG_CONTEXT,
                ProviderCapability.CRITIQUE,
                ProviderCapability.SYNTHESIS,
                ProviderCapability.STRUCTURED_OUTPUT
            ],
            'deepseek': [
                ProviderCapability.REASONING,
                ProviderCapability.CODING,
                ProviderCapability.COST_EFFICIENT
            ],
            'openai': [
                ProviderCapability.STRUCTURED_OUTPUT,
                ProviderCapability.CREATIVITY,
                ProviderCapability.CODING
            ],
            'gemini': [
                ProviderCapability.CREATIVITY,
                ProviderCapability.REASONING,
                ProviderCapability.COST_EFFICIENT
            ],
            'grok': [
                ProviderCapability.CREATIVITY,
                ProviderCapability.CRITIQUE,
                ProviderCapability.REAL_TIME
            ],
            'perplexity': [
                ProviderCapability.REAL_TIME,
                ProviderCapability.STRUCTURED_OUTPUT
            ]
        }
        
        # Task to capability mapping
        self.task_capabilities = {
            TaskType.CRITIQUE: [ProviderCapability.CRITIQUE, ProviderCapability.LONG_CONTEXT],
            TaskType.SYNTHESIS: [ProviderCapability.SYNTHESIS, ProviderCapability.LONG_CONTEXT],
            TaskType.DECOMPOSE: [ProviderCapability.REASONING, ProviderCapability.STRUCTURED_OUTPUT],
            TaskType.MATH: [ProviderCapability.REASONING, ProviderCapability.STRUCTURED_OUTPUT],
            TaskType.CODING: [ProviderCapability.CODING, ProviderCapability.STRUCTURED_OUTPUT],
            TaskType.STRUCTURING: [ProviderCapability.STRUCTURED_OUTPUT, ProviderCapability.REASONING],
            TaskType.FRESHNESS: [ProviderCapability.REAL_TIME]
        }
        
        logger.info(f"[ROUTER] Initialized with providers: {self.available_providers}")
    
    def choose_provider(self, 
                       task: TaskType,
                       agent_persona: Optional[AgentPersona] = None,
                       budget: str = "balanced",  # "low", "balanced", "high"
                       latency: str = "balanced"  # "fast", "balanced", "slow"
                       ) -> str:
        """Choose optimal provider for task"""
        
        # Start with agent's preferred provider if available
        if agent_persona and agent_persona.preferred_provider in self.available_providers:
            preferred = agent_persona.preferred_provider
            if self._provider_supports_task(preferred, task):
                logger.debug(f"[ROUTER] Using preferred provider {preferred} for {task}")
                return preferred
        
        # Find providers that support the task
        suitable_providers = []
        required_caps = self.task_capabilities.get(task, [])
        
        for provider in self.available_providers:
            provider_caps = self.capabilities.get(provider, [])
            if any(cap in provider_caps for cap in required_caps):
                suitable_providers.append(provider)
        
        if not suitable_providers:
            # Fallback to OpenAI if available, else first available
            fallback = 'openai' if 'openai' in self.available_providers else self.available_providers[0]
            logger.warning(f"[ROUTER] No suitable provider for {task}, using fallback: {fallback}")
            return fallback
        
        # Apply budget and latency constraints
        provider = self._apply_constraints(suitable_providers, budget, latency, task)
        logger.debug(f"[ROUTER] Selected {provider} for {task} (budget: {budget}, latency: {latency})")
        return provider
    
    def _provider_supports_task(self, provider: str, task: TaskType) -> bool:
        """Check if provider supports task"""
        provider_caps = self.capabilities.get(provider, [])
        required_caps = self.task_capabilities.get(task, [])
        return any(cap in provider_caps for cap in required_caps)
    
    def _apply_constraints(self, 
                          providers: List[str], 
                          budget: str, 
                          latency: str, 
                          task: TaskType) -> str:
        """Apply budget and latency constraints to provider selection"""
        
        # Cost rankings (rough estimates)
        cost_ranking = {
            'openai': 3,  # Most expensive 
            'claude': 3,
            'gemini': 2,  # Medium cost
            'perplexity': 2,
            'deepseek': 1,  # Most cost-efficient
            'grok': 2
        }
        
        # Speed rankings (rough estimates)
        speed_ranking = {
            'grok': 3,     # Fastest
            'openai': 3,
            'deepseek': 3,
            'gemini': 2,   # Medium
            'perplexity': 2,
            'claude': 1    # Slower but thorough
        }
        
        # Apply budget constraint
        if budget == "low":
            providers = [p for p in providers if cost_ranking.get(p, 2) <= 2]
        elif budget == "high":
            # Prefer higher quality providers
            providers.sort(key=lambda p: cost_ranking.get(p, 2), reverse=True)
        
        # Apply latency constraint
        if latency == "fast":
            providers = [p for p in providers if speed_ranking.get(p, 2) >= 2]
            providers.sort(key=lambda p: speed_ranking.get(p, 2), reverse=True)
        elif latency == "slow":
            # Prefer more thorough providers
            providers.sort(key=lambda p: speed_ranking.get(p, 2))
        
        # Task-specific preferences
        task_preferences = {
            TaskType.CRITIQUE: ['claude', 'grok'],
            TaskType.SYNTHESIS: ['claude', 'openai'],
            TaskType.DECOMPOSE: ['deepseek', 'openai'],
            TaskType.MATH: ['deepseek', 'openai'],
            TaskType.CODING: ['deepseek', 'openai'],
            TaskType.STRUCTURING: ['openai', 'claude'],
            TaskType.FRESHNESS: ['perplexity', 'grok']
        }
        
        preferred = task_preferences.get(task, [])
        for pref in preferred:
            if pref in providers:
                return pref
        
        # Return first available after constraints
        return providers[0] if providers else self.available_providers[0]
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics"""
        return {
            'available_providers': self.available_providers,
            'capabilities': {p: list(caps) for p, caps in self.capabilities.items()},
            'task_mappings': {str(task): list(caps) for task, caps in self.task_capabilities.items()}
        }
    
    def validate_routing(self, sessions: List[Dict]) -> Dict:
        """Validate routing decisions across sessions"""
        routing_analysis = {
            'total_sessions': len(sessions),
            'provider_usage': {},
            'task_routing': {},
            'success_rates': {}
        }
        
        for session in sessions:
            for round_data in session.get('rounds', []):
                for response in round_data.get('agent_responses', []):
                    provider = response.get('provider', 'unknown')
                    
                    # Count provider usage
                    routing_analysis['provider_usage'][provider] = \
                        routing_analysis['provider_usage'].get(provider, 0) + 1
                    
                    # Track success (based on judge scores if available)
                    if 'judge_scores' in round_data:
                        for score in round_data['judge_scores']:
                            if score.get('agent') == response.get('agent_name'):
                                total_score = score.get('scores', {}).get('total', 0)
                                if provider not in routing_analysis['success_rates']:
                                    routing_analysis['success_rates'][provider] = []
                                routing_analysis['success_rates'][provider].append(total_score)
        
        # Calculate average success rates
        for provider, scores in routing_analysis['success_rates'].items():
            routing_analysis['success_rates'][provider] = {
                'average': sum(scores) / len(scores) if scores else 0,
                'count': len(scores)
            }
        
        return routing_analysis


def create_agent_personas() -> List[AgentPersona]:
    """Create default agent personas with task routing"""
    personas = [
        AgentPersona(
            name="Dr. Analysis",
            role="Chief Analyst", 
            expertise="Data analysis, logical reasoning, structured thinking",
            personality="Methodical, thorough, fact-focused",
            preferred_provider="openai",
            task_types=[TaskType.STRUCTURING, TaskType.DECOMPOSE],
            system_prompt="""You are Dr. Analysis, the Chief Analyst of the AI Council.
            Your role is to break down complex problems systematically and provide structured, 
            evidence-based analysis. You excel at organizing information and identifying patterns."""
        ),
        
        AgentPersona(
            name="Claude Critic",
            role="Critical Evaluator",
            expertise="Critical thinking, finding flaws, alternative perspectives", 
            personality="Skeptical, questioning, detail-oriented",
            preferred_provider="claude",
            task_types=[TaskType.CRITIQUE, TaskType.SYNTHESIS],
            system_prompt="""You are Claude Critic, the Critical Evaluator of the AI Council.
            Your mission is to identify weaknesses, challenge assumptions, and provide alternative 
            viewpoints. You ensure intellectual rigor and prevent groupthink."""
        ),
        
        AgentPersona(
            name="Gemini Synthesizer",
            role="Synthesis Specialist",
            expertise="Pattern recognition, creative connections, holistic view",
            personality="Creative, integrative, big-picture thinking", 
            preferred_provider="gemini",
            task_types=[TaskType.SYNTHESIS, TaskType.STRUCTURING],
            system_prompt="""You are Gemini Synthesizer, the Synthesis Specialist of the AI Council.
            You excel at connecting disparate ideas, finding creative solutions, and seeing the 
            bigger picture. You bring together different perspectives into coherent wholes."""
        ),
        
        AgentPersona(
            name="Perplexity Researcher",
            role="Research Specialist", 
            expertise="Current information, fact-checking, external knowledge",
            personality="Curious, thorough, evidence-based",
            preferred_provider="perplexity",
            task_types=[TaskType.FRESHNESS, TaskType.STRUCTURING],
            system_prompt="""You are Perplexity Researcher, the Research Specialist of the AI Council.
            You focus on grounding discussions in current, factual information and ensure 
            accuracy through thorough research and verification."""
        ),
        
        AgentPersona(
            name="Grok Challenger",
            role="Devil's Advocate",
            expertise="Challenging assumptions, unconventional thinking",
            personality="Provocative, unconventional, boundary-pushing",
            preferred_provider="grok", 
            task_types=[TaskType.CRITIQUE, TaskType.FRESHNESS],
            system_prompt="""You are Grok Challenger, the Devil's Advocate of the AI Council.
            Your job is to challenge conventional wisdom, push boundaries, and explore 
            unconventional approaches. You prevent stagnant thinking."""
        ),
        
        AgentPersona(
            name="DeepSeek Strategist", 
            role="Strategic Planner",
            expertise="Long-term thinking, strategic planning, implementation roadmaps",
            personality="Strategic, forward-thinking, practical",
            preferred_provider="deepseek",
            task_types=[TaskType.DECOMPOSE, TaskType.MATH, TaskType.CODING],
            system_prompt="""You are DeepSeek Strategist, the Strategic Planner of the AI Council.
            You focus on long-term implications, strategic planning, and practical implementation.
            You excel at reasoning through complex problems systematically."""
        )
    ]
    
    return personas