"""
AI Council Orchestrator
Main interface for Tree-of-Thought AI Council system
Coordinates moderator, agents, judges, and routing
"""

import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime

from .schemas import CouncilConfig, CouncilSession
from .moderator import CouncilModerator
from enhanced_summarizer import EnhancedSummarizer

logger = logging.getLogger(__name__)


class CouncilOrchestrator:
    """Main orchestrator for AI Council Tree-of-Thought debates"""
    
    def __init__(self, enhanced_summarizer: EnhancedSummarizer, vector_search=None):
        self.summarizer = enhanced_summarizer
        self.vector_search = vector_search
        self.moderator = CouncilModerator(enhanced_summarizer, vector_search)
        
        # Performance tracking
        self.metrics = {
            'total_sessions': 0,
            'successful_sessions': 0,
            'average_duration': 0.0,
            'provider_usage': {},
            'phase_completion_rates': {}
        }
        
        logger.info("[ORCHESTRATOR] AI Council system initialized")
    
    async def start_council_session(self,
                                  question: str,
                                  context: str = "",
                                  config_dict: Optional[Dict] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Start and run complete council session with streaming updates"""
        
        self.metrics['total_sessions'] += 1
        session_start = datetime.now()
        
        try:
            # Parse configuration
            config = CouncilConfig(**(config_dict or {}))
            
            # Validate session parameters
            if not question.strip():
                yield {"type": "error", "message": "Question cannot be empty"}
                return
            
            if len(question) > 1000:
                yield {"type": "warning", "message": "Question is very long, truncating..."}
                question = question[:1000]
            
            # Start session
            yield {"type": "status", "message": "Initializing AI Council...", "progress": 5}
            
            session_id = await self.moderator.start_session(question, context, config)
            
            yield {
                "type": "session_started", 
                "session_id": session_id,
                "config": config.dict(),
                "participants": list(self.moderator.personas.keys())
            }
            
            # Stream session execution
            session_completed = False
            async for update in self.moderator.run_session(session_id):
                yield update
                
                if update.get("type") == "session_complete":
                    session_completed = True
                    self.metrics['successful_sessions'] += 1
                    
                    # Update duration metrics
                    duration = (datetime.now() - session_start).total_seconds()
                    self._update_duration_metrics(duration)
                    
                elif update.get("type") == "round_complete":
                    # Track provider usage
                    round_data = update.get("round", {})
                    self._update_provider_metrics(round_data)
                    
                elif update.get("type") == "error":
                    logger.error(f"[ORCHESTRATOR] Session failed: {update.get('message')}")
                    break
            
            if session_completed:
                yield {"type": "status", "message": "Council session completed successfully!", "progress": 100}
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Session orchestration failed: {e}")
            yield {"type": "error", "message": f"Session failed: {str(e)}"}
    
    async def quick_council_query(self,
                                question: str,
                                context: str = "",
                                max_rounds: int = 2) -> Dict:
        """Quick council query with simplified output"""
        
        config = CouncilConfig(
            max_rounds=max_rounds,
            branching_factor=1,  # Single response per agent
            parallel_agents=True,
            stream_responses=False
        )
        
        result = {"question": question, "responses": [], "consensus": None, "error": None}
        
        try:
            async for update in self.start_council_session(question, context, config.dict()):
                if update.get("type") == "session_complete":
                    session_data = update.get("session", {})
                    result["consensus"] = session_data.get("final_consensus")
                    
                    # Extract top responses
                    for round_data in session_data.get("rounds", []):
                        for response in round_data.get("agent_responses", []):
                            result["responses"].append({
                                "agent": response.get("agent_name"),
                                "role": response.get("role"),
                                "answer": response.get("answer"),
                                "uncertainty": response.get("uncertainty"),
                                "claims": len(response.get("claims", []))
                            })
                    break
                elif update.get("type") == "error":
                    result["error"] = update.get("message")
                    break
                    
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get status of specific session"""
        return self.moderator.get_session_status(session_id)
    
    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get recent session history"""
        history = self.moderator.session_history[-limit:]
        
        return [{
            "session_id": session.session_id,
            "question": session.question[:100] + "..." if len(session.question) > 100 else session.question,
            "status": session.status,
            "started_at": session.started_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "duration": session.duration_seconds,
            "rounds": len(session.rounds),
            "participants": len(session.participants)
        } for session in history]
    
    def get_available_personas(self) -> List[Dict]:
        """Get available AI personas"""
        return [
            {
                "name": persona.name,
                "role": persona.role,
                "expertise": persona.expertise,
                "personality": persona.personality,
                "preferred_provider": persona.preferred_provider,
                "task_types": [str(task) for task in persona.task_types]
            }
            for persona in self.moderator.personas.values()
        ]
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        
        moderator_stats = self.moderator.get_moderator_stats()
        router_stats = self.moderator.router.get_routing_stats()
        
        return {
            "orchestrator_metrics": self.metrics,
            "moderator_stats": moderator_stats,
            "router_stats": router_stats,
            "available_providers": list(self.summarizer.providers.keys()),
            "system_health": self._get_system_health()
        }
    
    def _update_duration_metrics(self, duration: float):
        """Update duration tracking metrics"""
        if self.metrics['successful_sessions'] == 1:
            self.metrics['average_duration'] = duration
        else:
            # Running average
            prev_avg = self.metrics['average_duration']
            count = self.metrics['successful_sessions']
            self.metrics['average_duration'] = (prev_avg * (count - 1) + duration) / count
    
    def _update_provider_metrics(self, round_data: Dict):
        """Update provider usage metrics"""
        for response in round_data.get("agent_responses", []):
            provider = response.get("provider", "unknown")
            self.metrics['provider_usage'][provider] = \
                self.metrics['provider_usage'].get(provider, 0) + 1
    
    def _get_system_health(self) -> Dict:
        """Get system health indicators"""
        
        total_sessions = self.metrics['total_sessions']
        successful_sessions = self.metrics['successful_sessions']
        
        success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 100
        
        health = {
            "overall_health": "healthy" if success_rate >= 90 else "degraded" if success_rate >= 70 else "unhealthy",
            "success_rate": round(success_rate, 2),
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "average_session_duration": round(self.metrics['average_duration'], 2)
        }
        
        # Check provider availability
        available_providers = len(self.summarizer.providers)
        if available_providers < 3:
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append(f"Only {available_providers} AI providers available")
        
        # Check vector search
        if not self.vector_search:
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("Vector search not available - limited knowledge grounding")
        
        return health
    
    async def validate_system(self) -> Dict:
        """Validate system components"""
        
        validation_results = {
            "overall_status": "unknown",
            "component_status": {},
            "recommendations": []
        }
        
        try:
            # Test enhanced summarizer
            try:
                test_response = self.summarizer.summarize(
                    "Test prompt for validation",
                    summary_type="brief"
                )
                validation_results["component_status"]["summarizer"] = "healthy"
            except Exception as e:
                validation_results["component_status"]["summarizer"] = f"error: {str(e)}"
                validation_results["recommendations"].append("Check AI provider API keys")
            
            # Test vector search
            if self.vector_search:
                try:
                    # Simple test query
                    results = await self.vector_search.semantic_search("test", limit=1)
                    validation_results["component_status"]["vector_search"] = "healthy"
                except Exception as e:
                    validation_results["component_status"]["vector_search"] = f"error: {str(e)}"
                    validation_results["recommendations"].append("Check vector database connection")
            else:
                validation_results["component_status"]["vector_search"] = "not_configured"
                validation_results["recommendations"].append("Configure vector search for better knowledge grounding")
            
            # Test personas
            persona_count = len(self.moderator.personas)
            if persona_count >= 4:
                validation_results["component_status"]["personas"] = "healthy"
            else:
                validation_results["component_status"]["personas"] = f"limited: {persona_count} personas"
                validation_results["recommendations"].append("Ensure all AI providers are configured for full persona diversity")
            
            # Overall status
            healthy_components = sum(1 for status in validation_results["component_status"].values() 
                                   if status == "healthy")
            total_components = len(validation_results["component_status"])
            
            if healthy_components == total_components:
                validation_results["overall_status"] = "healthy"
            elif healthy_components >= total_components * 0.7:
                validation_results["overall_status"] = "partially_healthy"
            else:
                validation_results["overall_status"] = "unhealthy"
            
        except Exception as e:
            validation_results["overall_status"] = "error"
            validation_results["error"] = str(e)
        
        return validation_results
    
    async def run_test_session(self) -> Dict:
        """Run a test council session for validation"""
        
        test_question = "What are the key benefits and risks of artificial intelligence in education?"
        test_context = "Consider both current applications and future possibilities"
        
        test_config = CouncilConfig(
            max_rounds=2,
            branching_factor=1,
            max_tokens_per_agent=300,
            parallel_agents=True,
            save_session=False
        )
        
        test_results = {
            "question": test_question,
            "status": "unknown",
            "duration_seconds": 0,
            "responses_generated": 0,
            "consensus_generated": False,
            "errors": []
        }
        
        start_time = datetime.now()
        
        try:
            async for update in self.start_council_session(
                test_question, test_context, test_config.dict()
            ):
                if update.get("type") == "session_complete":
                    test_results["status"] = "success"
                    test_results["consensus_generated"] = update.get("session", {}).get("final_consensus") is not None
                    
                    # Count responses
                    session_data = update.get("session", {})
                    for round_data in session_data.get("rounds", []):
                        test_results["responses_generated"] += len(round_data.get("agent_responses", []))
                    
                    break
                elif update.get("type") == "error":
                    test_results["status"] = "failed"
                    test_results["errors"].append(update.get("message"))
                    break
        
        except Exception as e:
            test_results["status"] = "error"
            test_results["errors"].append(str(e))
        
        test_results["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        
        return test_results


# Global orchestrator instance
_orchestrator_instance = None


def get_orchestrator(enhanced_summarizer: EnhancedSummarizer = None, 
                    vector_search=None) -> CouncilOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        if enhanced_summarizer is None:
            raise ValueError("enhanced_summarizer required for first initialization")
        _orchestrator_instance = CouncilOrchestrator(enhanced_summarizer, vector_search)
    
    return _orchestrator_instance


def reset_orchestrator():
    """Reset global orchestrator instance (for testing)"""
    global _orchestrator_instance
    _orchestrator_instance = None