"""
Orchestrator for planner-grounder-actor architecture.
Coordinates the flow between agents without shared global state.
"""
import json
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.common.state import ExecutionState
from src.agents.planner import Planner
from src.agents.grounder import Grounder
from src.agents.actor import Actor
from src.observability.logger import get_logger
from src.observability.tracing import TraceContext

# Type hint for Critic (lazy import to avoid circular dependency)
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.agents.critic import Critic

class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        grounder: Grounder,
        actor: Actor,
        critic: Optional["Critic"] = None,
        enable_critic: bool = True
    ):
        self.planner = planner
        self.grounder = grounder
        self.actor = actor
        self.critic = critic if enable_critic else None
        self.logger = get_logger("orchestrator")
    
    def execute(self, user_query: str, trace_context: Optional[TraceContext] = None) -> ExecutionState:
        """
        Execute the full planner-grounder-actor flow.
        
        Args:
            user_query: User's input query
            trace_context: Optional tracing context for observability
        
        Returns:
            ExecutionState with final answer and execution details
        """
        request_id = str(uuid.uuid4())
        state = ExecutionState(request_id=request_id, user_query=user_query)
        
        if trace_context:
            state.metadata["trace_id"] = trace_context.trace_id
        
        self.logger.info(
            "Starting execution",
            request_id=request_id,
            query=user_query
        )
        
        try:
            # Phase 1: Planning
            state.plan = self._plan(user_query, state)
            
            # Phase 2: Grounding (if retrieval needed)
            if state.plan and state.plan.get("retrieval_needs"):
                state.evidence = self._ground(state.plan["retrieval_needs"], state)
            
            # Phase 3: Acting
            state.answer, state.steps_executed, state.tools_used = self._act(
                user_query, state.plan, state.evidence, state
            )
            
            # Phase 4: Critic (optional)
            if self.critic:
                state.validation_result = self._criticize(state)
                if state.validation_result.get("trigger_fallback"):
                    self.logger.warning(
                        "Critic triggered fallback",
                        request_id=request_id,
                        feedback=state.validation_result.get("feedback")
                    )
                    # Fallback: simple response
                    state.answer = self._fallback_response(user_query)
            
            self.logger.info(
                "Execution completed",
                request_id=request_id,
                answer_length=len(state.answer) if state.answer else 0
            )
            
        except Exception as e:
            self.logger.error(
                "Execution failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            state.answer = self._fallback_response(user_query)
            state.metadata["error"] = str(e)
        
        return state
    
    def _plan(self, user_query: str, state: ExecutionState) -> Dict[str, Any]:
        """Execute planner agent."""
        self.logger.debug("Executing planner", request_id=state.request_id)
        plan = self.planner.plan(user_query, state.request_id)
        self.logger.info(
            "Plan generated",
            request_id=state.request_id,
            steps_count=len(plan.get("steps", [])),
            tools_needed=plan.get("tools_needed", [])
        )
        return plan
    
    def _ground(self, retrieval_needs: List[Dict[str, Any]], state: ExecutionState) -> List[Dict[str, Any]]:
        """Execute grounder agent."""
        self.logger.debug("Executing grounder", request_id=state.request_id)
        evidence = self.grounder.retrieve(retrieval_needs, state.request_id)
        self.logger.info(
            "Evidence retrieved",
            request_id=state.request_id,
            evidence_count=len(evidence)
        )
        return evidence
    
    def _act(
        self,
        user_query: str,
        plan: Dict[str, Any],
        evidence: Optional[List[Dict[str, Any]]],
        state: ExecutionState
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute actor agent."""
        self.logger.debug("Executing actor", request_id=state.request_id)
        # Store user_query in plan for actor
        plan_with_query = {**plan, "user_query": user_query}
        answer, steps_executed, tools_used = self.actor.execute(
            plan_with_query, evidence, state.request_id
        )
        self.logger.info(
            "Actor completed",
            request_id=state.request_id,
            steps_executed=len(steps_executed),
            tools_used_count=len(tools_used)
        )
        return answer, steps_executed, tools_used
    
    def _criticize(self, state: ExecutionState) -> Dict[str, Any]:
        self.logger.debug("Executing critic", request_id=state.request_id)
        validation = self.critic.validate(state, state.request_id)
        self.logger.info(
            "Critic validation",
            request_id=state.request_id,
            valid=validation.get("valid", False)
        )
        return validation
    
    def _fallback_response(self, user_query: str) -> str:
        """Fallback response when execution fails."""
        return (
            "I apologize, but I encountered an error processing your request. "
            "Please try rephrasing your question or contact support."
        )
