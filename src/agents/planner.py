"""
Planner agent: outputs structured JSON plan (no tool calls, no retrieval).
"""
import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .base import BaseAgent
from src.observability.logger import get_logger


class PlanStep(BaseModel):
    """Single step in execution plan."""
    step_id: int = Field(..., description="Sequential step number")
    description: str = Field(..., description="What this step accomplishes")
    type: str = Field(..., description="Step type: retrieval, tool, or synthesis")
    tool_name: str | None = Field(None, description="Tool to execute (if type is 'tool')")
    tool_params: Dict[str, Any] | None = Field(None, description="Parameters for tool")
    retrieval_query: str | None = Field(None, description="Query for retrieval (if type is 'retrieval')")


class RetrievalNeed(BaseModel):
    """Retrieval requirement."""
    query: str = Field(..., description="Search query for retrieval")
    purpose: str = Field(..., description="Why this information is needed")


class ExecutionPlan(BaseModel):
    """Structured execution plan."""
    steps: list[PlanStep] = Field(..., description="Ordered list of execution steps")
    retrieval_needs: list[RetrievalNeed] = Field(default_factory=list, description="List of information retrieval requirements")
    tools_needed: list[str] = Field(default_factory=list, description="List of tools required for execution")


class Planner(BaseAgent):
    """
    Planner agent that creates structured execution plans.
    No tool calls, no retrieval - only planning.
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        super().__init__(
            role="planner",
            config_path=config_path,
            prompt_path="prompts/planner.yaml"
        )
        self.json_parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        self._build_chain()
    
    def _build_chain(self):
        """Build the planning chain."""
        system_prompt = self.get_system_prompt()
        format_instructions = self.json_parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}\n\n{format_instructions}"),
            ("human", "{user_query}"),
        ])
        
        self.chain = prompt | self.llm | self.json_parser
    
    def plan(self, user_query: str, request_id: str) -> Dict[str, Any]:
        """
        Generate execution plan for user query.
        
        Args:
            user_query: User's input query
            request_id: Request identifier for logging
        
        Returns:
            Structured plan dictionary
        """
        self.logger.info(
            "Planning",
            request_id=request_id,
            query=user_query
        )
        
        try:
            result = self.chain.invoke({"user_query": user_query})
            
            # Convert Pydantic model to dict
            if isinstance(result, ExecutionPlan):
                plan_dict = result.model_dump()
            else:
                plan_dict = result
            
            self.logger.info(
                "Plan generated",
                request_id=request_id,
                steps_count=len(plan_dict.get("steps", [])),
                tools_needed=plan_dict.get("tools_needed", [])
            )
            
            return plan_dict
            
        except Exception as e:
            self.logger.error(
                "Planning failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            # Fallback: minimal plan
            return {
                "steps": [{
                    "step_id": 1,
                    "description": "Process query",
                    "type": "synthesis"
                }],
                "retrieval_needs": [],
                "tools_needed": []
            }
    
    def execute(self, user_query: str, request_id: str) -> Dict[str, Any]:
        """Alias for plan() to match base interface."""
        return self.plan(user_query, request_id)
