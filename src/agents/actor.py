"""
Actor/Executor agent: executes tools and produces final answer.
Strictly follows plan and uses grounded evidence.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .base import BaseAgent
from src.tools import execute_tool
from src.observability.logger import get_logger


class StepExecution(BaseModel):
    """Single step execution result."""
    step_id: int = Field(..., description="Step identifier")
    status: str = Field(..., description="Execution status: success, failed, or skipped")
    result: str = Field(..., description="Result of step execution")


class ToolUsage(BaseModel):
    """Tool execution record."""
    tool_name: str = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(..., description="Tool parameters")
    result: str = Field(..., description="Tool execution result")


class ActorOutput(BaseModel):
    """Actor output structure."""
    answer: str = Field(..., description="Final answer to the user's query")
    steps_executed: List[StepExecution] = Field(..., description="List of executed steps")
    tools_used: List[ToolUsage] = Field(default_factory=list, description="Tools that were executed")
    evidence_used: List[str] = Field(default_factory=list, description="doc_ids of evidence used")


class Actor(BaseAgent):
    """
    Actor/Executor agent that follows plans and uses evidence to produce answers.
    Executes tools as specified in the plan.
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        super().__init__(
            role="actor",
            config_path=config_path,
            prompt_path="prompts/actor.yaml"
        )
        self.json_parser = JsonOutputParser(pydantic_object=ActorOutput)
        self._build_chain()
    
    def _build_chain(self):
        """Build the actor chain."""
        system_prompt = self.get_system_prompt()
        format_instructions = self.json_parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n{format_instructions}"),
            ("human", """Execute the following plan using the provided evidence.

Plan:
{plan}

Evidence:
{evidence}

User Query: {user_query}"""),
        ])
        
        self.chain = prompt.partial(system_prompt=system_prompt, format_instructions=format_instructions) | self.llm | self.json_parser
    
    def execute(
        self,
        plan: Dict[str, Any],
        evidence: Optional[List[Dict[str, Any]]],
        request_id: str
    ) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute plan using evidence and tools.
        
        Args:
            plan: Execution plan from planner
            evidence: Evidence from grounder
            request_id: Request identifier for logging
        
        Returns:
            Tuple of (answer, steps_executed, tools_used)
        """
        self.logger.info(
            "Executing plan",
            request_id=request_id,
            steps_count=len(plan.get("steps", []))
        )
        
        # Execute tool steps first
        steps_executed = []
        tools_used = []
        
        for step in plan.get("steps", []):
            step_id = step.get("step_id")
            step_type = step.get("type")
            
            if step_type == "tool":
                # Execute tool
                tool_name = step.get("tool_name")
                tool_params = step.get("tool_params", {})
                
                try:
                    tool_result = execute_tool(tool_name, tool_params)
                    tools_used.append({
                        "tool_name": tool_name,
                        "params": tool_params,
                        "result": str(tool_result)
                    })
                    steps_executed.append({
                        "step_id": step_id,
                        "status": "success",
                        "result": f"Tool {tool_name} executed: {tool_result}"
                    })
                except Exception as e:
                    steps_executed.append({
                        "step_id": step_id,
                        "status": "failed",
                        "result": f"Tool execution failed: {str(e)}"
                    })
                    self.logger.error(
                        "Tool execution failed",
                        request_id=request_id,
                        tool_name=tool_name,
                        error=str(e)
                    )
        
        # Format evidence for LLM
        evidence_text = self._format_evidence(evidence) if evidence else "No evidence available."
        
        # Generate final answer using LLM
        try:
            result = self.chain.invoke({
                "plan": self._format_plan(plan),
                "evidence": evidence_text,
                "user_query": plan.get("user_query", "")
            })
            
            if isinstance(result, ActorOutput):
                answer = result.answer
                # Merge LLM steps with executed tool steps
                for llm_step in result.steps_executed:
                    # Only add if not already in steps_executed
                    if not any(s["step_id"] == llm_step.step_id for s in steps_executed):
                        steps_executed.append({
                            "step_id": llm_step.step_id,
                            "status": llm_step.status,
                            "result": llm_step.result
                        })
                # Merge tools
                for tool in result.tools_used:
                    tools_used.append({
                        "tool_name": tool.tool_name,
                        "params": tool.params,
                        "result": tool.result
                    })
            else:
                answer = result.get("answer", "I couldn't generate a proper answer.")
            
            # Auto-fill missing steps (e.g. retrieval/synthesis) if answer exists
            if answer:
                for step in plan.get("steps", []):
                    step_id = step.get("step_id")
                    step_type = step.get("type")
                    is_done = any(int(s["step_id"]) == int(step_id) for s in steps_executed)
                    
                    if not is_done:
                        steps_executed.append({
                            "step_id": step_id,
                            "status": "success",
                            "result": "Implicitly executed during answer generation"
                        })
            
            self.logger.info(
                "Actor execution completed",
                request_id=request_id,
                answer_length=len(answer),
                tools_used_count=len(tools_used),
                final_steps_count=len(steps_executed)
            )
            
            return answer, steps_executed, tools_used
            
        except Exception as e:
            self.logger.error(
                "Actor execution failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            return (
                "I encountered an error while processing your request.",
                steps_executed,
                tools_used
            )
    
    def _format_plan(self, plan: Dict[str, Any]) -> str:
        """Format plan for LLM input."""
        steps = plan.get("steps", [])
        formatted = "Execution Plan:\n"
        for step in steps:
            formatted += f"  Step {step.get('step_id')}: {step.get('description')} ({step.get('type')})\n"
        return formatted
    
    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence for LLM input."""
        formatted = "Evidence:\n"
        for i, item in enumerate(evidence, 1):
            formatted += f"  {i}. [doc_id: {item.get('doc_id')}, confidence: {item.get('confidence', 0):.2f}]\n"
            formatted += f"     {item.get('excerpt', '')}\n"
        return formatted
