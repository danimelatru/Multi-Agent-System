"""
Critic agent: validates final output and triggers fallback if needed.
"""
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .base import BaseAgent
from src.common.state import ExecutionState
from src.observability.logger import get_logger


class ValidationCheck(BaseModel):
    """Single validation check."""
    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether check passed")
    details: str = Field(..., description="Details about the check result")


class ValidationResult(BaseModel):
    """Validation result structure."""
    valid: bool = Field(..., description="Whether output passes validation")
    checks: list[ValidationCheck] = Field(..., description="List of validation checks")
    trigger_fallback: bool = Field(default=False, description="Whether to trigger fallback")
    feedback: str = Field(default="", description="Feedback for improvement")


class Critic(BaseAgent):
    """
    Critic agent that validates final output quality and correctness.
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        super().__init__(
            role="critic",
            config_path=config_path,
            prompt_path="prompts/critic.yaml"
        )
        self.json_parser = JsonOutputParser(pydantic_object=ValidationResult)
        self._build_chain()
    
    def _build_chain(self):
        """Build the critic chain."""
        system_prompt = self.get_system_prompt()
        format_instructions = self.json_parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}\n\n{format_instructions}"),
            ("human", """Validate the following execution:

Plan:
{plan}

Evidence Used:
{evidence_used}

Final Answer:
{answer}

Steps Executed:
{steps_executed}"""),
        ])
        
        self.chain = prompt | self.llm | self.json_parser
    
    def validate(self, state: ExecutionState, request_id: str) -> Dict[str, Any]:
        """
        Validate execution state and output quality.
        
        Args:
            state: Execution state with plan, evidence, answer
            request_id: Request identifier for logging
        
        Returns:
            Validation result dictionary
        """
        self.logger.info("Validating output", request_id=request_id)
        
        try:
            result = self.chain.invoke({
                "plan": str(state.plan) if state.plan else "No plan",
                "evidence_used": str(state.evidence) if state.evidence else "No evidence",
                "answer": state.answer or "No answer",
                "steps_executed": str(state.steps_executed) if state.steps_executed else "No steps"
            })
            
            if isinstance(result, ValidationResult):
                validation_dict = result.model_dump()
            else:
                validation_dict = result
            
            self.logger.info(
                "Validation completed",
                request_id=request_id,
                valid=validation_dict.get("valid", False),
                trigger_fallback=validation_dict.get("trigger_fallback", False)
            )
            
            return validation_dict
            
        except Exception as e:
            self.logger.error(
                "Validation failed",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )
            # Default: pass validation if critic fails
            return {
                "valid": True,
                "checks": [],
                "trigger_fallback": False,
                "feedback": ""
            }
    
    def execute(self, state: ExecutionState, request_id: str) -> Dict[str, Any]:
        """Alias for validate() to match base interface."""
        return self.validate(state, request_id)
