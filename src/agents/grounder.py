"""
Grounder/Perceiver agent: performs RAG and returns evidence only.
"""
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .base import BaseAgent
from src.retrieval import Retriever
from src.observability.logger import get_logger


class EvidenceItem(BaseModel):
    """Single evidence item."""
    doc_id: str = Field(..., description="Unique document/chunk identifier")
    excerpt: str = Field(..., description="Relevant text excerpt from document")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(..., description="Source document name/path")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GroundedEvidence(BaseModel):
    """Structured evidence output."""
    evidence: List[EvidenceItem] = Field(..., description="List of retrieved evidence items")


class Grounder(BaseAgent):
    """
    Grounder/Perceiver agent that performs RAG and returns structured evidence.
    No answer generation, only evidence retrieval.
    """
    
    def __init__(
        self,
        retriever: Retriever,
        config_path: str = "config/models.yaml"
    ):
        super().__init__(
            role="grounder",
            config_path=config_path,
            prompt_path="prompts/grounder.yaml"
        )
        self.retriever = retriever
        self.json_parser = JsonOutputParser(pydantic_object=GroundedEvidence)
        self._build_chain()
    
    def _build_chain(self):
        """Build the grounding chain."""
        system_prompt = self.get_system_prompt()
        format_instructions = self.json_parser.get_format_instructions()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\n{format_instructions}"),
            ("human", "Retrieve evidence for the following retrieval needs:\n{retrieval_needs}"),
        ])
        
        self.chain = prompt.partial(system_prompt=system_prompt, format_instructions=format_instructions) | self.llm | self.json_parser
    
    def retrieve(self, retrieval_needs: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve evidence based on retrieval needs from plan.
        
        Args:
            retrieval_needs: List of retrieval requirements from planner
            request_id: Request identifier for logging
        
        Returns:
            List of evidence items with doc_id, excerpt, confidence, source
        """
        self.logger.info(
            "Grounding",
            request_id=request_id,
            retrieval_needs_count=len(retrieval_needs)
        )
        
        # Extract queries from retrieval needs
        queries = [need.get("query", "") for need in retrieval_needs if need.get("query")]
        
        if not queries:
            self.logger.warning("No queries to retrieve", request_id=request_id)
            return []
        
        # Perform retrieval
        evidence = self.retriever.retrieve(queries)
        
        # Format as structured evidence
        evidence_items = []
        for item in evidence:
            evidence_items.append({
                "doc_id": item["doc_id"],
                "excerpt": item["excerpt"],
                "confidence": item["confidence"],
                "source": item["source"],
                "metadata": item.get("metadata", {})
            })
        
        self.logger.info(
            "Grounding completed",
            request_id=request_id,
            evidence_count=len(evidence_items)
        )
        
        return evidence_items
    
    def execute(self, retrieval_needs: List[Dict[str, Any]], request_id: str) -> List[Dict[str, Any]]:
        """Alias for retrieve() to match base interface."""
        return self.retrieve(retrieval_needs, request_id)
