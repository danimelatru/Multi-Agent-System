from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm

# Define the output structure
class RouteQuery(BaseModel):
    """
    Schema to categorize user queries.
    """
    destination: Literal["technical", "billing", "general"] = Field(
        ..., 
        description="Select 'technical' for hardware/software issues, 'billing' for payments/refunds, or 'general' for casual chat."
    )

def build_router_agent():
    """
    Creates the routing chain that classifies user intent.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteQuery)
    
    system_prompt = """You are an expert triage system. 
    Analyze the user's input and strictly classify it into one of the allowed categories."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    return prompt | structured_llm