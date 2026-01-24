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

    system_prompt = """You are an expert triage system. Classify the user's input into exactly one category:

- **billing**: Order status (e.g. "ORD-123", "ORD-456"), refunds, payments, invoices, order lookup. Any mention of "order", "ORD-", "refund", "payment", "billing" belongs here.
- **technical**: Error codes, hardware/software issues, device reset, drivers, troubleshooting. E.g. "error 101", "blue screen", "transaction failed" (as a technical error).
- **general**: Greetings, small talk, "hello", "how are you", or unclear/off-topic.

Examples:
- "What's the status of ORD-123?" -> billing
- "Refund status for ORD-456?" -> billing
- "How do I fix error 101?" -> technical
- "Hello" -> general

Strictly output one of: technical, billing, general."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    return prompt | structured_llm