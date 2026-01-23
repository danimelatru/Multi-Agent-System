from langchain_core.tools import tool
from src.config import get_llm

# Define the Tool
@tool
def get_refund_status(order_id: str):
    """
    Queries the database for the refund status of a specific order ID.
    """
    # Mock Database
    mock_db = {
        "ORD-123": "Refund Processed",
        "ORD-456": "Pending Manager Approval",
        "ORD-999": "Rejected: Item damaged by user"
    }
    return mock_db.get(order_id, "Order ID not found in system.")

def build_billing_agent():
    """
    Returns the LLM binded with the billing tools.
    """
    llm = get_llm()
    tools = [get_refund_status]
    # Bind tools so the LLM knows it can call them
    return llm.bind_tools(tools)