import sys
import os
import uuid

# Add the project root to the python path to allow imports from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END, START
from src.agents.router import build_router_agent
from src.agents.technical import build_technical_agent
from src.agents.billing import build_billing_agent, get_refund_status, init_billing_db
from src.utils.logging import get_logger

# --- 1. Define State ---
class AgentState(TypedDict):
    question: str
    category: str
    answer: str
    request_id: Optional[str]  # Added for observability

# --- 2. Initialize System ---
logger = get_logger("system")
logger.info("Initializing system...")

# Initialize billing database explicitly (not at import time)
logger.info("Initializing billing database...")
init_billing_db()
logger.info("Billing database initialized")

# Initialize Agents
logger.info("Initializing agents...")
router_chain = build_router_agent()
rag_chain = build_technical_agent()
billing_llm = build_billing_agent()
logger.info("Agents initialized")

# --- 3. Define Nodes ---
router_logger = get_logger("router")
technical_logger = get_logger("technical")
billing_logger = get_logger("billing")

def router_node(state: AgentState):
    """Analyzes the input and updates the category."""
    request_id = state.get("request_id", "unknown")
    router_logger.info(
        "Analyzing query",
        request_id=request_id,
        question=state["question"]
    )
    result = router_chain.invoke({"question": state["question"]})
    router_logger.info(
        "Routing decision",
        request_id=request_id,
        destination=result.destination.upper()
    )
    return {"category": result.destination}

def technical_node(state: AgentState):
    """Handles technical queries using RAG."""
    request_id = state.get("request_id", "unknown")
    technical_logger.info(
        "Searching knowledge base",
        request_id=request_id
    )
    response = rag_chain.invoke(state["question"])
    technical_logger.debug(
        "RAG response generated",
        request_id=request_id,
        response_length=len(response)
    )
    return {"answer": response}

def billing_node(state: AgentState):
    """Handles billing queries using Tools."""
    request_id = state.get("request_id", "unknown")
    billing_logger.info(
        "Processing billing query",
        request_id=request_id
    )
    
    # Invoke LLM with tools
    msg = billing_llm.invoke(state["question"])
    
    # Check if the LLM wants to call a tool
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call["name"] == "get_refund_status":
                order_id = tool_call["args"].get("order_id", "unknown")
                billing_logger.info(
                    "Calling tool",
                    request_id=request_id,
                    tool="get_refund_status",
                    order_id=order_id
                )
                # Execute tool
                tool_result = get_refund_status.invoke(tool_call["args"])
                billing_logger.info(
                    "Tool result",
                    request_id=request_id,
                    tool="get_refund_status",
                    result=tool_result
                )
                return {"answer": f"System Update: {tool_result}"}
    
    # If no tool called, return the text
    return {"answer": msg.content}

def general_node(state: AgentState):
    """Handles general chit-chat."""
    return {"answer": "Hello! I am the MA-System Support Bot. I can help with Technical Issues or Billing inquiries."}

# --- 4. Build Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("technical", technical_node)
workflow.add_node("billing", billing_node)
workflow.add_node("general", general_node)

# Add Edges
workflow.add_edge(START, "router")

# Conditional Routing Logic
def get_next_node(state):
    return state["category"]

workflow.add_conditional_edges(
    "router",
    get_next_node,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general"
    }
)

workflow.add_edge("technical", END)
workflow.add_edge("billing", END)
workflow.add_edge("general", END)

# Compile
app = workflow.compile()

# --- 5. Main Execution Loop ---
if __name__ == "__main__":
    logger.info("MA-SYSTEM ONLINE")
    logger.info("Type 'exit' or 'quit' to stop.")
    
    while True:
        # Get user input from terminal
        request_id = None
        try:
            user_query = input("\n👤 USER: ")
            if user_query.lower() in ["exit", "quit"]:
                logger.info("Shutting down system")
                break
            
            # Generate request ID for this query
            request_id = str(uuid.uuid4())
            logger.info(
                "Processing user query",
                request_id=request_id,
                question=user_query
            )
            
            # Run the graph with request_id
            result = app.invoke({
                "question": user_query,
                "request_id": request_id
            })
            
            # Print the final result clearly
            logger.info(
                "Query completed",
                request_id=request_id,
                category=result.get("category", "unknown"),
                answer_length=len(result.get("answer", ""))
            )
            print(f"🤖 AI: {result['answer']}")
            
        except Exception as e:
            logger.error(
                "Error processing query",
                request_id=request_id or "unknown",
                error=str(e)
            )
            print(f"❌ Error: {e}")