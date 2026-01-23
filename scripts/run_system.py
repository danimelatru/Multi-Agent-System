import sys
import os

# Add the project root to the python path to allow imports from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from src.agents.router import build_router_agent
from src.agents.technical import build_technical_agent
from src.agents.billing import build_billing_agent, get_refund_status

# --- 1. Define State ---
class AgentState(TypedDict):
    question: str
    category: str
    answer: str

# --- 2. Initialize Agents ---
print("⚙️ Initializing Agents...")
router_chain = build_router_agent()
rag_chain = build_technical_agent()
billing_llm = build_billing_agent()

# --- 3. Define Nodes ---

def router_node(state: AgentState):
    """Analyzes the input and updates the category."""
    print(f"\n📡 ROUTER: Analyzing query: '{state['question']}'")
    result = router_chain.invoke({"question": state["question"]})
    print(f"   ↳ Decision: {result.destination.upper()}")
    return {"category": result.destination}

def technical_node(state: AgentState):
    """Handles technical queries using RAG."""
    print("🛠️ TECH AGENT: Searching knowledge base...")
    response = rag_chain.invoke(state["question"])
    return {"answer": response}

def billing_node(state: AgentState):
    """Handles billing queries using Tools."""
    print("💰 BILLING AGENT: Processing...")
    
    # Invoke LLM with tools
    msg = billing_llm.invoke(state["question"])
    
    # Check if the LLM wants to call a tool
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            if tool_call["name"] == "get_refund_status":
                print(f"   ↳ Calling Tool: get_refund_status({tool_call['args']})")
                # Execute tool
                tool_result = get_refund_status.invoke(tool_call["args"])
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
    print("\n🚀 MA-SYSTEM ONLINE")
    print("-------------------------------------")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        # Get user input from terminal
        try:
            user_query = input("\n👤 USER: ")
            if user_query.lower() in ["exit", "quit"]:
                print("👋 Shutting down system.")
                break
            
            # Run the graph
            result = app.invoke({"question": user_query})
            
            # Print the final result clearly
            print(f"🤖 AI: {result['answer']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")