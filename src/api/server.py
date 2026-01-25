"""
FastAPI server for the multi-agent system.
"""
import uuid
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.orchestrator import Orchestrator
from src.common.state import ExecutionState
from src.agents import Planner, Grounder, Actor, Critic
from src.retrieval import Retriever
from src.tools import init_billing_db
from src.observability.tracing import create_trace_context
from src.observability.logger import get_logger

from dotenv import load_dotenv
load_dotenv()

# Initialize logger
logger = get_logger("api")

# Initialize components
logger.info("Initializing system components...")

# Initialize database
init_billing_db()

# Initialize retrieval
retriever = Retriever()

# Initialize agents
planner = Planner()
grounder = Grounder(retriever)
actor = Actor()
critic = Critic()

# Initialize orchestrator
orchestrator = Orchestrator(
    planner=planner,
    grounder=grounder,
    actor=actor,
    critic=critic,
    enable_critic=True
)

logger.info("System initialized")

# FastAPI app
app = FastAPI(
    title="Multi-Agent System API",
    description="Planner-Grounder-Actor multi-agent system",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    """Request model for user queries."""
    query: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Response model for queries."""
    request_id: str
    answer: str
    execution_state: Dict[str, Any]
    trace_id: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "multi-agent-system",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    x_trace_id: Optional[str] = Header(None, alias="X-Trace-Id")
):
    """
    Process a user query through the planner-grounder-actor system.
    
    Args:
        request: Query request with user query
        x_trace_id: Optional trace ID for request tracking
    
    Returns:
        Query response with answer and execution details
    """
    # Create trace context
    trace_context = create_trace_context(trace_id=x_trace_id)
    
    logger.info(
        "Processing query",
        trace_id=trace_context.trace_id,
        query=request.query,
        user_id=request.user_id
    )
    
    try:
        # Execute through orchestrator
        state = orchestrator.execute(
            user_query=request.query,
            trace_context=trace_context
        )
        
        # Format response
        response = QueryResponse(
            request_id=state.request_id,
            answer=state.answer or "No answer generated.",
            execution_state={
                "plan": state.plan,
                "evidence_count": len(state.evidence) if state.evidence else 0,
                "steps_executed_count": len(state.steps_executed),
                "tools_used_count": len(state.tools_used),
                "validation_passed": state.validation_result.get("valid", False) if state.validation_result else None
            },
            trace_id=trace_context.trace_id
        )
        
        logger.info(
            "Query processed successfully",
            trace_id=trace_context.trace_id,
            request_id=state.request_id
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Query processing failed",
            trace_id=trace_context.trace_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get system metrics (placeholder for future implementation)."""
    return {
        "requests_total": 0,
        "requests_by_agent": {},
        "average_response_time_ms": 0
    }
