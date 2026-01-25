"""
Basic tracing support for request tracking.
"""
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from src.observability.logger import get_logger


@dataclass
class TraceContext:
    """Trace context for request tracking."""
    trace_id: str
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Span:
    """Represents a single span in a trace."""
    
    def __init__(self, name: str, trace_context: TraceContext, parent_span_id: Optional[str] = None):
        self.name = name
        self.trace_context = trace_context
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = parent_span_id
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
        self.logger = get_logger("tracing")
    
    def __enter__(self):
        self.logger.debug(
            "Span started",
            trace_id=self.trace_context.trace_id,
            span_id=self.span_id,
            span_name=self.name
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        self.logger.info(
            "Span completed",
            trace_id=self.trace_context.trace_id,
            span_id=self.span_id,
            span_name=self.name,
            duration_ms=duration_ms,
            success=exc_type is None
        )
        return False
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to span."""
        self.metadata[key] = value


def create_trace_context(trace_id: Optional[str] = None) -> TraceContext:
    """Create a new trace context."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    return TraceContext(trace_id=trace_id)


def get_trace_context() -> Optional[TraceContext]:
    """Get current trace context (thread-local in future)."""
    # For now, return None - can be enhanced with thread-local storage
    return None
