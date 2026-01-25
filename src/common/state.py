"""
Execution state definition - separated to avoid circular imports.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ExecutionState:
    """Execution state - no shared global state, passed between agents."""
    request_id: str
    user_query: str
    plan: Optional[Dict[str, Any]] = None
    evidence: Optional[List[Dict[str, Any]]] = None
    answer: Optional[str] = None
    steps_executed: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    validation_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
