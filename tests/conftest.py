"""
Pytest configuration and fixtures for deterministic tests (no network calls).
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def mock_llm():
    """Mock LLM that returns predictable responses without network calls."""
    llm = Mock()
    
    # Mock router responses
    router_response = Mock()
    router_response.destination = "technical"  # Default, can be overridden
    llm.with_structured_output.return_value.invoke.return_value = router_response
    
    # Mock technical agent responses
    llm.invoke.return_value.content = "Mock technical response"
    
    # Mock billing agent responses
    billing_msg = Mock()
    billing_msg.content = "Mock billing response"
    billing_msg.tool_calls = []
    llm.bind_tools.return_value.invoke.return_value = billing_msg
    
    return llm

@pytest.fixture
def mock_embeddings():
    """Mock embeddings to avoid loading models."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    embeddings.embed_documents.return_value = [[0.1] * 384] * 10
    return embeddings

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    from langchain_core.documents import Document
    return [
        Document(page_content="ERROR CODE 101: Blue Screen of Death. Solution: Restart in Safe Mode."),
        Document(page_content="ERROR CODE 202: Transaction Failed. Solution: Check credit card."),
    ]
