"""
Test billing agent tool-calling functionality - deterministic, no network calls.
"""
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from src.agents.billing import get_refund_status, init_billing_db, get_db_connection

def test_billing_tool_call_existing_order():
    """Test that get_refund_status returns correct status for existing order."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    # Override DB_PATH for this test
    import src.agents.billing as billing_module
    original_path = billing_module.DB_PATH
    billing_module.DB_PATH = db_path
    
    try:
        # Initialize database
        init_billing_db()
        
        # Test tool call
        result = get_refund_status.invoke({"order_id": "ORD-123"})
        
        assert "Refund Processed" in result
    finally:
        # Cleanup
        billing_module.DB_PATH = original_path
        if db_path.exists():
            db_path.unlink()

def test_billing_tool_call_nonexistent_order():
    """Test that get_refund_status handles nonexistent orders."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = Path(tmp.name)
    
    import src.agents.billing as billing_module
    original_path = billing_module.DB_PATH
    billing_module.DB_PATH = db_path
    
    try:
        init_billing_db()
        
        result = get_refund_status.invoke({"order_id": "ORD-NONEXISTENT"})
        
        assert "not found" in result.lower()
    finally:
        billing_module.DB_PATH = original_path
        if db_path.exists():
            db_path.unlink()

def test_billing_agent_tool_binding():
    """Test that billing agent correctly binds tools."""
    mock_llm = Mock()
    mock_bind = Mock()
    mock_llm.bind_tools = Mock(return_value=mock_bind)
    
    with patch('src.agents.billing.get_llm', return_value=mock_llm):
        from src.agents.billing import build_billing_agent
        agent = build_billing_agent()
        
        # Verify bind_tools was called
        mock_llm.bind_tools.assert_called_once()
        # Verify tools list contains get_refund_status
        call_args = mock_llm.bind_tools.call_args[0][0]
        assert len(call_args) > 0
        assert any(tool.name == "get_refund_status" for tool in call_args)
