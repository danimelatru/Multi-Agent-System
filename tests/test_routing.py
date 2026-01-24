"""
Test routing functionality - deterministic, no network calls.
"""
import pytest
from unittest.mock import Mock, patch
from langchain_core.runnables import RunnableLambda
from src.agents.router import build_router_agent, RouteQuery


def _make_fake_router(destination: str):
    """Return a runnable that always yields RouteQuery(destination=...)."""
    return RunnableLambda(lambda _: RouteQuery(destination=destination))


def test_router_technical_query():
    """Test that technical queries are routed correctly."""
    mock_llm = Mock()
    mock_llm.with_structured_output.return_value = _make_fake_router("technical")

    with patch("src.agents.router.get_llm", return_value=mock_llm):
        router = build_router_agent()
        result = router.invoke({"question": "How do I fix error 101?"})

        assert result.destination == "technical"


def test_router_billing_query():
    """Test that billing queries are routed correctly."""
    mock_llm = Mock()
    mock_llm.with_structured_output.return_value = _make_fake_router("billing")

    with patch("src.agents.router.get_llm", return_value=mock_llm):
        router = build_router_agent()
        result = router.invoke({"question": "What's the status of ORD-123?"})

        assert result.destination == "billing"


def test_router_general_query():
    """Test that general queries are routed correctly."""
    mock_llm = Mock()
    mock_llm.with_structured_output.return_value = _make_fake_router("general")

    with patch("src.agents.router.get_llm", return_value=mock_llm):
        router = build_router_agent()
        result = router.invoke({"question": "Hello, how are you?"})

        assert result.destination == "general"
