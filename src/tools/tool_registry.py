"""
Tool registry for managing available tools.
"""
from typing import Dict, Callable, Any
from .billing_tool import get_refund_status

logger = None  # Lazy import to avoid circular dependency


class ToolRegistry:
    """Registry of available tools for actor execution."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        self.register("get_refund_status", get_refund_status)
    
    def register(self, name: str, tool_func: Callable):
        """Register a tool."""
        self._tools[name] = tool_func
    
    def get(self, name: str) -> Callable:
        """Get a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        return self._tools[name]
    
    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
        
        Returns:
            Tool execution result
        """
        tool = self.get(tool_name)
        return tool(**params)


# Global registry instance
_registry = ToolRegistry()


def get_tool(name: str) -> Callable:
    """Get a tool from the global registry."""
    return _registry.get(name)


def register_tool(name: str, tool_func: Callable):
    """Register a tool in the global registry."""
    _registry.register(name, tool_func)


def list_tools() -> list[str]:
    """List all available tools."""
    return _registry.list_tools()


def execute_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool from the global registry."""
    return _registry.execute(tool_name, params)
