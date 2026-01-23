"""
Minimal structured logging utility for the multi-agent system.
Uses key-value format for structured logs.
"""
import json
import sys
from datetime import datetime
from typing import Optional

class StructuredLogger:
    """
    Simple structured logger that outputs key-value pairs or JSON.
    """
    
    def __init__(self, name: str, json_format: bool = False):
        self.name = name
        self.json_format = json_format
    
    def _format_message(self, level: str, message: str, **kwargs) -> str:
        """Format log message as key-value or JSON."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        if self.json_format:
            return json.dumps(log_data)
        else:
            # Key-value format
            parts = [f"timestamp={log_data['timestamp']}", f"level={level}", f"logger={self.name}"]
            parts.append(f"message={message}")
            for key, value in kwargs.items():
                parts.append(f"{key}={value}")
            return " | ".join(parts)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        print(self._format_message("INFO", message, **kwargs), file=sys.stdout)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        print(self._format_message("ERROR", message, **kwargs), file=sys.stderr)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        print(self._format_message("DEBUG", message, **kwargs), file=sys.stdout)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        print(self._format_message("WARNING", message, **kwargs), file=sys.stdout)

def get_logger(name: str, json_format: bool = False) -> StructuredLogger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Logger name (e.g., 'router', 'billing', 'technical')
        json_format: If True, output JSON; otherwise key-value format
    
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, json_format=json_format)
