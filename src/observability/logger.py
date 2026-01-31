"""
Structured JSON logging with version tracking.
Logs prompt/model/retrieval versions per request.
"""
import json
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path


class StructuredLogger:
    """
    Structured JSON logger that outputs machine-readable logs.
    Tracks versions for prompts, models, and retrieval configs.
    Automatically rotates logs, keeping only the last 7 days.
    """
    
    LOG_RETENTION_DAYS = 7  # Keep logs for 7 days
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.name = name
        self.log_file = log_file or Path("logs") / f"{name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove log files older than LOG_RETENTION_DAYS."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.LOG_RETENTION_DAYS)
            log_dir = self.log_file.parent
            
            # Find all log files for this logger
            pattern = f"{self.name}_*.jsonl"
            for log_file in log_dir.glob(pattern):
                try:
                    # Extract date from filename (format: name_YYYYMMDD.jsonl)
                    date_str = log_file.stem.split('_')[-1]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        print(f"Deleted old log file: {log_file.name}", file=sys.stderr)
                except (ValueError, IndexError):
                    # Skip files that don't match the expected format
                    continue
        except Exception as e:
            # Don't crash if cleanup fails
            print(f"Warning: Log cleanup failed: {e}", file=sys.stderr)
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        # Write to file (JSONL format)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Also print to stdout for development
        print(json.dumps(log_entry), file=sys.stdout)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log("DEBUG", message, **kwargs)


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name (e.g., 'planner', 'grounder', 'actor')
    
    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]
