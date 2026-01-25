"""
Billing tool for querying order refund status.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any

from src.observability.logger import get_logger

logger = get_logger("tool.billing")

# Database path
DB_PATH = Path("data/billing.db")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_billing_db():
    """Initialize billing database schema and seed data."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                refund_status TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor = conn.execute("SELECT COUNT(*) as count FROM orders")
        if cursor.fetchone()["count"] == 0:
            conn.execute("""
                INSERT INTO orders (order_id, refund_status) VALUES
                ('ORD-123', 'Refund Processed'),
                ('ORD-456', 'Pending Manager Approval'),
                ('ORD-999', 'Rejected: Item damaged by user')
            """)


def get_refund_status(order_id: str) -> Dict[str, Any]:
    """
    Query database for refund status of an order.
    
    Args:
        order_id: Order identifier (e.g., 'ORD-123')
    
    Returns:
        Dictionary with status and result
    """
    logger.info("Executing get_refund_status", order_id=order_id)
    
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT refund_status FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()
            
            if row:
                result = {
                    "success": True,
                    "order_id": order_id,
                    "refund_status": row["refund_status"]
                }
                logger.info("Tool execution successful", order_id=order_id)
                return result
            else:
                result = {
                    "success": False,
                    "order_id": order_id,
                    "error": f"Order ID '{order_id}' not found in system."
                }
                logger.warning("Order not found", order_id=order_id)
                return result
                
    except Exception as e:
        error_msg = f"Database error: {str(e)}"
        logger.error("Tool execution failed", order_id=order_id, error=error_msg)
        return {
            "success": False,
            "order_id": order_id,
            "error": error_msg
        }
