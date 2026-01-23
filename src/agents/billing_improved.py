"""
Improved Billing Agent with Real Database Integration

This is a reference implementation showing how to replace the mock database
with a real SQLite database. For production, use PostgreSQL (see ARCHITECTURE_IMPROVEMENTS.md).
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from src.config import get_llm

# Database configuration
DB_PATH = Path("data/billing.db")
DB_INITIALIZED = False

@contextmanager
def get_db_connection():
    """
    Context manager for database connections with automatic commit/rollback.
    
    Usage:
        with get_db_connection() as conn:
            cursor = conn.execute("SELECT ...")
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_billing_db():
    """
    Initialize the billing database schema and seed initial data.
    This is idempotent - safe to call multiple times.
    """
    global DB_INITIALIZED
    
    if DB_INITIALIZED:
        return
    
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db_connection() as conn:
        # Create orders table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                refund_status TEXT NOT NULL,
                amount DECIMAL(10, 2),
                currency TEXT DEFAULT 'USD',
                customer_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_order_id ON orders(order_id)
        """)
        
        # Seed initial data (only if table is empty)
        cursor = conn.execute("SELECT COUNT(*) as count FROM orders")
        if cursor.fetchone()["count"] == 0:
            conn.execute("""
                INSERT INTO orders (order_id, refund_status, amount, customer_id) VALUES
                ('ORD-123', 'Refund Processed', 99.99, 'CUST-001'),
                ('ORD-456', 'Pending Manager Approval', 149.50, 'CUST-002'),
                ('ORD-999', 'Rejected: Item damaged by user', 75.00, 'CUST-003')
            """)
    
    DB_INITIALIZED = True

@tool
def get_refund_status(order_id: str) -> str:
    """
    Queries the database for the refund status of a specific order ID.
    
    This replaces the mock database with a real SQLite database.
    For production use, replace with PostgreSQL (see ARCHITECTURE_IMPROVEMENTS.md).
    
    Args:
        order_id: The order identifier (e.g., 'ORD-123')
    
    Returns:
        The refund status string or error message
    """
    # Ensure database is initialized
    init_billing_db()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT refund_status, amount, currency FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()
            
            if row:
                status = row["refund_status"]
                amount = row.get("amount")
                currency = row.get("currency", "USD")
                
                # Format response with additional context if available
                if amount:
                    return f"{status} (Amount: {currency} {amount:.2f})"
                return status
            else:
                return f"Order ID '{order_id}' not found in system."
                
    except sqlite3.OperationalError as e:
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

@tool
def update_refund_status(order_id: str, new_status: str) -> str:
    """
    Updates the refund status of an order (for future enhancement).
    
    Args:
        order_id: The order identifier
        new_status: The new refund status
    
    Returns:
        Success or error message
    """
    init_billing_db()
    
    try:
        with get_db_connection() as conn:
            # Check if order exists
            cursor = conn.execute(
                "SELECT order_id FROM orders WHERE order_id = ?",
                (order_id,)
            )
            if not cursor.fetchone():
                return f"Order ID '{order_id}' not found."
            
            # Update status
            conn.execute("""
                UPDATE orders 
                SET refund_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE order_id = ?
            """, (new_status, order_id))
            
            return f"Order {order_id} status updated to: {new_status}"
            
    except Exception as e:
        return f"Error updating status: {str(e)}"

def build_billing_agent():
    """
    Returns the LLM bound with the billing tools.
    
    This function signature matches the original to maintain compatibility.
    """
    llm = get_llm()
    tools = [get_refund_status, update_refund_status]
    
    # Bind tools so the LLM knows it can call them
    return llm.bind_tools(tools)

# Initialize database on module import
init_billing_db()
