import sqlite3
from pathlib import Path
from contextlib import contextmanager
from langchain_core.tools import tool
from src.config import get_llm

# Database path
DB_PATH = Path("data/billing.db")

@contextmanager
def get_db_connection():
    """
    Context manager for database connections with automatic commit/rollback.
    """
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
    """
    Initialize the billing database schema and seed initial data.
    Must be called explicitly from run_system.py, not at import time.
    """
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with get_db_connection() as conn:
        # Create orders table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                refund_status TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Seed existing ORD examples (only if table is empty)
        cursor = conn.execute("SELECT COUNT(*) as count FROM orders")
        if cursor.fetchone()["count"] == 0:
            conn.execute("""
                INSERT INTO orders (order_id, refund_status) VALUES
                ('ORD-123', 'Refund Processed'),
                ('ORD-456', 'Pending Manager Approval'),
                ('ORD-999', 'Rejected: Item damaged by user')
            """)

# Define the Tool
@tool
def get_refund_status(order_id: str):
    """
    Queries the database for the refund status of a specific order ID.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT refund_status FROM orders WHERE order_id = ?",
                (order_id,)
            )
            row = cursor.fetchone()
            if row:
                return row["refund_status"]
            return f"Order ID '{order_id}' not found in system."
    except Exception as e:
        return f"Database error: {str(e)}"

def build_billing_agent():
    """
    Returns the LLM binded with the billing tools.
    """
    llm = get_llm()
    tools = [get_refund_status]
    # Bind tools so the LLM knows it can call them
    return llm.bind_tools(tools)