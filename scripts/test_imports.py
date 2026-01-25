"""
Test script to verify all imports work correctly.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        print("  - Testing src.tools...")
        from src.tools import execute_tool, init_billing_db, get_refund_status
        print("    ✓ src.tools imports OK")
        
        print("  - Testing src.agents...")
        from src.agents import Planner, Grounder, Actor, Critic
        print("    ✓ src.agents imports OK")
        
        print("  - Testing src.retrieval...")
        from src.retrieval import Retriever
        print("    ✓ src.retrieval imports OK")
        
        print("  - Testing src.orchestrator...")
        from src.orchestrator import Orchestrator
        from src.common.state import ExecutionState
        print("    ✓ src.orchestrator imports OK")
        
        print("  - Testing src.observability...")
        from src.observability import get_logger, TraceContext
        print("    ✓ src.observability imports OK")
        
        print("  - Testing src.api...")
        from src.api import app
        print("    ✓ src.api imports OK")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
