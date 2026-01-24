# Phase A: Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the System

```bash
# Start the CLI system
python scripts/run_system.py
```

The system will:
1. Initialize the billing database (creates `data/billing.db`)
2. Load or create the ChromaDB vector store (`data/chroma_db/`)
3. Start the interactive CLI

Example interactions:
- "How do I fix error 101?" → Routes to technical agent
- "What's the status of ORD-123?" → Routes to billing agent
- "Hello" → Routes to general agent

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_routing.py -v # Test routing
pytest tests/test_billing.py -v # Test billing tool-calling
pytest tests/test_rag.py -v # Test RAG retrieval

# Run with output
pytest tests/ -v -s

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Expected Test Output

All tests should pass:
- ✅ `test_router_technical_query` - Routes technical queries
- ✅ `test_router_billing_query` - Routes billing queries  
- ✅ `test_router_general_query` - Routes general queries
- ✅ `test_billing_tool_call_existing_order` - Billing tool works
- ✅ `test_billing_tool_call_nonexistent_order` - Handles missing orders
- ✅ `test_billing_agent_tool_binding` - Tools are bound correctly
- ✅ `test_rag_retrieval_works` - RAG retrieves documents
- ✅ `test_rag_persistence` - Vector store persists

## File Structure

```
ma_system/
├── data/
│   ├── billing.db
│   ├── chroma_db/
│   └── tech_manual.txt
├── src/
│   ├── agents/
│   │   ├── billing.py
│   │   ├── router.py
│   │   └── technical.py
│   ├── utils/
│   │   └── logging.py
│   └── config.py
├── scripts/
│   └── run_system.py
├── tests/
│   ├── test_routing.py
│   ├── test_billing.py
│   └── test_rag.py
└── requirements.txt
```

## Troubleshooting

**Database errors:**
- Ensure `data/` directory exists and is writable
- Delete `data/billing.db` to reset database

**Vector store issues:**
- Delete `data/chroma_db/` to rebuild vector store
- First run will be slower (embedding generation)

**Test failures:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Run with `-v` flag for verbose output: `pytest tests/ -v`
