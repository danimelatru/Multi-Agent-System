# Multi-Agent System: Planner-Grounder-Actor Architecture

A production-ready **multi-agent RAG system** implementing a strict **Planner–Grounder–Actor** architecture. The system is model-agnostic, stateless by design, and built for observability, evaluation, and deployment.

---

## Quick Start

### 1. Install Dependencies

**Option A: Install as package (recommended)**
```bash
pip install -e .
```

**Option B: Install dependencies only**
```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. Configure Environment

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Initialize Database
```bash
python -c "from src.tools import init_billing_db; init_billing_db()"
```

### 4. Run API Server
```bash
python scripts/run_api.py
```

Server available at `http://localhost:8000`

### 5. Test the API
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What\'s the status of ORD-123?"}'
```

---

## Architecture

### Overview

This system implements a **Planner–Grounder–Actor** architecture designed for **reliable, auditable, and extensible multi-agent RAG workflows**. Each agent has a single responsibility, uses structured I/O, and never shares global state.

The orchestrator coordinates agents by explicitly passing an `ExecutionState` object.

---

### Architecture Components

#### 1. Planner Agent
**Responsibility:** Planning and task decomposition  
- Input: User query  
- Output: Structured execution plan (JSON / Pydantic)  
- Constraints: No retrieval, no tools  
- File: `src/agents/planner.py`

#### 2. Grounder (Perceiver) Agent
**Responsibility:** Evidence retrieval (RAG)  
- Input: Retrieval needs from planner  
- Output: Evidence set (documents, excerpts, confidence scores)  
- Constraints: No answer generation  
- File: `src/agents/grounder.py`

#### 3. Actor (Executor) Agent
**Responsibility:** Tool execution and answer synthesis  
- Input: Planner plan + Grounder evidence  
- Output: Final answer and execution trace  
- Constraints: Must follow the plan strictly  
- File: `src/agents/actor.py`

#### 4. Critic Agent (Optional)
**Responsibility:** Output validation  
- Input: Full execution state  
- Output: Validation decision / retry trigger  
- File: `src/agents/critic.py`

---

### Architectural Guarantees

- **Model-agnostic agents** (different LLM per role)
- **No shared global state**
- **Explicit state passing**
- **Structured, versioned outputs**
- **Deterministic orchestration flow**
- **Production-grade observability**
- **Hybrid RAG**: Prioritizes retrieved evidence but falls back to general model knowledge for out-of-domain queries
- **Automatic log rotation**: Keeps only the last 7 days of logs to prevent disk bloat

---

## Directory Structure

```
ma_system/
├── config/
│   ├── models.yaml
│   ├── retrieval.yaml
│   └── policies.yaml
├── prompts/
│   ├── planner.yaml
│   ├── grounder.yaml
│   ├── actor.yaml
│   └── critic.yaml
├── src/
│   ├── agents/
│   ├── orchestrator/
│   ├── retrieval/
│   ├── tools/
│   ├── observability/
│   └── api/
├── scripts/
├── data/
└── logs/
```

---

## Configuration

### Models (`config/models.yaml`)
```yaml
planner:
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.1
```

### Retrieval (`config/retrieval.yaml`)
```yaml
retrieval:
  k: 3
  score_threshold: 0.5
```

### Policies (`config/policies.yaml`)
```yaml
orchestration:
  max_iterations: 3
  enable_critic: true
```

---

## API

### Endpoints

- `GET /health`
- `POST /query`

**Response includes:**
- `request_id`
- `answer`
- `execution_state`
- `trace_id`

Swagger UI: `http://localhost:8000/docs`

---

## Observability

- Structured JSON logs in `logs/`
- Prompt / model / retrieval version tracking
- Trace IDs per request
- Tool usage metrics

Example log:
```json
{
  "timestamp": "2026-01-24T...",
  "level": "INFO",
  "agent": "planner",
  "model": "llama-3.3-70b-versatile",
  "prompt_version": "1.0.0"
}
```

---

## Evaluation

Run:
```bash
python scripts/eval.py
```

Metrics:
- Routing accuracy
- Retrieval hit@k
- Tool success rate

Results: `data/eval_results.json`

---

## Docker

```bash
docker-compose up --build
```

or

```bash
docker build -t ma-system .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key ma-system
```

---

## Extensibility

### Add a Tool
1. Implement in `src/tools/`
2. Register in `tool_registry.py`
3. Update planner prompt

### Add an Agent
1. Extend `BaseAgent`
2. Add prompt YAML
3. Update orchestrator

### Change Models
Edit `config/models.yaml`

---

## Migration Notes

Replaces previous LangGraph-based system:
- Router → Planner
- Agents → Grounder + Actor
- Adds critic, observability, offline eval
