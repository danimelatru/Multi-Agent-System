# Multi-Agent Support System (RAG + Tools)

## 🚀 Overview
This project implements a **Compound AI System** designed to triage and resolve customer support queries autonomously. It utilizes a **Multi-Agent Architecture** orchestrated with **LangGraph** and powered by **Llama 3.3** (via Groq).

## 🏗️ Architecture
The system is composed of specialized agents coordinated by a semantic router:

1.  **Router Agent:** Analyzes intent and routes traffic (Classification).
2.  **Technical Agent:** Uses **RAG (Retrieval Augmented Generation)** to answer technical questions based on internal documentation (`ChromaDB` + `HuggingFace Embeddings`).
3.  **Billing Agent:** Utilizes **Tool Calling** (Function Execution) to query a simulated SQL database for real-time order status.

## 🛠️ Tech Stack
* **Orchestration:** LangChain & LangGraph (State Graph)
* **LLM:** Llama-3.3-70b-versatile (Groq API)
* **Vector Store:** ChromaDB
* **Environment:** Python 3.10 / Conda

## ⚡ How to Run

1.  **Clone the repository**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment:**
    Create a `.env` file and add your Groq API Key:
    ```env
    GROQ_API_KEY=gsk_...
    ```
4.  **Run the System:**
    ```bash
    python scripts/run_system.py
    ```

## 🧪 Example Usage
* **User:** "How do I fix error 101?" -> **Agent:** RAG Retrieval from manual.
* **User:** "Status of ORD-123?" -> **Agent:** Database Tool Execution.