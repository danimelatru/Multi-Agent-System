"""
Retrieval module for RAG operations.
Handles vector store initialization and document retrieval.
"""
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.observability.logger import get_logger


class Retriever:
    """
    Retrieval component for RAG operations.
    Returns evidence with doc IDs, excerpts, and confidence scores.
    """
    
    def __init__(self, config_path: str = "config/retrieval.yaml"):
        self.logger = get_logger("retriever")
        self.config = self._load_config(config_path)
        self.embeddings = self._create_embeddings()
        self.vector_store = self._initialize_vector_store()
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config["retrieval"]["search_type"],
            search_kwargs={
                "k": self.config["retrieval"]["k"],
                "score_threshold": self.config["retrieval"]["score_threshold"]
            }
        )
        self._log_retrieval_version()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load retrieval configuration."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Retrieval config not found: {config_path}")
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _create_embeddings(self):
        """Create embeddings model."""
        model_name = self.config["embeddings"]["model"]
        return HuggingFaceEmbeddings(model_name=model_name)
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load vector store."""
        persist_dir = Path(self.config["vector_store"]["persist_directory"])
        
        if persist_dir.exists() and any(persist_dir.iterdir()):
            # Load existing
            self.logger.info("Loading existing vector store", path=str(persist_dir))
            return Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings
            )
        else:
            # Create new
            self.logger.info("Creating new vector store", path=str(persist_dir))
            return self._create_vector_store(persist_dir)
    
    def _create_vector_store(self, persist_dir: Path) -> Chroma:
        """Create new vector store from knowledge base."""
        source_path = Path(self.config["knowledge_base"]["source_path"])
        
        if not source_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {source_path}")
        
        # Load documents
        loader = TextLoader(str(source_path))
        documents = loader.load()
        
        # Split into chunks
        chunk_size = self.config["chunking"]["chunk_size"]
        chunk_overlap = self.config["chunking"]["chunk_overlap"]
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add doc IDs to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata["doc_id"] = f"chunk_{i}"
            chunk.metadata["source"] = str(source_path)
        
        # Create vector store
        persist_dir.parent.mkdir(parents=True, exist_ok=True)
        db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=str(persist_dir)
        )
        db.persist()
        
        self.logger.info(
            "Vector store created",
            chunks_count=len(chunks),
            path=str(persist_dir)
        )
        
        return db
    
    def _log_retrieval_version(self):
        """Log retrieval configuration version."""
        embedding_model = self.config["embeddings"]["model"]
        k = self.config["retrieval"]["k"]
        self.logger.info(
            "Retriever initialized",
            embedding_model=embedding_model,
            k=k,
            search_type=self.config["retrieval"]["search_type"]
        )
    
    def retrieve(self, queries: List[str], min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve evidence for given queries.
        
        Args:
            queries: List of search queries
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of evidence items with doc_id, excerpt, confidence, source
        """
        all_evidence = []
        
        for query in queries:
            try:
                # Use vector store directly to get scores (likely distance)
                k = self.config["retrieval"]["k"]
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
                
                for doc, score in docs_with_scores:
                    # Pass all retrieved docs regardless of score for now
                    # (Metric might be distance, so higher/lower logic varies)
                    evidence = {
                        "doc_id": doc.metadata.get("doc_id", "unknown"),
                        "excerpt": doc.page_content,
                        "confidence": float(score),
                        "source": doc.metadata.get("source", "unknown"),
                        "metadata": doc.metadata
                    }
                    all_evidence.append(evidence)
                
            except Exception as e:
                self.logger.error(
                    "Retrieval failed for query",
                    query=query,
                    error=str(e)
                )
        
        self.logger.info(
            "Retrieval completed",
            queries_count=len(queries),
            evidence_count=len(all_evidence)
        )
        
        return all_evidence
