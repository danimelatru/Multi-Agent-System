"""
Test RAG retrieval functionality - deterministic, no network calls.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

def test_rag_retrieval_works():
    """Test that RAG can retrieve relevant documents."""
    documents = [
        Document(page_content="ERROR CODE 101: Blue Screen of Death. Solution: Restart in Safe Mode."),
        Document(page_content="ERROR CODE 202: Transaction Failed. Solution: Check credit card."),
    ]
    dim = 384

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_embeddings = Mock()
        # Distinct embeddings so retrieval is deterministic: doc0 vs doc1.
        mock_embeddings.embed_documents.return_value = [
            [1.0] * dim,
            [0.1] * dim,
        ]
        # Query "error 101" closer to doc0 (101)
        mock_embeddings.embed_query.return_value = [1.0] * dim

        db = Chroma.from_documents(
            documents,
            mock_embeddings,
            persist_directory=tmpdir
        )
        retriever = db.as_retriever(search_kwargs={"k": 1})
        results = retriever.invoke("error 101")

        assert len(results) > 0
        assert isinstance(results[0], Document)
        assert "101" in results[0].page_content or "Blue Screen" in results[0].page_content

def test_rag_persistence():
    """Test that RAG vector store can be persisted and loaded."""
    documents = [
        Document(page_content="Test document 1"),
        Document(page_content="Test document 2"),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 384] * len(documents)
        mock_embeddings.embed_query.return_value = [0.1] * 384
        
        # Create and persist
        db1 = Chroma.from_documents(
            documents,
            mock_embeddings,
            persist_directory=tmpdir
        )
        db1.persist()
        
        # Load from persistence
        db2 = Chroma(
            persist_directory=tmpdir,
            embedding_function=mock_embeddings
        )
        
        # Verify both can retrieve
        retriever1 = db1.as_retriever()
        retriever2 = db2.as_retriever()
        
        results1 = retriever1.invoke("test")
        results2 = retriever2.invoke("test")
        
        assert len(results1) > 0
        assert len(results2) > 0
