import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import get_llm

# Persistent storage path for ChromaDB
VECTOR_DB_PATH = Path("data/chroma_db")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_technical_agent():
    """
    Builds the RAG (Retrieval Augmented Generation) chain with persistent storage.
    Loads existing vector store if present; otherwise creates and persists it.
    """
    # Initialize embeddings (reusable)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    
    # Check if vector store exists
    if VECTOR_DB_PATH.exists() and any(VECTOR_DB_PATH.iterdir()):
        # Load existing vector store (avoids re-embedding)
        db = Chroma(
            persist_directory=str(VECTOR_DB_PATH),
            embedding_function=embeddings
        )
    else:
        # Create new vector store
        # 1. Load Knowledge Base
        file_path = os.path.join("data", "tech_manual.txt")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Document not found at: {file_path}")

        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        
        # 2. Create Vector Store and persist
        VECTOR_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=str(VECTOR_DB_PATH)
        )
        db.persist()
    
    retriever = db.as_retriever()

    # 3. Create the Chain
    llm = get_llm()
    
    template = """You are a technical support assistant. 
    Answer the question based ONLY on the following context. 
    If the answer is not in the context, say "I don't have that information in my manual."

    Context:
    {context}

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )