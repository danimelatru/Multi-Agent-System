import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

def get_llm():
    """
    Returns the configured LLM instance (Llama 3 on Groq).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ Error: GROQ_API_KEY not found in .env file.")
        
    # Using Llama 3 70B for high performance and low cost
    return ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=api_key
    )