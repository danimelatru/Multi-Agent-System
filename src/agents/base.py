"""
Base agent class with model-agnostic LLM interface.
"""
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.observability.logger import get_logger


class BaseAgent(ABC):
    """Base class for all agents with model-agnostic LLM support."""
    
    def __init__(
        self,
        role: str,
        config_path: str = "config/models.yaml",
        prompt_path: Optional[str] = None
    ):
        self.role = role
        self.logger = get_logger(f"agent.{role}")
        self.config = self._load_config(config_path)
        self.prompt_config = self._load_prompt(prompt_path) if prompt_path else None
        self.llm = self._create_llm()
        self._log_model_version()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path) as f:
            config = yaml.safe_load(f)
        return config.get(self.role, {})
    
    def _load_prompt(self, prompt_path: str) -> Dict[str, Any]:
        """Load prompt configuration from YAML."""
        path = Path(prompt_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _create_llm(self) -> BaseChatModel:
        """Create LLM instance based on config (model-agnostic)."""
        provider = self.config.get("provider", "groq")
        model = self.config.get("model", "llama-3.3-70b-versatile")
        temperature = self.config.get("temperature", 0.0)
        max_tokens = self.config.get("max_tokens", 2048)
        
        if provider == "groq":
            import os
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(f"GROQ_API_KEY not found for {self.role}")
            return ChatGroq(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _log_model_version(self):
        """Log model version for observability."""
        model = self.config.get("model", "unknown")
        provider = self.config.get("provider", "unknown")
        prompt_version = self.prompt_config.get("version", "unknown") if self.prompt_config else "unknown"
        
        self.logger.info(
            "Agent initialized",
            role=self.role,
            model=model,
            provider=provider,
            prompt_version=prompt_version
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt from prompt config."""
        if not self.prompt_config:
            return ""
        return self.prompt_config.get("system_prompt", "")
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """Execute agent logic - implemented by subclasses."""
        pass
