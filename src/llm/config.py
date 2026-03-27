"""
Environment variable loading through dataclasses to ensure type safety
Single unified config for the entire pipeline (LLM + RAG settings)
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class PipelineConfig:
    # LLM Settings
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "mistral"
    temperature: float = 0.1
    max_tokens: int = 500
    
    # RAG/Retrieval Settings
    top_k_chunks: int = 25
    top_k_cases: int = 3
    
    # Index Paths
    index_path: str = "faiss_indices/paragraph_chunks_l2.index"
    metadata_path: str = "faiss_indices/paragraph_chunks_metadata_enriched.json"

    # Log path
    log_path: str = "logs/pipeline_interactions.jsonl"
    
    @classmethod    
    def load_from_env(cls):
        """Load configuration from .env file."""
        env_file = Path(".env")
        if env_file.exists():
            try: 
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                print("Warning: python-dotenv not installed")
        
        return cls(
            # LLM settings
            base_url=os.getenv("LLM_BASE_URL", cls.base_url),  
            api_key=os.getenv("LLM_API_KEY", cls.api_key),  
            model=os.getenv("LLM_MODEL", cls.model),  
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.temperature)),  
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", cls.max_tokens)),
            
            # RAG settings
            top_k_chunks=int(os.getenv("RETRIEVAL_TOP_K_CHUNKS", cls.top_k_chunks)),
            top_k_cases=int(os.getenv("RETRIEVAL_TOP_K_CASES", cls.top_k_cases)),
            
            # Index paths
            index_path=os.getenv("FAISS_INDEX_PATH", cls.index_path),
            metadata_path=os.getenv("FAISS_METADATA_PATH", cls.metadata_path),
            
            # Log path
            log_path=os.getenv("LOG_PATH", cls.log_path)
        )