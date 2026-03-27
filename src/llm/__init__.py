"""
LLM Module

Provides LLM client, configuration, and prompt templates for the RAG pipeline.
"""

from .client import MistralClient
from .config import PipelineConfig
from .prompts import build_legal_prompt

__all__ = ["MistralClient", "PipelineConfig", "build_legal_prompt"]
