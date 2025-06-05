"""
RAG (Retrieval-Augmented Generation) module for the Voice-Powered Car Assistant.

This module provides the public API for the RAG components, including the
ManualRetriever, QueryProcessor, ContextRetriever, and ResponseGenerator.
"""

# Import from project modules
from rag.retriever import ManualRetriever, ContextRetriever
from rag.processor import QueryProcessor
from rag.generator import ResponseGenerator
from rag.context import ContextManager
from rag.templates import PromptTemplateManager

# Version information
__version__ = "0.1.0"

# Public exports
__all__ = [
    'ManualRetriever',
    'QueryProcessor',
    'ContextRetriever',
    'ResponseGenerator',
    'ContextManager',
    'PromptTemplateManager',
]