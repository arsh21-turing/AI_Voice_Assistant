#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG (Retrieval-Augmented Generation) system components.

This package provides the core components for the RAG system:
- QueryProcessor: Processes and analyzes user queries
- ContextRetriever: Retrieves relevant context from the knowledge base
- ResponseGenerator: Generates natural language responses from context
- ManualRetriever: Orchestrates the complete RAG pipeline
"""

# Package metadata
__version__ = '0.1.0'
__author__ = 'Voice-Powered Car Assistant Team'

# Import public classes
from .processor import QueryProcessor
from .context import ContextRetriever
from .generator import ResponseGenerator
from .retriever import ManualRetriever

# Define public API
__all__ = [
    'QueryProcessor',
    'ContextRetriever',
    'ResponseGenerator',
    'ManualRetriever'
]

# Package initialization
def get_version():
    """Return the package version."""
    return __version__

def get_components():
    """Return a dictionary of available RAG components."""
    return {
        'QueryProcessor': 'Processes and analyzes user queries',
        'ContextRetriever': 'Retrieves relevant context from the knowledge base',
        'ResponseGenerator': 'Generates natural language responses from context',
        'ManualRetriever': 'Orchestrates the complete RAG pipeline'
    }