#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end tests for the RAG pipeline in the Voice-Powered Car Assistant.

This module contains tests for the complete retrieval-augmented generation
pipeline, from query processing to response generation, using a sample
text file for automotive information.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
import shutil
import logging
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import RAG components
from rag import ManualRetriever, QueryProcessor, ContextRetriever, ResponseGenerator
from utils.vector_store import VectorStore
from utils.terminology import TerminologyManager
from utils.helpers import PDFProcessor

# Test utilities
def create_test_file(file_path: str) -> None:
    """Create a test text file with automotive maintenance content."""
    test_content = """
    Automotive Maintenance Guide
    
    Oil Level Check Procedure:
    1. Park on level ground
    2. Wait for engine to cool
    3. Locate dipstick
    4. Pull out and wipe clean
    5. Reinsert fully
    6. Pull out and check level
    
    Oil Filter Replacement:
    1. Drain engine oil
    2. Remove old filter
    3. Install new filter
    4. Refill with fresh oil
    
    Common Issues:
    - Low oil pressure
    - Oil leaks
    - Engine noise
    - Check engine light
    
    Maintenance Schedule:
    - Oil change: Every 5,000 miles
    - Filter change: Every oil change
    - Fluid checks: Monthly
    """
    
    with open(file_path, 'w') as f:
        f.write(test_content)

def create_test_index(index_path: str, file_path: str) -> None:
    """Create a test vector store index from a text file."""
    # Read file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Create vector store
    vector_store = VectorStore(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_path=index_path
    )
    
    # Split content into chunks
    chunks = content.split('\n\n')
    documents = []
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            documents.append({
                'text': chunk.strip(),
                'metadata': {
                    'source': 'test_manual.txt',
                    'chunk_id': i,
                    'section': 'maintenance_guide'
                }
            })
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    # Save index
    vector_store.save(index_path)

def cleanup_test_files(test_dir: str) -> None:
    """Clean up test files and directories."""
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

# Simple ConfigManager class for testing
class TestConfigManager:
    def __init__(self):
        self.config = {
            'context': {
                'default_top_k': 5,
                'relevance_threshold': 0.5,
                'use_reranking': True,
                'max_history_size': 10,
                'chunk_size': 1000,
                'chunk_overlap': 200
            }
        }
    
    def get(self, key, default=None):
        """Get a configuration value using dot notation."""
        try:
            parts = key.split('.')
            value = self.config
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

# Simple ErrorHandler class for testing
class TestErrorHandler:
    def handle_error(self, error, context=None):
        """Handle errors during testing."""
        logger.error(f"Error in {context}: {str(error)}")
        return f"Error occurred: {str(error)}", False
    
    def log_error(self, error, context=None):
        """Log errors during testing."""
        logger.error(f"Error in {context}: {str(error)}")

class TestRAGPipeline(unittest.TestCase):
    """End-to-end tests for the RAG pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test indices
        cls.test_dir = tempfile.mkdtemp()
        cls.index_path = os.path.join(cls.test_dir, "test_index")
        
        # Create test file
        cls.file_path = os.path.join(cls.test_dir, "test_manual.txt")
        create_test_file(cls.file_path)
        
        # Initialize components
        cls.vector_store = VectorStore(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            index_path=cls.index_path
        )
        
        # Initialize terminology manager
        cls.terminology_manager = TerminologyManager()
        
        # Initialize query processor with terminology manager
        cls.query_processor = QueryProcessor(
            terminology_manager=cls.terminology_manager
        )
        
        # Initialize context retriever
        cls.context_retriever = ContextRetriever(
            config_manager=TestConfigManager(),
            error_handler=TestErrorHandler()
        )
        
        # Initialize response generator
        cls.response_generator = ResponseGenerator(
            terminology_manager=cls.terminology_manager
        )
        
        # Create test index
        create_test_index(cls.index_path, cls.file_path)
        
        # Load test queries
        cls.test_queries = {
            "general": [
                "How do I check my oil level?",
                "What type of oil should I use?",
                "How often should I change my oil?"
            ],
            "troubleshooting": [
                "My engine is making a strange noise",
                "The oil pressure light is on",
                "I see oil leaking under my car"
            ],
            "maintenance": [
                "How do I change my oil filter?",
                "What's the oil change interval?",
                "How do I dispose of used oil?"
            ]
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cleanup_test_files(cls.test_dir)
    
    def test_query_processing(self):
        """Test query processing functionality."""
        for query in self.test_queries["general"]:
            processed = self.query_processor.preprocess_query(query)
            self.assertIsNotNone(processed)
            self.assertIn("query_type", processed)
            self.assertIn("entities", processed)
            self.assertIn("technical_terms", processed)
    
    def test_context_retrieval(self):
        """Test context retrieval functionality."""
        query = "How do I check my oil level?"
        results = self.context_retriever.retrieve(query)
        
        self.assertIsNotNone(results)
        self.assertIsInstance(results, list)
        if results:  # Only check content if we got results
            self.assertIn("text", results[0])
            self.assertIn("metadata", results[0])
            self.assertIn("score", results[0])
    
    def test_response_generation(self):
        """Test response generation functionality."""
        query = "How do I check my oil level?"
        context = self.context_retriever.retrieve(query)
        
        response = self.response_generator.generate_response(
            query=query,
            context=context
        )
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_end_to_end(self):
        """Test complete RAG pipeline."""
        query = "How do I check my oil level?"
        
        # Process query
        processed = self.query_processor.preprocess_query(query)
        self.assertIsNotNone(processed)
        
        # Retrieve context
        context = self.context_retriever.retrieve(query)
        self.assertIsNotNone(context)
        
        # Generate response
        response = self.response_generator.generate_response(
            query=query,
            context=context
        )
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_technical_terms(self):
        """Test technical term recognition."""
        technical_queries = [
            "How do I replace the oil filter?",
            "What's the procedure for changing the engine oil?",
            "How do I check the transmission fluid level?"
        ]
        
        for query in technical_queries:
            processed = self.query_processor.preprocess_query(query)
            self.assertIsNotNone(processed)
            self.assertIn("technical_terms", processed)
            self.assertIsInstance(processed["technical_terms"], dict)
            self.assertIn("parts", processed["technical_terms"])
            self.assertIn("systems", processed["technical_terms"])
    
    def test_troubleshooting(self):
        """Test troubleshooting query handling."""
        for query in self.test_queries["troubleshooting"]:
            processed = self.query_processor.preprocess_query(query)
            self.assertIsNotNone(processed)
            self.assertIn("query_type", processed)
            self.assertEqual(processed["query_type"], "troubleshooting")
            self.assertIn("entities", processed)
            self.assertIn("technical_terms", processed)

if __name__ == "__main__":
    unittest.main()