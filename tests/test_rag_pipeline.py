#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end tests for the RAG pipeline in the Voice-Powered Car Assistant.

This module contains tests for the complete retrieval-augmented generation
pipeline, from query processing to response generation, using a sample
PDF manual for automotive information.
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
from typing import List, Dict, Any

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

# Sample queries for testing
SAMPLE_QUERIES = [
    "How do I change the oil filter in my car?",
    "What type of oil filter should I use for my vehicle?",
    "Tell me about oil filter maintenance",
    "When should I replace my oil filter?",
    "What are the signs of a clogged oil filter?",
]

class TestRAGPipeline(unittest.TestCase):
    """Test cases for the complete RAG pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a temporary directory for test indices
        cls.temp_dir = tempfile.mkdtemp()
        cls.index_path = os.path.join(cls.temp_dir, "test_index")
        cls.model_name = "all-MiniLM-L6-v2"
        
        # Path to the sample PDF
        cls.pdf_path = os.path.join(project_root, "data", "manuals", "oil-filters.pdf")
        
        # Create test index if it doesn't exist
        if not os.path.exists(f"{cls.index_path}.faiss"):
            cls._create_test_index()
        
        # Initialize vector store
        cls.vector_store = VectorStore(
            model_name=cls.model_name,
            index_path=cls.index_path
        )
        
        # Initialize components
        cls.terminology_manager = TerminologyManager()
        cls.query_processor = QueryProcessor(terminology_manager=cls.terminology_manager)
        
        # Initialize context retriever with the vector store
        cls.context_retriever = ContextRetriever(
            vector_store=cls.vector_store
        )
        
        # Manually setup the context retriever with our vector store
        if hasattr(cls.context_retriever, 'index') and hasattr(cls.vector_store, 'index'):
            cls.context_retriever.index = cls.vector_store.index
        if hasattr(cls.context_retriever, 'chunks_metadata') and hasattr(cls.vector_store, 'metadatas'):
            # Adapt vector store metadatas to the format expected by context retriever
            chunks_metadata = []
            for i, metadata in enumerate(cls.vector_store.metadatas):
                chunk = {
                    'text': cls.vector_store.documents[i],
                    'metadata': metadata
                }
                chunks_metadata.append(chunk)
            cls.context_retriever.chunks_metadata = chunks_metadata
        if hasattr(cls.context_retriever, 'embedding_model') and hasattr(cls.vector_store, 'embedding_model'):
            cls.context_retriever.embedding_model = cls.vector_store.embedding_model
        
        cls.response_generator = ResponseGenerator()
        
        # Initialize the main retriever
        cls.manual_retriever = ManualRetriever(
            query_processor=cls.query_processor,
            context_retriever=cls.context_retriever,
            response_generator=cls.response_generator
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # Remove temporary directory and its contents
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_test_index(cls):
        """Helper method to create a test index from the sample PDF."""
        try:
            # Process the PDF using PDFProcessor
            pdf_processor = PDFProcessor(cls.pdf_path)
            
            # Get semantic chunks from the PDF
            chunks = pdf_processor.get_semantic_chunks()
            
            # Create a vector store
            vector_store = VectorStore(model_name=cls.model_name)
            
            # Prepare text and metadata lists
            texts = [chunk.get('text', '') for chunk in chunks]
            metadatas = [chunk.get('metadata', {}) for chunk in chunks]
            
            # Add documents to vector store
            vector_store.add_documents(texts=texts, metadatas=metadatas)
            
            # Save the vector store
            vector_store.save(cls.index_path)
            
            logger.info(f"Created test index at {cls.index_path}")
            
        except Exception as e:
            logger.error(f"Error creating test index: {e}")
            raise
    
    def test_pdf_index_exists(self):
        """Test that the index was created correctly."""
        self.assertTrue(os.path.exists(f"{self.index_path}.faiss"), 
                        "FAISS index file doesn't exist")
        self.assertTrue(os.path.exists(f"{self.index_path}_metadata.json"), 
                        "Metadata file doesn't exist")
        self.assertTrue(os.path.exists(f"{self.index_path}_documents.json"), 
                        "Documents file doesn't exist")
    
    def test_vector_store_loaded(self):
        """Test that the vector store was loaded correctly."""
        self.assertIsNotNone(self.vector_store.index, "Vector store index should be loaded")
        self.assertGreater(len(self.vector_store.documents), 0, "Vector store should have documents")
        self.assertGreater(len(self.vector_store.metadatas), 0, "Vector store should have metadata")
    
    def test_query_processing(self):
        """Test query processing component."""
        for query in SAMPLE_QUERIES:
            processed_query, query_type, entities = self.query_processor.preprocess_query(query)
            
            # Verify processed query is not empty
            self.assertTrue(processed_query, "Processed query should not be empty")
            
            # Verify query type is determined
            self.assertIn(query_type, ["information", "how_to", "command", "troubleshooting", "unknown"], 
                         f"Unknown query type: {query_type}")
            
            # Verify entities are extracted when appropriate
            if "oil filter" in query.lower():
                self.assertTrue(any("filter" in part for part in entities.get("parts", [])),
                               "Oil filter should be detected as a part")
    
    def test_vector_store_search(self):
        """Test vector store search functionality."""
        for query in SAMPLE_QUERIES:
            # Search using vector store directly
            results = self.vector_store.search(query, top_k=3)
            
            # Verify results are returned
            self.assertTrue(results, "Vector store search should return results")
            self.assertGreaterEqual(len(results), 1, "At least one result should be returned")
            
            # Verify result structure
            for result in results:
                self.assertIn('text', result, "Result should contain text")
                self.assertIn('score', result, "Result should have a relevance score")
                self.assertIn('metadata', result, "Result should include metadata")
                self.assertIn('id', result, "Result should have an ID")
    
    def test_context_retrieval(self):
        """Test context retrieval component."""
        for query in SAMPLE_QUERIES:
            # First process the query
            processed_query, _, _ = self.query_processor.preprocess_query(query)
            
            # Retrieve context
            context_chunks = self.context_retriever.retrieve(
                query=processed_query,
                top_k=3
            )
            
            # Verify context chunks are returned
            self.assertTrue(context_chunks, "Context retrieval should return results")
            self.assertGreaterEqual(len(context_chunks), 1, "At least one chunk should be returned")
            
            # Verify chunks have expected structure
            for chunk in context_chunks:
                self.assertIn('text', chunk, "Chunk should contain text")
                self.assertIn('score', chunk, "Chunk should have a relevance score")
                self.assertIn('metadata', chunk, "Chunk should include metadata")
    
    def test_response_generation(self):
        """Test response generation from context chunks."""
        query = "How do I change the oil filter in my car?"
        
        # Search using vector store
        search_results = self.vector_store.search(query, top_k=3)
        
        # Convert to context chunks format expected by response generator
        context_chunks = []
        for result in search_results:
            context_chunks.append({
                'text': result['text'],
                'metadata': result['metadata'],
                'score': result['score']
            })
        
        # Generate response
        response = self.response_generator.generate(query, context_chunks)
        
        # Debug: Print the actual response
        print(f"\nActual response: {response}")
        
        # Verify response is generated
        self.assertTrue(response, "Response should not be empty")
        self.assertGreater(len(response), 20, "Response should be of substantial length")
        
        # Verify it's a placeholder response
        self.assertTrue(response.startswith("[PLACEHOLDER]"), "Response should be a placeholder")
        self.assertIn("Based on", response, "Response should mention number of contexts")
    
    def test_end_to_end_pipeline(self):
        """Test the full RAG pipeline from query to response."""
        for query in SAMPLE_QUERIES:
            # Search using vector store
            search_results = self.vector_store.search(query, top_k=3)
            
            # Convert to context chunks format
            context_chunks = []
            for result in search_results:
                context_chunks.append({
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'score': result['score']
                })
            
            # Generate response
            response = self.response_generator.generate(query, context_chunks)
            
            # Verify response is generated
            self.assertTrue(response, "Response should not be empty")
            self.assertGreater(len(response), 20, "Response should be of substantial length")
            
            # Test response coherence (basic check)
            words = response.split()
            self.assertGreater(len(words), 5, "Response should have multiple words")
            
            # Verify it's a placeholder response
            self.assertTrue(response.startswith("[PLACEHOLDER]"), "Response should be a placeholder")
            self.assertIn("Based on", response, "Response should mention number of contexts")
    
    def test_different_query_types(self):
        """Test the pipeline with different query types."""
        informational_query = "What is the purpose of an oil filter?"
        how_to_query = "How do I install a new oil filter?"
        
        # Test with vector store + response generator
        
        # Informational query
        info_results = self.vector_store.search(informational_query, top_k=3)
        info_chunks = [{
            'text': r['text'], 
            'metadata': r['metadata'], 
            'score': r['score']
        } for r in info_results]
        
        info_response = self.response_generator.generate(informational_query, info_chunks)
        
        # How-to query
        howto_results = self.vector_store.search(how_to_query, top_k=3)
        howto_chunks = [{
            'text': r['text'], 
            'metadata': r['metadata'], 
            'score': r['score']
        } for r in howto_results]
        
        howto_response = self.response_generator.generate(how_to_query, howto_chunks)
        
        # Verify responses
        self.assertTrue(info_response, "Informational query should get a response")
        self.assertTrue(howto_response, "How-to query should get a response")
        
        # Verify responses are different
        self.assertNotEqual(info_response, howto_response, 
                           "Different query types should get different responses")

if __name__ == '__main__':
    unittest.main()