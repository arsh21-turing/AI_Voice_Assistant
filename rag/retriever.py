#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual retriever module for the RAG system.
Orchestrates the RAG pipeline components.
"""

from typing import Dict, Any, Optional, List
from .processor import QueryProcessor
from .context import ContextRetriever
from .generator import ResponseGenerator

class ManualRetriever:
    """Orchestrates the RAG pipeline components."""
    
    def __init__(
        self,
        config_manager=None,
        error_handler=None,
        query_processor: Optional[QueryProcessor] = None,
        context_retriever: Optional[ContextRetriever] = None,
        response_generator: Optional[ResponseGenerator] = None
    ):
        """
        Initialize the ManualRetriever.
        
        Args:
            config_manager: Configuration manager instance
            error_handler: Error handler instance
            query_processor: Optional instance of QueryProcessor
            context_retriever: Optional instance of ContextRetriever
            response_generator: Optional instance of ResponseGenerator
        """
        # Get configuration if not provided
        if config_manager is None:
            from config import get_config
            config_manager = get_config()
            
        # Create error handler if not provided
        if error_handler is None:
            from utils.error_handler import ErrorHandler
            error_handler = ErrorHandler(config_manager)
            
        # Initialize components
        self.query_processor = query_processor or QueryProcessor()
        self.context_retriever = context_retriever or ContextRetriever(
            config_manager=config_manager,
            error_handler=error_handler
        )
        self.response_generator = response_generator or ResponseGenerator(
            config_manager=config_manager
        )
    
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The query string
            
        Returns:
            List of dictionaries containing document information
        """
        # Process the query
        processed_query, _, _ = self.query_processor.preprocess_query(query)
        
        # Retrieve contexts
        contexts = self.context_retriever.retrieve_context(processed_query)
        
        # Rank contexts
        ranked_contexts = self.context_retriever.rank_contexts(processed_query, contexts)
        
        # Filter contexts
        filtered_contexts = self.context_retriever.filter_contexts(ranked_contexts)
        
        return filtered_contexts
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: The raw query string
            
        Returns:
            Dictionary containing:
            - processed_query: The processed query
            - query_type: Type of query
            - entities: Extracted entities
            - contexts: Retrieved contexts
            - response: Generated response
        """
        # 1. Process the query
        processed_query, query_type, entities = self.query_processor.preprocess_query(query)
        
        # 2. Retrieve relevant contexts
        contexts = self.context_retriever.retrieve_context(processed_query)
        
        # 3. Rank contexts by relevance
        ranked_contexts = self.context_retriever.rank_contexts(processed_query, contexts)
        
        # 4. Filter contexts by minimum relevance score
        filtered_contexts = self.context_retriever.filter_contexts(ranked_contexts)
        
        # 5. Generate response
        response = self.response_generator.generate(filtered_contexts, processed_query, query_type)
        
        return {
            'processed_query': processed_query,
            'query_type': query_type,
            'entities': entities,
            'contexts': filtered_contexts,
            'response': response
        }
