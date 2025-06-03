#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual retriever module for the RAG system.
Orchestrates the RAG pipeline components.
"""

from typing import Dict, Any, Optional
from .processor import QueryProcessor
from .context import ContextRetriever
from .generator import ResponseGenerator

class ManualRetriever:
    """Orchestrates the RAG pipeline components."""
    
    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        context_retriever: Optional[ContextRetriever] = None,
        response_generator: Optional[ResponseGenerator] = None
    ):
        """
        Initialize the ManualRetriever.
        
        Args:
            query_processor: Optional instance of QueryProcessor
            context_retriever: Optional instance of ContextRetriever
            response_generator: Optional instance of ResponseGenerator
        """
        self.query_processor = query_processor or QueryProcessor()
        self.context_retriever = context_retriever or ContextRetriever()
        self.response_generator = response_generator or ResponseGenerator()
    
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
        contexts = self.context_retriever.retrieve(processed_query)
        
        # 3. Rank contexts by relevance
        ranked_contexts = self.context_retriever.rank_contexts(processed_query, contexts)
        
        # 4. Filter contexts by minimum relevance score
        filtered_contexts = self.context_retriever.filter_contexts(ranked_contexts)
        
        # 5. Generate response
        response = self.response_generator.generate(processed_query, filtered_contexts)
        
        return {
            'processed_query': processed_query,
            'query_type': query_type,
            'entities': entities,
            'contexts': filtered_contexts,
            'response': response
        }
