#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query processing module for the RAG system.
Handles query normalization, entity extraction, and intent classification.
"""

from typing import Dict, List, Tuple, Optional
from utils.terminology import TerminologyManager

class QueryProcessor:
    """Handles query preprocessing and normalization."""
    
    def __init__(self, terminology_manager: Optional[TerminologyManager] = None):
        """
        Initialize the QueryProcessor.
        
        Args:
            terminology_manager: Optional instance of TerminologyManager for term normalization
        """
        self.terminology_manager = terminology_manager or TerminologyManager()
    
    def process(self, query: str) -> str:
        """
        Process and normalize a query.
        
        Args:
            query: The raw query string
            
        Returns:
            str: Normalized query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove common filler words
        filler_words = ['how', 'do', 'i', 'can', 'you', 'tell', 'me', 'about', 'what', 'is', 'the']
        words = query.split()
        words = [w for w in words if w not in filler_words]
        
        # Normalize technical terms
        normalized_words = []
        for word in words:
            normalized_word = self.terminology_manager.normalize_term(word)
            normalized_words.append(normalized_word)
        
        # Join words back into a query
        processed_query = ' '.join(normalized_words)
        
        return processed_query
    
    def preprocess_query(self, query: str) -> Tuple[str, str, Dict[str, List[str]]]:
        """
        Preprocess query and extract entities and intent.
        
        Args:
            query: The raw query string
            
        Returns:
            Tuple[str, str, Dict[str, List[str]]]: 
                - Processed query
                - Query type (information, how_to, command, troubleshooting, unknown)
                - Extracted entities
        """
        # Process the query
        processed_query = self.process(query)
        
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # Extract entities
        entities = self._extract_entities(processed_query)
        
        return processed_query, query_type, entities
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query = query.lower()
        
        if any(phrase in query for phrase in ['how to', 'how do i', 'steps to']):
            return 'how_to'
        elif any(phrase in query for phrase in ['what is', 'tell me about', 'explain']):
            return 'information'
        elif any(phrase in query for phrase in ['fix', 'problem', 'issue', 'error']):
            return 'troubleshooting'
        elif any(phrase in query for phrase in ['do', 'perform', 'execute']):
            return 'command'
        else:
            return 'unknown'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query."""
        entities = {
            'parts': [],
            'actions': [],
            'specifications': []
        }
        
        # Simple entity extraction based on known terms
        # This should be enhanced with proper NER in production
        if 'filter' in query:
            entities['parts'].append('filter')
        if 'oil' in query:
            entities['parts'].append('oil')
        
        return entities