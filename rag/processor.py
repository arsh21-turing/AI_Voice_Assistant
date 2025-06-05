#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query processing module for the RAG system.
Handles query normalization, entity extraction, and intent classification.
"""

from typing import Dict, List, Tuple, Optional, Set
import re
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
        
        # Define automotive-specific categories
        self.categories = {
            'maintenance': {
                'keywords': ['maintenance', 'service', 'check', 'inspect', 'replace', 'change'],
                'entities': ['filter', 'oil', 'brake', 'tire', 'battery', 'fluid']
            },
            'troubleshooting': {
                'keywords': ['problem', 'issue', 'error', 'warning', 'fix', 'repair'],
                'entities': ['engine', 'transmission', 'electrical', 'mechanical']
            },
            'safety': {
                'keywords': ['safety', 'warning', 'caution', 'danger', 'hazard'],
                'entities': ['airbag', 'seatbelt', 'brake', 'steering']
            },
            'specifications': {
                'keywords': ['spec', 'type', 'model', 'capacity', 'pressure'],
                'entities': ['engine', 'transmission', 'tire', 'battery']
            }
        }
    
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
        
        # Expand query with context
        expanded_query = self._expand_query(processed_query, query_type, entities)
        
        return expanded_query, query_type, entities
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query = query.lower()
        
        # Check for maintenance queries
        if any(phrase in query for phrase in ['how to', 'how do i', 'steps to', 'maintenance', 'service']):
            return 'maintenance'
        
        # Check for troubleshooting queries
        if any(phrase in query for phrase in ['fix', 'problem', 'issue', 'error', 'warning']):
            return 'troubleshooting'
        
        # Check for safety queries
        if any(phrase in query for phrase in ['safety', 'warning', 'caution', 'danger']):
            return 'safety'
        
        # Check for specification queries
        if any(phrase in query for phrase in ['spec', 'type', 'model', 'capacity']):
            return 'specification'
        
        # Check for general information queries
        if any(phrase in query for phrase in ['what is', 'tell me about', 'explain']):
            return 'information'
        
        # Check for command queries
        if any(phrase in query for phrase in ['do', 'perform', 'execute']):
            return 'command'
        
        return 'unknown'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from the query."""
        entities = {
            'parts': [],
            'actions': [],
            'specifications': [],
            'symptoms': [],
            'systems': []
        }
        
        # Extract parts
        for category in self.categories.values():
            for entity in category['entities']:
                if entity in query:
                    if entity in ['engine', 'transmission', 'electrical', 'mechanical']:
                        entities['systems'].append(entity)
                    else:
                        entities['parts'].append(entity)
        
        # Extract actions
        for category in self.categories.values():
            for keyword in category['keywords']:
                if keyword in query:
                    entities['actions'].append(keyword)
        
        # Extract specifications
        spec_patterns = [
            r'(\d+)\s*(?:psi|bar|kpa)',
            r'(\d+)\s*(?:liter|l|ml)',
            r'(\d+)\s*(?:volt|v)',
            r'(\d+)\s*(?:amp|a)'
        ]
        
        for pattern in spec_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities['specifications'].append(match.group(0))
        
        # Extract symptoms
        symptom_patterns = [
            r'(?:makes|producing|causing)\s+(\w+(?:\s+\w+)*)\s+(?:noise|sound)',
            r'(?:showing|displaying)\s+(\w+(?:\s+\w+)*)\s+(?:warning|light)',
            r'(?:leaking|dripping)\s+(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities['symptoms'].append(match.group(1))
        
        return entities
    
    def _expand_query(self, query: str, query_type: str, entities: Dict[str, List[str]]) -> str:
        """Expand query with relevant context."""
        expanded_terms = set()
        
        # Add original query terms
        expanded_terms.update(query.split())
        
        # Add related terms based on query type
        if query_type in self.categories:
            expanded_terms.update(self.categories[query_type]['keywords'])
            expanded_terms.update(self.categories[query_type]['entities'])
        
        # Add related terms for extracted entities
        for entity_list in entities.values():
            for entity in entity_list:
                # Add related terms from terminology manager
                related_terms = self.terminology_manager.get_related_terms(entity)
                expanded_terms.update(related_terms)
        
        # Join terms back into a query
        expanded_query = ' '.join(expanded_terms)
        
        return expanded_query