#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Terminology management module for the RAG system.
Provides functionality for normalizing technical terms.
"""

from typing import Dict, Optional

class TerminologyManager:
    """Handles normalization of technical terms."""
    
    def __init__(self):
        """Initialize the TerminologyManager."""
        self._terminology_mapping = {
            # Common variations
            'ac': 'air conditioning',
            'a/c': 'air conditioning',
            'a.c.': 'air conditioning',
            'a c': 'air conditioning',
            
            # Parts and components
            'oil filter': 'engine oil filter',
            'air filter': 'engine air filter',
            'fuel filter': 'fuel system filter',
            'cabin filter': 'cabin air filter',
            
            # Actions
            'change oil': 'replace engine oil',
            'change filter': 'replace filter',
            'check oil': 'inspect engine oil level',
            'check tire': 'inspect tire pressure',
            
            # Measurements
            'psi': 'pounds per square inch',
            'rpm': 'revolutions per minute',
            'mph': 'miles per hour',
            'mpg': 'miles per gallon'
        }
    
    def normalize_term(self, term: str) -> str:
        """
        Normalize a technical term.
        
        Args:
            term: The term to normalize
            
        Returns:
            Normalized term
        """
        # Convert to lowercase for consistent matching
        term = term.lower()
        
        # Check if term exists in mapping
        if term in self._terminology_mapping:
            return self._terminology_mapping[term]
        
        return term
    
    def add_term(self, term: str, normalized: str):
        """
        Add a new term to the terminology mapping.
        
        Args:
            term: The term to add
            normalized: The normalized form of the term
        """
        self._terminology_mapping[term.lower()] = normalized.lower()
    
    def remove_term(self, term: str):
        """
        Remove a term from the terminology mapping.
        
        Args:
            term: The term to remove
        """
        self._terminology_mapping.pop(term.lower(), None)