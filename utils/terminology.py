#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Terminology management module for the RAG system.
Provides functionality for normalizing technical terms and queries.
"""

from typing import Dict, Optional, List
import re
import logging

class TerminologyManager:
    """Handles normalization of technical terms and queries."""
    
    def __init__(self):
        """Initialize the TerminologyManager."""
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize terminology mappings
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
        
        # Initialize automotive-specific patterns
        self._patterns = {
            'measurements': r'\b(\d+)\s*(psi|rpm|mph|mpg)\b',
            'part_numbers': r'\b([A-Z0-9-]+)\b',
            'model_years': r'\b(19|20)\d{2}\b'
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize a complete query string.
        
        Args:
            query: The query string to normalize
            
        Returns:
            Normalized query string
        """
        try:
            # Convert to lowercase
            query = query.lower()
            
            # Split into words
            words = query.split()
            
            # Normalize each word
            normalized_words = []
            for word in words:
                normalized_word = self.normalize_term(word)
                normalized_words.append(normalized_word)
            
            # Join words back into a query
            normalized_query = ' '.join(normalized_words)
            
            # Apply pattern-based normalization
            normalized_query = self._apply_pattern_normalization(normalized_query)
            
            return normalized_query
            
        except Exception as e:
            self.logger.error(f"Error normalizing query: {str(e)}")
            return query
    
    def normalize_term(self, term: str) -> str:
        """
        Normalize a technical term.
        
        Args:
            term: The term to normalize
            
        Returns:
            Normalized term
        """
        try:
            # Convert to lowercase for consistent matching
            term = term.lower()
            
            # Check if term exists in mapping
            if term in self._terminology_mapping:
                return self._terminology_mapping[term]
            
            return term
            
        except Exception as e:
            self.logger.error(f"Error normalizing term: {str(e)}")
            return term
    
    def _apply_pattern_normalization(self, text: str) -> str:
        """
        Apply pattern-based normalization to text.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text
        """
        try:
            # Normalize measurements
            for unit in ['psi', 'rpm', 'mph', 'mpg']:
                pattern = self._patterns['measurements']
                matches = re.finditer(pattern, text)
                for match in matches:
                    value, unit = match.groups()
                    normalized = f"{value} {self.normalize_term(unit)}"
                    text = text.replace(match.group(), normalized)
            
            # Normalize part numbers (keep as is, just ensure proper spacing)
            pattern = self._patterns['part_numbers']
            matches = re.finditer(pattern, text)
            for match in matches:
                part_number = match.group()
                # Add spaces around part numbers if needed
                text = text.replace(part_number, f" {part_number} ").strip()
            
            # Normalize model years (keep as is)
            pattern = self._patterns['model_years']
            matches = re.finditer(pattern, text)
            for match in matches:
                year = match.group()
                # Add spaces around years if needed
                text = text.replace(year, f" {year} ").strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error applying pattern normalization: {str(e)}")
            return text
    
    def add_term(self, term: str, normalized: str):
        """
        Add a new term to the terminology mapping.
        
        Args:
            term: The term to add
            normalized: The normalized form of the term
        """
        try:
            self._terminology_mapping[term.lower()] = normalized.lower()
            self.logger.debug(f"Added term mapping: {term} -> {normalized}")
        except Exception as e:
            self.logger.error(f"Error adding term: {str(e)}")
    
    def remove_term(self, term: str):
        """
        Remove a term from the terminology mapping.
        
        Args:
            term: The term to remove
        """
        try:
            self._terminology_mapping.pop(term.lower(), None)
            self.logger.debug(f"Removed term: {term}")
        except Exception as e:
            self.logger.error(f"Error removing term: {str(e)}")
    
    def get_related_terms(self, term: str) -> List[str]:
        """
        Get related terms for a given term.
        
        Args:
            term: The term to find related terms for
            
        Returns:
            List of related terms
        """
        try:
            term = term.lower()
            related_terms = []
            
            # Find terms that contain the given term
            for key in self._terminology_mapping:
                if term in key:
                    related_terms.append(key)
            
            # Find terms that the given term contains
            for key in self._terminology_mapping:
                if key in term:
                    related_terms.append(key)
            
            return list(set(related_terms))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error getting related terms: {str(e)}")
            return []