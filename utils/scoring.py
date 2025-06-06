#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relevance scoring module for the RAG system.
Handles semantic similarity, keyword matching, and automotive-specific scoring.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import re

class RelevanceScorer:
    """Handles relevance scoring for retrieved chunks."""
    
    def __init__(self, 
                 semantic_weight: float = 0.4,
                 keyword_weight: float = 0.3,
                 automotive_weight: float = 0.2,
                 recency_weight: float = 0.05,
                 length_weight: float = 0.05):
        """
        Initialize the RelevanceScorer.
        
        Args:
            semantic_weight: Weight for semantic similarity score
            keyword_weight: Weight for keyword matching score
            automotive_weight: Weight for automotive-specific bonuses
            recency_weight: Weight for recency score
            length_weight: Weight for length penalty
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.automotive_weight = automotive_weight
        self.recency_weight = recency_weight
        self.length_weight = length_weight
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
    
    def calculate_relevance(self, 
                          query: str, 
                          text: str, 
                          metadata: Dict[str, Any], 
                          base_score: float = 0.0) -> float:
        """
        Calculate relevance score for a text chunk.
        
        Args:
            query: The search query
            text: The text chunk to score
            metadata: Metadata about the text chunk
            base_score: Base similarity score (e.g., from vector search)
            
        Returns:
            float: Relevance score between 0 and 1
        """
        try:
            # Initialize scores
            scores = {
                'semantic': base_score,
                'keyword': self._calculate_keyword_score(query, text),
                'automotive': self._calculate_automotive_bonus(query, text),
                'recency': self._calculate_recency_score(metadata),
                'length': self._calculate_length_penalty(text)
            }
            
            # Calculate weighted sum
            final_score = (
                scores['semantic'] * self.semantic_weight +
                scores['keyword'] * self.keyword_weight +
                scores['automotive'] * self.automotive_weight +
                scores['recency'] * self.recency_weight +
                scores['length'] * self.length_weight
            )
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))
            
            self.logger.debug(f"Scores for text: {scores}, Final: {final_score}")
            return final_score
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance: {str(e)}")
            return 0.0
    
    def score_chunks(self, 
                    chunks: List[Dict[str, Any]], 
                    query: str, 
                    automotive_entities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Score a list of text chunks.
        
        Args:
            chunks: List of text chunks with metadata
            query: The search query
            automotive_entities: Optional list of automotive entities to consider
            
        Returns:
            List of chunks with added relevance scores
        """
        scored_chunks = []
        
        for chunk in chunks:
            try:
                # Get base score if available
                base_score = chunk.get('score', 0.0)
                
                # Calculate relevance
                relevance = self.calculate_relevance(
                    query=query,
                    text=chunk['text'],
                    metadata=chunk.get('metadata', {}),
                    base_score=base_score
                )
                
                # Add score to chunk
                scored_chunk = chunk.copy()
                scored_chunk['relevance_score'] = relevance
                scored_chunks.append(scored_chunk)
                
            except Exception as e:
                self.logger.error(f"Error scoring chunk: {str(e)}")
                continue
        
        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return scored_chunks
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword matching score."""
        try:
            # Normalize text
            query = query.lower()
            text = text.lower()
            
            # Split into words
            query_words = set(re.findall(r'\w+', query))
            text_words = set(re.findall(r'\w+', text))
            
            # Calculate intersection
            matches = query_words.intersection(text_words)
            
            # Calculate score
            if not query_words:
                return 0.0
                
            return len(matches) / len(query_words)
            
        except Exception as e:
            self.logger.error(f"Error calculating keyword score: {str(e)}")
            return 0.0
    
    def _calculate_automotive_bonus(self, query: str, text: str) -> float:
        """Calculate automotive-specific bonus score."""
        try:
            # Define automotive terms
            automotive_terms = {
                'maintenance': ['maintenance', 'service', 'repair', 'check', 'inspect'],
                'parts': ['engine', 'transmission', 'brake', 'filter', 'battery'],
                'safety': ['warning', 'caution', 'danger', 'safety', 'hazard'],
                'procedures': ['procedure', 'step', 'instruction', 'guide', 'manual']
            }
            
            # Normalize text
            query = query.lower()
            text = text.lower()
            
            # Calculate category matches
            category_scores = {}
            for category, terms in automotive_terms.items():
                matches = sum(1 for term in terms if term in text)
                category_scores[category] = matches / len(terms)
            
            # Return highest category score
            return max(category_scores.values()) if category_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating automotive bonus: {str(e)}")
            return 0.0
    
    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on metadata."""
        try:
            # Get timestamp from metadata
            timestamp = metadata.get('timestamp')
            if not timestamp:
                return 0.5  # Default score if no timestamp
            
            # Convert to datetime if string
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Calculate recency score
            now = datetime.now()
            age_days = (now - timestamp).days
            
            # Score decreases with age
            return max(0.0, 1.0 - (age_days / 365))  # Linear decay over a year
            
        except Exception as e:
            self.logger.error(f"Error calculating recency score: {str(e)}")
            return 0.5
    
    def _calculate_length_penalty(self, text: str) -> float:
        """Calculate length penalty score."""
        try:
            # Get word count
            word_count = len(text.split())
            
            # Ideal length range (adjust as needed)
            min_words = 10
            max_words = 100
            
            if word_count < min_words:
                # Penalize very short texts
                return word_count / min_words
            elif word_count > max_words:
                # Penalize very long texts
                return max_words / word_count
            else:
                # No penalty for texts in ideal range
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating length penalty: {str(e)}")
            return 0.5 