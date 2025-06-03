#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Context retrieval module for the RAG system.
Handles retrieval of relevant context from the knowledge base.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from utils.vector_store import VectorStore

class ContextRetriever:
    """Handles retrieval of relevant context from the knowledge base."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the ContextRetriever.
        
        Args:
            vector_store: Optional instance of VectorStore for similarity search
        """
        self.vector_store = vector_store or VectorStore()
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The processed query
            top_k: Number of most relevant contexts to retrieve
            
        Returns:
            List of relevant context documents
        """
        # Perform similarity search directly using the vector store
        results = self.vector_store.search(query, top_k=top_k)
        
        # Format results
        contexts = []
        for result in results:
            context = {
                'text': result['text'],
                'metadata': result['metadata'],
                'score': result['score']
            }
            contexts.append(context)
        
        return contexts
    
    def rank_contexts(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank retrieved contexts by relevance.
        
        Args:
            query: The processed query
            contexts: List of context documents
            
        Returns:
            Ranked list of context documents
        """
        # Get query embedding
        query_embedding = self.vector_store.compute_embedding(query)
        
        # Calculate relevance scores
        for context in contexts:
            context_embedding = self.vector_store.compute_embedding(context['text'])
            similarity = np.dot(query_embedding, context_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
            )
            context['relevance_score'] = float(similarity)
        
        # Sort by relevance score
        ranked_contexts = sorted(contexts, key=lambda x: x['relevance_score'], reverse=True)
        
        return ranked_contexts
    
    def filter_contexts(self, contexts: List[Dict[str, Any]], min_score: float = 0.5) -> List[Dict[str, Any]]:
        """
        Filter contexts by minimum relevance score.
        
        Args:
            contexts: List of context documents
            min_score: Minimum relevance score threshold
            
        Returns:
            Filtered list of context documents
        """
        return [ctx for ctx in contexts if ctx['relevance_score'] >= min_score]