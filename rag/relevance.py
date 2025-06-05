import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class RelevanceScorer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize the embedding model
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.logger.info("Initialized relevance scorer model")
        except Exception as e:
            self.logger.error(f"Failed to initialize relevance scorer model: {e}")
            raise
            
    def score(self, query: str, text: str) -> float:
        """
        Score the relevance of a text to a query.
        
        Args:
            query: The query string
            text: The text to score
            
        Returns:
            float: Relevance score between 0 and 1
        """
        try:
            # Compute embeddings
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            text_embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Compute cosine similarity
            similarity = float(np.dot(query_embedding, text_embedding) / 
                            (np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)))
            
            # Normalize to 0-1 range
            score = (similarity + 1) / 2
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error computing relevance score: {e}")
            return 0.0
            
    def score_batch(self, query: str, texts: List[str]) -> List[float]:
        """
        Score multiple texts against a query.
        
        Args:
            query: The query string
            texts: List of texts to score
            
        Returns:
            List[float]: List of relevance scores
        """
        try:
            # Compute query embedding
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Compute text embeddings
            text_embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Compute cosine similarities
            similarities = np.dot(text_embeddings, query_embedding) / (
                np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Normalize to 0-1 range
            scores = [(s + 1) / 2 for s in similarities]
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error computing batch relevance scores: {e}")
            return [0.0] * len(texts) 