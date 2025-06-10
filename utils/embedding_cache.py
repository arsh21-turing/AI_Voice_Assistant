import numpy as np
from typing import Optional, Dict, List, Callable, Union, Any, Tuple
from collections import OrderedDict
import time
import os
import torch
import hashlib
import threading
from sentence_transformers import SentenceTransformer

class EmbeddingCache:
    """Cache system for storing and reusing embeddings with LRU (Least Recently Used) strategy."""
    
    def __init__(self, size_limit: int = 10000, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the embedding cache.
        
        Args:
            size_limit: Maximum number of embeddings to cache
            model_name: Optional SentenceTransformer model name to initialize embedder
            device: Device to use for embedding generation ('cuda', 'cpu', or None for auto-detect)
        """
        self.size_limit = size_limit
        self.cache = OrderedDict()  # LRU cache to store embeddings
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "last_used": {}, "created_at": time.time()}
        self.embedder = None
        
        # Initialize the embedding model if model_name is provided
        if model_name:
            self._initialize_embedder(model_name, device)
    
    def _initialize_embedder(self, model_name: str, device: Optional[str] = None) -> None:
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to use for embedding generation
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Initializing embedding model '{model_name}' on {device}")
        self.embedder = SentenceTransformer(
            model_name,
            device=device,
            cache_folder='./model_cache'  # Cache the model locally
        )
    
    def set_embedder(self, embedder: Union[SentenceTransformer, Callable]) -> None:
        """Set or update the embedding function or model.
        
        Args:
            embedder: SentenceTransformer model or function that generates embeddings
        """
        self.embedder = embedder
        print(f"Embedding model set: {type(embedder).__name__}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text, using cache if available.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            The embedding vector
            
        Raises:
            ValueError: If embedder is not set or text is empty
        """
        if not self.embedder:
            raise ValueError("Embedder not set. Either provide model_name when initializing or call set_embedder()")
        
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Normalize text to ensure consistent caching
        normalized_text = text.strip()
        cache_key = self._get_cache_key(normalized_text)
        
        with self.lock:
            # Check if embedding is in cache
            if cache_key in self.cache:
                self.stats["hits"] += 1
                self.stats["last_used"][cache_key] = time.time()
                
                # Move to end to mark as recently used (LRU behavior)
                embedding = self.cache.pop(cache_key)
                self.cache[cache_key] = embedding
                return embedding
            
            # Not in cache, need to compute embedding
            self.stats["misses"] += 1
            
            # Generate embedding
            try:
                if hasattr(self.embedder, 'encode'):
                    # If embedder is an object with encode method (like SentenceTransformer)
                    embedding = self.embedder.encode([normalized_text], convert_to_tensor=False, normalize_embeddings=True)[0]
                else:
                    # If embedder is a function
                    embedding = self.embedder(normalized_text)
                    
                # Convert to numpy array if not already
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Add to cache
                self.cache[cache_key] = embedding
                self.stats["last_used"][cache_key] = time.time()
                
                # If cache size exceeds limit, remove least recently used item
                if len(self.cache) > self.size_limit:
                    oldest_key, _ = self.cache.popitem(last=False)
                    if oldest_key in self.stats["last_used"]:
                        del self.stats["last_used"][oldest_key]
                    
                return embedding
                
            except Exception as e:
                print(f"Error generating embedding for text: {str(e)}")
                raise
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for multiple texts, efficiently using cache.
        
        Args:
            texts: List of texts to get embeddings for
            batch_size: Size of batches for processing uncached texts
            
        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])
        
        # Check cache for each text, collect those needing embeddings
        results = []
        uncached_texts = []
        uncached_indices = []
        
        with self.lock:
            for i, text in enumerate(texts):
                if not text or not isinstance(text, str):
                    raise ValueError(f"Invalid text at index {i}. Text must be a non-empty string")
                
                normalized_text = text.strip()
                cache_key = self._get_cache_key(normalized_text)
                
                if cache_key in self.cache:
                    # Get from cache
                    self.stats["hits"] += 1
                    self.stats["last_used"][cache_key] = time.time()
                    
                    # Move to end to mark as recently used
                    embedding = self.cache.pop(cache_key)
                    self.cache[cache_key] = embedding
                    results.append(embedding)
                else:
                    # Need to compute embedding
                    self.stats["misses"] += 1
                    uncached_texts.append(normalized_text)
                    uncached_indices.append(i)
                    # Placeholder in results
                    results.append(None)
        
        # Process uncached texts in batches
        if uncached_texts:
            try:
                # Process in batches to avoid memory issues for large inputs
                all_new_embeddings = []
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i+batch_size]
                    
                    # Generate embeddings for batch
                    if hasattr(self.embedder, 'encode'):
                        # If embedder is an object with encode method
                        batch_embeddings = self.embedder.encode(batch, convert_to_tensor=False, normalize_embeddings=True)
                    else:
                        # If embedder is a function
                        batch_embeddings = [self.embedder(text) for text in batch]
                    
                    all_new_embeddings.extend(batch_embeddings)
                
                # Update cache with new embeddings
                with self.lock:
                    for text, embedding in zip(uncached_texts, all_new_embeddings):
                        cache_key = self._get_cache_key(text)
                        
                        # Convert to numpy array if not already
                        if not isinstance(embedding, np.ndarray):
                            embedding = np.array(embedding)
                            
                        # Add to cache
                        self.cache[cache_key] = embedding
                        self.stats["last_used"][cache_key] = time.time()
                        
                        # If cache size exceeds limit, remove least recently used item
                        if len(self.cache) > self.size_limit:
                            oldest_key, _ = self.cache.popitem(last=False)
                            if oldest_key in self.stats["last_used"]:
                                del self.stats["last_used"][oldest_key]
                
                # Fill in the results
                for i, idx in enumerate(uncached_indices):
                    results[idx] = all_new_embeddings[i]
                
            except Exception as e:
                print(f"Error batch generating embeddings: {str(e)}")
                raise
        
        # Make sure all results are filled
        for i, res in enumerate(results):
            if res is None:
                raise RuntimeError(f"Missing embedding for text at index {i}")
        
        return np.array(results)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text.
        
        Args:
            text: The text to generate a key for
            
        Returns:
            A cache key string
        """
        # For short texts, just use the text itself
        if len(text) < 100:
            return text
            
        # For longer texts, use hash to avoid extremely long keys
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def clear(self) -> None:
        """Clear the cache to free memory."""
        with self.lock:
            self.cache.clear()
            self.stats["last_used"].clear()
            self.stats["hits"] = 0
            self.stats["misses"] = 0
            print("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cache performance.
        
        Returns:
            Dictionary with hits, misses and size statistics
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_ratio = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "total_requests": total_requests,
                "hit_ratio": hit_ratio,
                "cache_size": len(self.cache),
                "capacity": self.size_limit,
                "utilization": len(self.cache) / self.size_limit if self.size_limit > 0 else 0,
                "uptime_seconds": time.time() - self.stats["created_at"]
            }
    
    def resize(self, new_limit: int) -> None:
        """Change cache capacity, evicting oldest as needed."""
        with self.lock:
            self.size_limit = new_limit
            while len(self.cache) > new_limit:
                oldest_key, _ = self.cache.popitem(last=False)
                if oldest_key in self.stats["last_used"]:
                    del self.stats["last_used"][oldest_key]
    
    def __len__(self) -> int:
        """Get the current number of cached embeddings."""
        with self.lock:
            return len(self.cache)
