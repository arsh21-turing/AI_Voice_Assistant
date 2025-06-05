#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vector store module for the Voice-Powered Car Assistant.

This module provides functionality for creating and managing
vector embeddings and FAISS indices for semantic search.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer, util
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector embeddings and FAISS indices for semantic search.
    
    This class provides functionality to create embeddings using 
    sentence-transformers models, manage FAISS indices, and perform
    semantic search over document collections.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: Optional[str] = None):
        """
        Initialize the VectorStore.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            index_path: Path to load an existing FAISS index (optional)
        """
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.documents = []  # List of document texts
        self.metadatas = []  # List of metadata dictionaries
        
        logger.info(f"Initializing VectorStore with model: {model_name}")
        
        # Initialize the embedding model
        self._init_embedding_model()
        
        # Load existing index if provided
        if index_path:
            self.load(index_path)
    
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector representation of the text
        """
        if self.embedding_model is None:
            self._init_embedding_model()
            
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            raise
    
    def batch_compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Matrix of embeddings
        """
        if self.embedding_model is None:
            self._init_embedding_model()
            
        if not texts:
            return np.array([])
            
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error computing batch embeddings: {e}")
            raise
    
    def add_documents(
        self,
        documents: Optional[List[Any]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document objects or text strings to add
            texts: List of text strings if documents are not strings
            metadatas: List of metadata dictionaries for each document
            
        Returns:
            List of IDs of the added documents
        """
        # Determine the text content to embed
        if texts is not None:
            content_to_embed = texts
        elif documents is not None:
            content_to_embed = [str(doc) for doc in documents]
        else:
            logger.error("Either documents or texts must be provided")
            return []
        
        # Ensure we have matching metadata
        if metadatas is None:
            metadatas = [{} for _ in content_to_embed]
        
        if len(metadatas) != len(content_to_embed):
            logger.warning(f"Metadata count ({len(metadatas)}) doesn't match document count ({len(content_to_embed)}). Using empty metadata.")
            metadatas = [{} for _ in content_to_embed]
        
        # Compute embeddings for the documents
        embeddings = self.batch_compute_embeddings(content_to_embed)
        
        # Get document IDs (positions in the index)
        start_id = len(self.documents)
        doc_ids = list(range(start_id, start_id + len(content_to_embed)))
        
        # Create or update the FAISS index
        if self.index is None:
            # Create a new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"Created new FAISS index with dimension {dimension}")
        
        # Add the embeddings to the index
        self.index.add(embeddings)
        
        # Store the documents and metadata
        self.documents.extend(content_to_embed)
        self.metadatas.extend(metadatas)
        
        logger.info(f"Added {len(content_to_embed)} documents to vector store")
        return doc_ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a query.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return
            
        Returns:
            List of top k results with text, metadata, and similarity scores
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in vector store to search")
            return []
        
        if not query:
            logger.warning("Empty query provided")
            return []
            
        try:
            # Compute query embedding
            query_embedding = self.compute_embedding(query)
            
            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            # Format results
            results = []
            for i, doc_idx in enumerate(indices[0]):
                if doc_idx < 0 or doc_idx >= len(self.documents):
                    continue
                    
                # Calculate similarity score (convert distance to similarity)
                # L2 distance -> similarity conversion
                similarity = 1.0 / (1.0 + distances[0][i])
                
                result = {
                    "id": int(doc_idx),
                    "text": self.documents[doc_idx],
                    "metadata": self.metadatas[doc_idx],
                    "score": float(similarity)
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def save(self, path: str):
        """
        Save the FAISS index and document metadata to disk.
        
        Args:
            path: Base path to save the index and metadata (without extension)
        """
        if self.index is None:
            logger.warning("No index to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save documents and metadata
            with open(f"{path}_documents.json", 'w') as f:
                json.dump(self.documents, f)
                
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(self.metadatas, f)
                
            logger.info(f"Saved vector store to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load a FAISS index and document metadata from disk.
        
        Args:
            path: Base path to load from (without extension)
            
        Returns:
            True if loading was successful
        """
        try:
            # Check if files exist
            if not os.path.exists(f"{path}.faiss"):
                logger.error(f"Index file not found: {path}.faiss")
                return False
                
            if not os.path.exists(f"{path}_documents.json"):
                logger.error(f"Documents file not found: {path}_documents.json")
                return False
                
            if not os.path.exists(f"{path}_metadata.json"):
                logger.error(f"Metadata file not found: {path}_metadata.json")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.faiss")
            
            # Load documents
            with open(f"{path}_documents.json", 'r') as f:
                self.documents = json.load(f)
                
            # Load metadata
            with open(f"{path}_metadata.json", 'r') as f:
                self.metadatas = json.load(f)
                
            logger.info(f"Loaded vector store from {path} with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def clear(self):
        """Clear the vector store."""
        self.index = None
        self.documents = []
        self.metadatas = []
        logger.info("Vector store cleared")
    
    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document with its metadata, or None if not found
        """
        if doc_id < 0 or doc_id >= len(self.documents):
            return None
            
        return {
            "id": doc_id,
            "text": self.documents[doc_id],
            "metadata": self.metadatas[doc_id]
        }