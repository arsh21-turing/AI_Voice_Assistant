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
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", index_path: str = "data/vector_store"):
        """
        Initialize the VectorStore.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            index_path: Path to load an existing FAISS index (optional)
        """
        self.model_name = model_name
        self.index_path = index_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Initialized embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise
            
        # Initialize or load FAISS index
        try:
            self._initialize_index()
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS index: {e}")
            raise
            
        # Store document metadata
        self.documents = []
    
    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        try:
            # Ensure directory exists
            os.makedirs(self.index_path, exist_ok=True)
            
            index_file = os.path.join(self.index_path, "index.faiss")
            metadata_file = os.path.join(self.index_path, "metadata.json")
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                # Load existing index
                self.index = faiss.read_index(index_file)
                # Load metadata
                with open(metadata_file, "r") as f:
                    self.documents = json.load(f)
                self.logger.info(f"Loaded existing index from {self.index_path}")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(384)  # 384 is the dimension for all-MiniLM-L6-v2
                self.documents = []
                
                # Save empty index and metadata
                faiss.write_index(self.index, index_file)
                with open(metadata_file, "w") as f:
                    json.dump(self.documents, f)
                    
                self.logger.info(f"Created new index at {self.index_path}")
                
        except Exception as e:
            self.logger.error(f"Error initializing index: {e}")
            raise
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            self.logger.error(f"Error computing embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Any]):
        """Add documents to the vector store."""
        try:
            # Extract text and metadata
            texts = []
            metadata_list = []
            
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    # LangChain Document
                    text = doc.page_content
                    metadata = doc.metadata
                elif isinstance(doc, dict):
                    # Dictionary with text key
                    text = doc.get('text', '')
                    metadata = {k: v for k, v in doc.items() if k != 'text'}
                elif isinstance(doc, str):
                    # Plain text
                    text = doc
                    metadata = {}
                else:
                    self.logger.warning(f"Unknown document format: {type(doc)}")
                    continue
                    
                texts.append(text)
                metadata_list.append(metadata)
                
            if not texts:
                return
                
            # Compute embeddings
            embeddings = self.compute_embeddings(texts)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            for text, metadata in zip(texts, metadata_list):
                self.documents.append({
                    'text': text,
                    'metadata': metadata
                })
                
            # Save index and metadata
            self.save_index()
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Compute query embedding
            query_embedding = self.compute_embeddings([query])[0]
            
            # Search in FAISS index
            distances, indices = self.index.search(
                np.array([query_embedding]), 
                min(top_k, len(self.documents))
            )
            
            # Format results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    # Convert distance to similarity score (0-1 range)
                    similarity = float(np.exp(-distance))
                    results.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'score': similarity
                    })
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
    
    def save_index(self):
        """Save the FAISS index and metadata."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
            
            # Save metadata
            with open(os.path.join(self.index_path, "metadata.json"), "w") as f:
                json.dump(self.documents, f)
                
            self.logger.info(f"Saved index and metadata to {self.index_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
            raise
    
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
            "text": self.documents[doc_id]['text'],
            "metadata": self.documents[doc_id]['metadata']
        }