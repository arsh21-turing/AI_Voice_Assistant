"""
End-to-end tests for the PDF embedding pipeline.
Tests the full embedding pipeline from PDF loading to FAISS index creation.
"""
import unittest
import os
import sys
import numpy as np
from pathlib import Path
import pytest
import faiss

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.embed_pipeline import (
    generate_chunks,
    generate_embeddings,
    create_faiss_index
)

class TestEmbedPipeline(unittest.TestCase):
    """Test cases for the embedding pipeline utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods."""
        # Get the project root directory
        cls.root_dir = Path(__file__).resolve().parent.parent
        
        # PDF test file path
        cls.pdf_path = cls.root_dir / "data" / "manuals" / "oil-filters.pdf"
        
        # Skip tests if the PDF file doesn't exist
        if not cls.pdf_path.exists():
            pytest.skip(f"Test PDF file not found at {cls.pdf_path}")
    
    def test_pdf_chunks_generation(self):
        """Test that chunks are generated correctly from the PDF."""
        # Generate chunks from the test PDF
        chunks = generate_chunks(str(self.pdf_path))
        
        # Verify that chunks were generated
        self.assertIsNotNone(chunks, "Chunks should not be None")
        self.assertGreater(len(chunks), 0, "Should have extracted at least one chunk")
        
        # Verify that each chunk has the expected structure
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, dict, f"Chunk {i} should be a dictionary")
            self.assertIn('text', chunk, f"Chunk {i} should have 'text' key")
            self.assertGreater(len(chunk['text']), 0, f"Chunk {i} should have non-empty text")
            
            # Check for metadata (page numbers should be present)
            self.assertIn('page', chunk, f"Chunk {i} should have page number metadata")
            
        print(f"Successfully generated {len(chunks)} chunks from {self.pdf_path}")
        
        return chunks  # Return for use in subsequent tests
    
    def test_embedding_generation(self):
        """Test that embeddings are generated correctly from chunks."""
        # Generate chunks
        chunks = self.test_pdf_chunks_generation()
        
        # Generate embeddings
        embeddings, metadata, texts = generate_embeddings(chunks)
        
        # Verify that embeddings were generated
        self.assertIsNotNone(embeddings, "Embeddings should not be None")
        self.assertIsInstance(embeddings, np.ndarray, "Embeddings should be a numpy array")
        
        # Verify the embeddings shape
        self.assertEqual(embeddings.shape[0], len(chunks), 
                        "Number of embeddings should match number of chunks")
        self.assertGreater(embeddings.shape[1], 0, 
                          "Embedding dimension should be greater than 0")
        
        # Verify metadata
        self.assertEqual(len(metadata), len(chunks),
                        "Metadata list should have same length as chunks")
        
        # Verify texts
        self.assertEqual(len(texts), len(chunks),
                        "Texts list should have same length as chunks")
        
        print(f"Successfully generated embeddings with shape {embeddings.shape}")
        
        return embeddings, metadata, texts
    
    def test_faiss_index_creation(self):
        """Test that a FAISS index is created correctly from embeddings."""
        # Generate embeddings
        embeddings, _, _ = self.test_embedding_generation()
        
        # Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Verify that the index was created
        self.assertIsNotNone(index, "FAISS index should not be None")
        self.assertIsInstance(index, faiss.Index, "Should be a FAISS index instance")
        
        # Verify the index properties
        self.assertEqual(index.ntotal, embeddings.shape[0],
                        "Index should contain same number of vectors as embeddings")
        self.assertEqual(index.d, embeddings.shape[1],
                        "Index dimension should match embedding dimension")
        
        print(f"Successfully created FAISS index with {index.ntotal} vectors of dimension {index.d}")
        
        return index
    
    def test_end_to_end_pipeline(self):
        """Test the full pipeline from PDF to searchable index."""
        # Generate chunks, embeddings, and create index
        chunks = generate_chunks(str(self.pdf_path))
        embeddings, metadata, texts = generate_embeddings(chunks)
        index = create_faiss_index(embeddings)
        
        # Verify the entire pipeline worked correctly
        self.assertEqual(len(chunks), len(metadata), 
                        "Number of chunks should match metadata entries")
        self.assertEqual(len(chunks), embeddings.shape[0], 
                        "Number of chunks should match number of embeddings")
        self.assertEqual(len(chunks), index.ntotal, 
                        "Number of chunks should match vectors in index")
        
        # Verify that dimensions are consistent
        self.assertEqual(embeddings.shape[1], index.d,
                        "Embedding dimension should match index dimension")
        
        print(f"Successfully tested end-to-end pipeline with {len(chunks)} chunks")


if __name__ == "__main__":
    unittest.main()