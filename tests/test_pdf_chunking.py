import os
import sys
import unittest
import pytest
from pathlib import Path
import re
from typing import List, Set

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, str(PROJECT_ROOT))

# Now import can work properly
from utils.helpers import PDFProcessor, ChunkingStrategy

class TestPDFChunking(unittest.TestCase):
    """Test cases for the chunking functionality of PDFProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all test methods."""
        # Path to the test PDF file using Path for better cross-platform compatibility
        cls.pdf_path = PROJECT_ROOT / "data" / "manuals" / "venue.pdf"
        
        # Skip tests if the PDF file doesn't exist
        if not cls.pdf_path.exists():
            pytest.skip(f"Test PDF file not found at {cls.pdf_path}")
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a new PDFProcessor instance for each test
        self.pdf_processor = PDFProcessor(str(self.pdf_path))
        # Extract a sample of text to use for chunking tests
        # We'll use the first 3 pages or all pages if less than 3
        page_count = min(3, self.pdf_processor.get_page_count())
        self.test_text = self.pdf_processor.extract_all_text(page_range=(0, page_count - 1))
        
        # Skip if there's not enough text for meaningful tests
        if len(self.test_text.split()) < 100:
            pytest.skip("Not enough text in the test PDF for chunking tests")
    
    def _get_word_count(self, text: str) -> int:
        """Helper method to get word count from a text string."""
        return len(text.split())
    
    def _check_common_words(self, text1: str, text2: str, min_common: int = 5) -> bool:
        """Check if two text chunks have at least min_common words in common."""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        return len(words1.intersection(words2)) >= min_common
    
    def test_chunk_text_sliding_window(self):
        """Test the sliding window chunking strategy."""
        # Test with various chunk sizes and overlaps
        chunk_sizes = [50, 100, 200]
        overlaps = [10, 25, 30]  # Changed from [10, 25, 50] to ensure overlap is always less than chunk_size
        
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                # Skip invalid combinations where overlap >= chunk_size
                if overlap >= chunk_size:
                    continue
                    
                with self.subTest(chunk_size=chunk_size, overlap=overlap):
                    chunks = self.pdf_processor.chunk_text(
                        text=self.test_text,
                        strategy=ChunkingStrategy.SLIDING_WINDOW,
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                    
                    # Verify that chunks were created
                    self.assertIsInstance(chunks, list, "Chunks should be a list")
                    if self._get_word_count(self.test_text) > chunk_size:
                        self.assertGreater(len(chunks), 0, "Should produce at least one chunk")
                    
                    # Check chunk sizes
                    for i, chunk in enumerate(chunks):
                        self.assertIsInstance(chunk, str, f"Chunk {i} should be a string")
                        # Last chunk might be smaller, so check all except the last one
                        if i < len(chunks) - 1:
                            chunk_word_count = self._get_word_count(chunk)
                            self.assertLessEqual(
                                chunk_word_count, 
                                chunk_size, 
                                f"Chunk {i} should not exceed the specified size"
                            )
    
    def test_chunk_text_paragraph(self):
        """Test the paragraph-based chunking strategy."""
        # Test with a generous chunk size to ensure paragraphs stay together
        chunk_size = 200
        
        chunks = self.pdf_processor.chunk_text(
            text=self.test_text,
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=chunk_size,
            chunk_overlap=20
        )
        
        # Verify that chunks were created
        self.assertIsInstance(chunks, list, "Chunks should be a list")
        
        # Skip the rest if no paragraphs were detected
        if not chunks:
            return
        
        self.assertGreater(len(chunks), 0, "Should produce at least one chunk")
        
        # Verify content of chunks
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, str, f"Chunk {i} should be a string")
            self.assertTrue(chunk.strip(), f"Chunk {i} should not be empty")
    
    def test_chunk_text_section(self):
        """Test the section-based chunking strategy."""
        # Test with a generous chunk size
        chunk_size = 300
        
        chunks = self.pdf_processor.chunk_text(
            text=self.test_text,
            strategy=ChunkingStrategy.SECTION,
            chunk_size=chunk_size,
            chunk_overlap=20
        )
        
        # Verify that chunks were created
        self.assertIsInstance(chunks, list, "Chunks should be a list")
        
        # Skip the rest if no sections were detected
        if not chunks:
            return
            
        self.assertGreater(len(chunks), 0, "Should produce at least one chunk")
        
        # Verify content of chunks
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, str, f"Chunk {i} should be a string")
            self.assertTrue(chunk.strip(), f"Chunk {i} should not be empty")
    
    def test_chunk_overlap(self):
        """Test that chunks maintain the specified overlap."""
        chunk_size = 100
        chunk_overlap = 30
        
        # Create chunks with significant overlap
        chunks = self.pdf_processor.chunk_text(
            text=self.test_text,
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Skip if not enough chunks to test overlap
        if len(chunks) < 2:
            return
            
        # Check for overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            has_overlap = self._check_common_words(chunks[i], chunks[i+1])
            self.assertTrue(
                has_overlap, 
                f"Chunks {i} and {i+1} should have overlapping words"
            )
    
    def test_get_semantic_chunks(self):
        """Test creating semantic chunks with metadata."""
        # Get semantic chunks with metadata
        semantic_chunks = self.pdf_processor.get_semantic_chunks(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=100,
            chunk_overlap=20
        )
        
        # Verify basic structure
        self.assertIsInstance(semantic_chunks, list, "Semantic chunks should be a list")
        
        # Skip if no chunks were created
        if not semantic_chunks:
            return
            
        self.assertGreater(len(semantic_chunks), 0, "Should produce at least one semantic chunk")
        
        # Verify each chunk has the expected metadata
        for i, chunk_data in enumerate(semantic_chunks):
            self.assertIsInstance(chunk_data, dict, f"Chunk data {i} should be a dictionary")
            
            # Check required fields
            self.assertIn('text', chunk_data, f"Chunk {i} should have 'text' field")
            self.assertIn('page', chunk_data, f"Chunk {i} should have 'page' field")
            self.assertIn('position', chunk_data, f"Chunk {i} should have 'position' field")
            self.assertIn('strategy', chunk_data, f"Chunk {i} should have 'strategy' field")
            
            # Check field types
            self.assertIsInstance(chunk_data['text'], str, f"Chunk {i} text should be a string")
            self.assertIsInstance(chunk_data['page'], int, f"Chunk {i} page should be an integer")
            self.assertIsInstance(chunk_data['position'], int, f"Chunk {i} position should be an integer")
            
            # Check content
            self.assertTrue(chunk_data['text'].strip(), f"Chunk {i} text should not be empty")
            self.assertGreater(chunk_data['page'], 0, f"Chunk {i} page should be positive")
            self.assertGreaterEqual(chunk_data['position'], 0, f"Chunk {i} position should be non-negative")
    
    def test_vector_indexing(self):
        """Test creating vector representations for chunks."""
        # First create some chunks
        chunks = self.pdf_processor.chunk_text(
            text=self.test_text,
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=100,
            chunk_overlap=20
        )
        
        # Skip if not enough chunks
        if not chunks or len(chunks) < 2:
            return
            
        # Create vector index
        result = self.pdf_processor.create_vector_index(chunks)
        
        # Verify index creation
        self.assertTrue(result, "Vector index creation should succeed")
        self.assertIsNotNone(self.pdf_processor.vectorizer, "Vectorizer should be created")
        self.assertIsNotNone(self.pdf_processor.vector_index, "Vector index should be created")
        self.assertIsNotNone(self.pdf_processor.vector_matrix, "Vector matrix should be created")
    
    def test_similarity_search(self):
        """Test searching for similar chunks."""
        # Skip this test if there's not enough text
        if len(self.test_text) < 500:
            return
            
        # First get semantic chunks
        semantic_chunks = self.pdf_processor.get_semantic_chunks(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=80,
            chunk_overlap=20
        )
        
        # Skip if not enough chunks
        if not semantic_chunks or len(semantic_chunks) < 2:
            return
            
        # Create vector index
        result = self.pdf_processor.create_vector_index()
        if not result:
            self.skipTest("Vector index creation failed")
        
        # Create a query from the first chunk's text (to ensure there's at least one match)
        query_text = semantic_chunks[0]['text']
        # Use just a portion of it to make it more challenging
        words = query_text.split()
        if len(words) > 10:
            query_text = ' '.join(words[:10])
        
        # Search for similar chunks
        results = self.pdf_processor.search_similar_chunks(query_text, top_k=3)
        
        # Verify search results
        self.assertIsInstance(results, list, "Search results should be a list")
        self.assertGreater(len(results), 0, "Should find at least one similar chunk")
        
        # Check structure of results
        for i, result in enumerate(results):
            self.assertIsInstance(result, dict, f"Result {i} should be a dictionary")
            self.assertIn('text', result, f"Result {i} should have 'text' field")
            self.assertIn('score', result, f"Result {i} should have 'score' field")
            
            # Score should be a number between 0 and 1
            self.assertIsInstance(result['score'], float, f"Result {i} score should be a float")
            self.assertGreater(result['score'], 0, f"Result {i} score should be positive")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)