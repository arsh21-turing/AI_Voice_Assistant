import os
import sys
import unittest
import pytest
from pathlib import Path

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, str(PROJECT_ROOT))

# Now import can work properly
from utils.helpers import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    """Test cases for the PDFProcessor class."""
    
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
    
    def test_pdf_initialization(self):
        """Test if the PDF is properly initialized and page count is correct."""
        # Verify that the PDFProcessor was initialized
        self.assertIsNotNone(self.pdf_processor)
        
        # Check if the PDF has pages
        page_count = self.pdf_processor.get_page_count()
        self.assertGreater(page_count, 0, "PDF should have at least one page")
        
        # Print the total number of pages for information
        print(f"Total pages in the test PDF: {page_count}")
    
    def test_extract_page_text(self):
        """Test extraction of text from a single page."""
        # Extract text from the first page
        first_page_text = self.pdf_processor.extract_page_text(0)
        
        # Verify that the extracted text is a string
        self.assertIsInstance(first_page_text, str, "Extracted page text should be a string")
        
        # Check if the first page has content (should have title or header information)
        self.assertTrue(first_page_text.strip(), "First page should have text content")
        
        # Print a sample of the extracted text for verification
        print(f"Sample text from first page: {first_page_text[:100]}...")
    
    def test_extract_all_text(self):
        """Test extraction of text from the entire PDF."""
        # Extract all text
        all_text = self.pdf_processor.extract_all_text()
        
        # Verify that the extracted text is a string
        self.assertIsInstance(all_text, str, "Extracted all text should be a string")
        
        # Check if the PDF has content
        self.assertTrue(all_text.strip(), "PDF should have text content")
        
        # Verify that the extracted text is substantial
        self.assertGreater(len(all_text), 100, "PDF should have substantial text content")
        
        # Test extraction with page range
        # Extract text from the first 3 pages or fewer if PDF is smaller
        page_count = min(3, self.pdf_processor.get_page_count())
        page_range_text = self.pdf_processor.extract_all_text(page_range=(0, page_count - 1))
        self.assertIsInstance(page_range_text, str, "Page range text should be a string")
    
    def test_extract_structured_text(self):
        """Test extraction of structured text with page numbers."""
        # Extract structured text
        structured_text = self.pdf_processor.extract_structured_text()
        
        # Verify the structured text is a list
        self.assertIsInstance(structured_text, list, "Structured text should be a list")
        
        # Verify the structure of each item in the list if not empty
        if structured_text:  # Some PDFs might have no extractable text on any page
            for page_data in structured_text:
                self.assertIsInstance(page_data, dict, "Each page entry should be a dictionary")
                self.assertIn('page', page_data, "Each page entry should have a 'page' key")
                self.assertIn('text', page_data, "Each page entry should have a 'text' key")
                self.assertIsInstance(page_data['page'], int, "Page number should be an integer")
                self.assertIsInstance(page_data['text'], str, "Page text should be a string")
                self.assertTrue(page_data['text'].strip(), "Page text should not be empty")
    
    def test_invalid_page(self):
        """Test handling of invalid page numbers."""
        # Get the total page count
        page_count = self.pdf_processor.get_page_count()
        
        # Test with a negative page number
        with self.assertRaises(IndexError, msg="Should raise IndexError for negative page number"):
            self.pdf_processor.extract_page_text(-1)
        
        # Test with a page number that is too large
        with self.assertRaises(IndexError, msg="Should raise IndexError for out-of-range page number"):
            self.pdf_processor.extract_page_text(page_count + 10)

if __name__ == "__main__":
    # Run all tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)