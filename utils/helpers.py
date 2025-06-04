import pdfplumber
import re
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
from enum import Enum
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ChunkingStrategy(Enum):
    """Enum defining different text chunking strategies."""
    SLIDING_WINDOW = "sliding_window"
    PARAGRAPH = "paragraph"
    SECTION = "section"

class PDFProcessor:
    """
    A utility class to extract text from PDF files using pdfplumber.
    Provides methods for extracting text page-by-page or as a single concatenated string.
    Handles empty pages and non-text content gracefully.
    
    Extended with semantic chunking and vector indexing capabilities.
    """
    
    def __init__(self, pdf_path: str, clean_text: bool = True):
        """
        Initialize the PDFProcessor with a path to a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            clean_text (bool): Whether to clean extracted text
        """
        self.pdf_path = pdf_path
        self.clean_text = clean_text
        self.pdf = None
        self._initialize_pdf()
        
        # Fields for vector indexing
        self.chunks = []
        self.chunk_metadata = []
        self.vectorizer = None
        self.vector_index = None
        self.vector_matrix = None
    
    def _initialize_pdf(self):
        """Open the PDF file and handle any initialization errors."""
        try:
            self.pdf = pdfplumber.open(self.pdf_path)
            self.total_pages = len(self.pdf.pages)
        except Exception as e:
            logging.error(f"Failed to open PDF file {self.pdf_path}: {str(e)}")
            raise
    
    def __del__(self):
        """Close the PDF file when the object is deleted."""
        if self.pdf:
            self.pdf.close()
    
    def get_page_count(self) -> int:
        """
        Return the total number of pages in the PDF.
        
        Returns:
            int: Number of pages
        """
        return self.total_pages
    
    def _clean_text_content(self, text: str) -> str:
        """
        Clean the extracted text by removing extra whitespace and fixing common OCR issues.
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or not self.clean_text:
            return text
        
        # Remove multiple whitespaces, newlines
        cleaned = re.sub(r'\s+', ' ', text)
        # Fix common OCR issues
        cleaned = re.sub(r'([a-z])- ([a-z])', r'\1\2', cleaned)
        # Trim whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def extract_page_text(self, page_num: int) -> str:
        """
        Extract text from a specific page.
        
        Args:
            page_num (int): Page number (0-indexed)
            
        Returns:
            str: Extracted text from the page
            
        Raises:
            IndexError: If page_num is out of range
        """
        if page_num < 0 or page_num >= self.total_pages:
            raise IndexError(f"Page number {page_num} is out of range (0-{self.total_pages-1})")
        
        try:
            page = self.pdf.pages[page_num]
            text = page.extract_text() or ""
            return self._clean_text_content(text)
        except Exception as e:
            logging.warning(f"Error extracting text from page {page_num}: {str(e)}")
            return ""
    
    def extract_all_text(self, page_range: Optional[Tuple[int, int]] = None) -> str:
        """
        Extract text from all pages and return a single concatenated string.
        
        Args:
            page_range (tuple, optional): A tuple of (start_page, end_page) to limit extraction.
                                         Pages are 0-indexed.
        
        Returns:
            str: Concatenated text from all pages
        """
        all_text = []
        
        start_page = 0
        end_page = self.total_pages - 1
        
        if page_range:
            start_page = max(0, page_range[0])
            end_page = min(self.total_pages - 1, page_range[1])
        
        for page_num in range(start_page, end_page + 1):
            page_text = self.extract_page_text(page_num)
            if page_text:
                all_text.append(page_text)
        
        return " ".join(all_text)
    
    def extract_structured_text(self, page_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Union[int, str]]]:
        """
        Extract text with page number information.
        
        Args:
            page_range (tuple, optional): A tuple of (start_page, end_page) to limit extraction.
                                         Pages are 0-indexed.
        
        Returns:
            list: List of dictionaries with page numbers and text
                 [{'page': page_num, 'text': page_text}, ...]
        """
        structured_text = []
        
        start_page = 0
        end_page = self.total_pages - 1
        
        if page_range:
            start_page = max(0, page_range[0])
            end_page = min(self.total_pages - 1, page_range[1])
        
        for page_num in range(start_page, end_page + 1):
            page_text = self.extract_page_text(page_num)
            
            # Only include pages with actual text content
            if page_text:
                structured_text.append({
                    'page': page_num + 1,  # Convert to 1-indexed for user-friendly output
                    'text': page_text
                })
        
        return structured_text
    
    def chunk_text(self, 
                  text: Optional[str] = None, 
                  strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW,
                  chunk_size: int = 300,
                  chunk_overlap: int = 50,
                  preserve_context: bool = True) -> List[str]:
        """
        Divide text into semantic segments using the specified chunking strategy.
        
        Args:
            text (str, optional): Text to chunk. If None, uses all text from the PDF.
            strategy (ChunkingStrategy): Chunking strategy to use.
            chunk_size (int): Target chunk size in words or characters.
            chunk_overlap (int): Overlap between consecutive chunks.
            preserve_context (bool): Whether to preserve context between chunks.
            
        Returns:
            list: List of text chunks
        """
        # If no text is provided, use all text from the PDF
        if text is None:
            text = self.extract_all_text()
        
        if not text:
            return []
        
        chunks = []
        
        if strategy == ChunkingStrategy.SLIDING_WINDOW:
            # Sliding window approach based on words
            words = word_tokenize(text)
            
            # If text is shorter than chunk size, return the entire text as one chunk
            if len(words) <= chunk_size:
                return [text]
            
            # Create chunks with sliding window
            for i in range(0, len(words) - chunk_overlap, chunk_size - chunk_overlap):
                # Ensure we don't go beyond the text length
                end_idx = min(i + chunk_size, len(words))
                chunk = ' '.join(words[i:end_idx])
                chunks.append(chunk)
                
                # If we've reached the end, break
                if end_idx == len(words):
                    break
                    
        elif strategy == ChunkingStrategy.PARAGRAPH:
            # Split by paragraphs (defined by double line breaks)
            paragraphs = re.split(r'\n\s*\n', text)
            current_chunk = []
            current_size = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Count words in paragraph
                paragraph_size = len(word_tokenize(paragraph))
                
                # If adding this paragraph exceeds the chunk size and we already have content,
                # finalize the current chunk and start a new one
                if current_size + paragraph_size > chunk_size and current_size > 0:
                    chunks.append(' '.join(current_chunk))
                    
                    if preserve_context and current_chunk:
                        # Keep the last paragraph for context preservation if needed
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text] if overlap_text else []
                        current_size = len(word_tokenize(overlap_text)) if overlap_text else 0
                    else:
                        current_chunk = []
                        current_size = 0
                
                current_chunk.append(paragraph)
                current_size += paragraph_size
            
            # Add the final chunk if there's anything left
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
        elif strategy == ChunkingStrategy.SECTION:
            # Split by sections or headers (assumed to be lines ending with : or lines in ALL CAPS)
            section_pattern = r'([A-Z][A-Z\s]+:|[A-Z][a-z\s]+:|\d+\.\s+[A-Z][\w\s]+)'
            sections = re.split(section_pattern, text)
            
            # Combine section headers with their content
            processed_sections = []
            for i in range(0, len(sections) - 1, 2):
                if i + 1 < len(sections):
                    processed_sections.append(f"{sections[i].strip()} {sections[i+1].strip()}")
                else:
                    # Handle leftover section
                    processed_sections.append(sections[i].strip())
            
            # Now chunk the processed sections
            current_chunk = []
            current_size = 0
            
            for section in processed_sections:
                section = section.strip()
                if not section:
                    continue
                
                # Count words in section
                section_size = len(word_tokenize(section))
                
                if section_size > chunk_size:
                    # If a single section is too large, use sliding window for just this section
                    section_words = word_tokenize(section)
                    for j in range(0, len(section_words), chunk_size - chunk_overlap):
                        end_j = min(j + chunk_size, len(section_words))
                        sub_section = ' '.join(section_words[j:end_j])
                        chunks.append(sub_section)
                else:
                    # If adding this section exceeds the chunk size and we already have content,
                    # finalize the current chunk and start a new one
                    if current_size + section_size > chunk_size and current_size > 0:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(section)
                    current_size += section_size
            
            # Add the final chunk if there's anything left
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_semantic_chunks(self, 
                           strategy: ChunkingStrategy = ChunkingStrategy.SLIDING_WINDOW,
                           chunk_size: int = 300,
                           chunk_overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Get content-aware chunks with metadata about their source.
        
        Args:
            strategy (ChunkingStrategy): Chunking strategy to use
            chunk_size (int): Target chunk size in words
            chunk_overlap (int): Overlap between consecutive chunks
            
        Returns:
            list: List of dictionaries containing chunk text and metadata
                 [{'text': chunk, 'page': page_num, 'position': position}, ...]
        """
        # Get structured text with page information
        structured_text = self.extract_structured_text()
        
        results = []
        chunk_position = 0
        
        for page_data in structured_text:
            page_num = page_data['page']
            page_text = page_data['text']
            
            # Chunk this page's text
            page_chunks = self.chunk_text(
                text=page_text,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Add each chunk with metadata
            for chunk in page_chunks:
                results.append({
                    'text': chunk,
                    'page': page_num,
                    'position': chunk_position,
                    'strategy': strategy.value,
                    'source': Path(self.pdf_path).name
                })
                chunk_position += 1
        
        # Store the chunks and metadata for later use
        self.chunks = [item['text'] for item in results]
        self.chunk_metadata = results
        
        return results
    
    def create_vector_index(self, chunks: Optional[List[str]] = None) -> bool:
        """
        Create vector representations for text chunks using TF-IDF vectorization and FAISS indexing.
        
        Args:
            chunks (list, optional): List of text chunks to vectorize. If None, uses previously generated chunks.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use provided chunks or previously generated ones
            if chunks is not None:
                self.chunks = chunks
            elif not self.chunks:
                # If no chunks provided and none stored, generate them
                semantic_chunks = self.get_semantic_chunks()
                self.chunks = [item['text'] for item in semantic_chunks]
            
            if not self.chunks:
                logging.warning("No chunks available for vectorization")
                return False
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            self.vector_matrix = self.vectorizer.fit_transform(self.chunks).toarray().astype('float32')
            
            # Create FAISS index for fast similarity search
            self.vector_index = faiss.IndexFlatL2(self.vector_matrix.shape[1])
            self.vector_index.add(self.vector_matrix)
            
            return True
        
        except Exception as e:
            logging.error(f"Error creating vector index: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query text using vector similarity.
        
        Args:
            query (str): Query text to search for
            top_k (int): Number of top matches to return
            
        Returns:
            list: List of dictionaries with matched chunks and metadata
                 [{'text': chunk, 'page': page_num, 'position': position, 'score': similarity_score}, ...]
        """
        if not self.vectorizer or not self.vector_index or not self.chunks:
            logging.error("Vector index not created. Call create_vector_index() first.")
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
            
            # Search the index
            distances, indices = self.vector_index.search(query_vector, min(top_k, len(self.chunks)))
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks):  # Skip invalid indices
                    continue
                
                # Calculate similarity score (1 - normalized distance)
                # Convert distance to similarity score (higher is better)
                similarity = 1.0 / (1.0 + distances[0][i])
                
                # Get the chunk and its metadata
                chunk = self.chunks[idx]
                metadata = self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                
                result = {
                    'text': chunk,
                    'score': similarity,
                }
                
                # Add metadata if available
                if metadata:
                    result.update({
                        'page': metadata.get('page', 0),
                        'position': metadata.get('position', 0),
                        'source': metadata.get('source', ''),
                    })
                
                results.append(result)
            
            # Sort by similarity score (highest first)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching similar chunks: {str(e)}")
            return []