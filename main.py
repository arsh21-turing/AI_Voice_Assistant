import speech_recognition as sr
import pyttsx3
import time
import json
import os
import numpy as np
import requests
from typing import Optional, Dict, Any, List, Tuple
import glob
import hashlib
from pathlib import Path
import torch

# Import for direct RAG implementation
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber

from utils.error_handler import ErrorHandler
from speech.formatting import VoiceResponseFormatter


def extract_text_from_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Extract and chunk text from a PDF file using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of dictionaries containing chunk text and page info
    """
    try:
        print(f"Extracting text from PDF: {pdf_path}")
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # First, extract full text with page numbers
            full_text = ""
            page_markers = {}  # Store the character position of each page start
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_number = page.page_number
                    page_marker = f"[Page {page_number}] "
                    page_markers[len(full_text)] = page_number
                    full_text += page_marker + page_text + "\n\n"
            
            # Now, create overlapping chunks with page number tracking
            if not full_text.strip():
                print(f"Warning: No text extracted from {pdf_path}")
                return []
                
            # Chunk the text
            start = 0
            text_length = len(full_text)
            
            while start < text_length:
                # Calculate end position of current chunk
                end = min(start + chunk_size, text_length)
                
                # Adjust chunk boundaries to avoid breaking sentences if possible
                if end < text_length:
                    # Try to find a period, question mark, or exclamation point followed by space/newline
                    for punct in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
                        punct_pos = full_text.rfind(punct, start, end)
                        if punct_pos != -1:
                            end = punct_pos + len(punct)
                            break
                
                # Extract the chunk
                chunk_text = full_text[start:end]
                
                # Determine which pages this chunk covers
                chunk_pages = []
                for pos, page_num in sorted(page_markers.items()):
                    if pos <= start and (len(chunk_pages) == 0 or chunk_pages[-1] != page_num):
                        chunk_pages.append(page_num)
                    elif pos <= end and page_num not in chunk_pages:
                        chunk_pages.append(page_num)
                    elif pos > end:
                        break
                
                # Create chunk info
                chunk_info = {
                    "text": chunk_text,
                    "pages": chunk_pages,
                    "start_char": start,
                    "end_char": end
                }
                chunks.append(chunk_info)
                
                # Move to next chunk position with overlap
                start = end - chunk_overlap if end < text_length else text_length
        
        print(f"Created {len(chunks)} chunks from {pdf_path}")
        return chunks
    
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return []


def create_index(texts: List[str], embedding_model_name: str = "all-MiniLM-L6-v2") -> Tuple[Any, np.ndarray]:
    """Create a FAISS index from a list of texts with optimized embedding generation.
    
    Args:
        texts: List of text chunks to index
        embedding_model_name: Name of the SentenceTransformer model to use
    
    Returns:
        Tuple of (FAISS index, numpy array of embeddings)
    """
    try:
        if not texts:
            print("No texts provided for indexing")
            return None, None
            
        print(f"Creating index with {len(texts)} text chunks using {embedding_model_name}")
        
        # Load the embedding model with optimized settings
        model = SentenceTransformer(
            embedding_model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_folder='./model_cache'  # Cache the model locally
        )
        
        # Set batch size based on available memory
        batch_size = 32 if torch.cuda.is_available() else 16
        
        # Generate embeddings in batches with progress bar
        print("Generating embeddings...")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True  # Normalize embeddings for better similarity search
        )
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index with optimized settings
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use Inner Product for normalized vectors
        
        # Add vectors to the index
        index.add(embeddings_np)
        
        print(f"Successfully created index with {index.ntotal} vectors of dimension {dimension}")
        return index, embeddings_np
        
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        return None, None


class VoiceAssistant:
    """Consolidated class for speech recognition, synthesis, and RAG using SentenceTransformer and FAISS."""
    
    def __init__(self, config_path="config.json", error_handler=None, wake_word="car assistant"):
        """Initialize the voice assistant with speech and RAG capabilities.
        
        Args:
            config_path: Path to the JSON configuration file
            error_handler: Error handler for centralized error handling
            wake_word: Wake word to activate the assistant
        """
        self.error_handler = error_handler
        self.wake_word = wake_word.lower()
        
        # Load configuration directly from JSON
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    self.config = json.load(f)
            else:
                print(f"Config file not found at {config_path}. Using default configuration.")
                self.config = self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {str(e)}. Using default configuration.")
            self.config = self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {str(e)}. Using default configuration.")
            self.config = self._get_default_config()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize speech synthesis
        self.engine = pyttsx3.init()
        
        # Configure speech properties from loaded config
        voice_config = self.config.get("voice", {})
        self.engine.setProperty('rate', voice_config.get('rate', 150))
        self.engine.setProperty('volume', voice_config.get('volume', 1.0))
        
        # Set voice based on configuration
        voices = self.engine.getProperty('voices')
        voice_id = voice_config.get('voice_id')
        if voice_id and voices:
            for voice in voices:
                if voice.id == voice_id:
                    self.engine.setProperty('voice', voice.id)
                    break
        
        # Response formatter for natural-sounding speech
        self.formatter = VoiceResponseFormatter(self.config.get("formatting", {}))
        
        # Get Groq API configuration
        groq_config = self.config.get("API_SETTINGS", {})
        self.groq_api_key = groq_config.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
        self.groq_model = groq_config.get("GROQ_MODEL", "llama3-8b-8192")
        self.groq_base_url = groq_config.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        
        if not self.groq_api_key:
            print("WARNING: No Groq API key found. Response generation will be limited.")
        
        # Initialize RAG components directly in __init__
        self._initialize_rag_components()
        
        # Adjust for ambient noise when initialized
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "voice": {
                "rate": 150,
                "volume": 1.0,
                "voice_id": None
            },
            "recognition": {
                "timeout": 10,
                "phrase_time_limit": 5
            },
            "formatting": {
                "use_ssml": False,
                "pause_words": ["however", "additionally", "furthermore", "nevertheless"],
                "emphasis_keywords": ["warning", "caution", "important", "note"]
            },
            "rag": {
                "model_name": "all-MiniLM-L6-v2",
                "index_path": "data/index/manual_index",
                "top_k": 3,
                "relevance_threshold": 0.6,
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "API_SETTINGS": {
                "GROQ_API_KEY": "",
                "GROQ_MODEL": "llama3-8b-8192",
                "GROQ_API_BASE": "https://api.groq.com/openai/v1"
            }
        }
    
    def _initialize_rag_components(self):
        """Initialize RAG components (SentenceTransformer and FAISS)."""
        # Get RAG configuration
        rag_config = self.config.get("rag", {})
        
        # Initialize with empty values for graceful degradation in case of errors
        self.embedding_model = None
        self.index = None
        self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
        
        try:
            # Initialize embedding model
            model_name = rag_config.get("model_name", "all-MiniLM-L6-v2")
            print(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            
            # FAISS index settings
            self.index_path = rag_config.get("index_path", "data/index/manual_index")
            self.metadata_path = os.path.join(self.index_path, "metadata.json")
            
            # Set default retrieval parameters
            self.top_k = rag_config.get("top_k", 3)
            self.relevance_threshold = rag_config.get("relevance_threshold", 0.6)
            self.chunk_size = rag_config.get("chunk_size", 1000)
            self.chunk_overlap = rag_config.get("chunk_overlap", 200)
            
            # Load or create FAISS index
            self._load_or_create_index()
            
            print("RAG components initialized successfully.")
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "rag_init", {"context": "initialization"})
            print(f"Error initializing RAG components: {str(e)}")
    
    def _load_or_create_index(self):
        """Load FAISS index and metadata if they exist, or create new ones if they don't."""
        try:
            # Ensure directory exists
            os.makedirs(self.index_path, exist_ok=True)
            
            index_file_path = os.path.join(self.index_path, "index.faiss")
            
            # Check if index file exists
            if os.path.exists(index_file_path):
                print(f"Loading existing FAISS index from {index_file_path}")
                try:
                    self.index = faiss.read_index(index_file_path)
                    print(f"Loaded FAISS index with {self.index.ntotal} vectors")
                    
                    # Load metadata if it exists
                    if os.path.exists(self.metadata_path):
                        try:
                            with open(self.metadata_path, 'r') as f:
                                self.metadata = json.load(f)
                            print(f"Loaded metadata with {len(self.metadata.get('texts', []))} entries")
                            
                            # Maintain backward compatibility - add document_hashes if not present
                            if "document_hashes" not in self.metadata:
                                self.metadata["document_hashes"] = {}
                                print("Added document_hashes field to metadata for tracking indexed documents")
                            
                            # Validate metadata format
                            if not all(key in self.metadata for key in ["texts", "ids", "metadata"]):
                                print("Warning: Metadata file has invalid format. Initializing with empty metadata.")
                                self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing metadata file: {str(e)}. Creating new metadata.")
                            self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
                        except Exception as e:
                            print(f"Error loading metadata: {str(e)}. Creating new metadata.")
                            self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
                    else:
                        print("Metadata file not found. Creating new metadata.")
                        self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
                        
                except Exception as e:
                    print(f"Error loading FAISS index: {str(e)}. Creating new index.")
                    self._create_new_index()
            else:
                print(f"FAISS index not found at {index_file_path}. Creating new index.")
                self._create_new_index()
                
        except Exception as e:
            print(f"Error in load_or_create_index: {str(e)}")
            # Create empty index with fallback dimension
            self._create_new_index(embedding_dim=384)  # Common dimension for sentence transformers
    
    def _create_new_index(self, embedding_dim=None):
        """Create a new FAISS index with the specified embedding dimension."""
        try:
            # Get embedding dimension from the model if not specified
            if not embedding_dim and self.embedding_model:
                embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            elif not embedding_dim:
                # Fallback dimension for common sentence transformer models
                embedding_dim = 384
                
            # Create empty index
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
            
            print(f"Created new empty FAISS index with dimension {embedding_dim}")
            
            # Save empty index and metadata
            self._save_index()
            
        except Exception as e:
            print(f"Error creating new index: {str(e)}")
            if self.error_handler:
                self.error_handler.log_error(e, "rag_init", {"context": "create_index"})
    
    def process_documents(self, docs_dir="data/manuals", force_reload=False) -> bool:
        """Process all PDF documents in a directory and add them to the index with optimized embedding generation.
        
        Args:
            docs_dir: Directory containing PDF files to process
            force_reload: Whether to force reprocessing of already indexed documents
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            if not os.path.exists(docs_dir):
                print(f"Documents directory not found: {docs_dir}")
                return False
            
            # Get list of PDF files
            pdf_files = glob.glob(os.path.join(docs_dir, "**", "*.pdf"), recursive=True)
            if not pdf_files:
                print(f"No PDF files found in {docs_dir}")
                return False
                
            print(f"Found {len(pdf_files)} PDF files to process")
            processed_count = 0
            
            # Process documents in batches for better memory management
            batch_size = 5  # Process 5 documents at a time
            for i in range(0, len(pdf_files), batch_size):
                batch_files = pdf_files[i:i + batch_size]
                batch_texts = []
                batch_metadatas = []
                
                for pdf_path in batch_files:
                    # Check if already processed
                    file_hash = self._get_file_hash(pdf_path)
                    rel_path = os.path.relpath(pdf_path, start=os.path.dirname(docs_dir))
                    
                    if not force_reload and file_hash in self.metadata.get("document_hashes", {}):
                        print(f"Skipping already indexed document: {rel_path}")
                        continue
                        
                    print(f"Processing document: {rel_path}")
                    
                    # Extract document info
                    file_name = os.path.basename(pdf_path)
                    vehicle_name = self._extract_vehicle_name(file_name, pdf_path)
                    document_type = self._extract_document_type(file_name)
                    
                    # Extract and chunk text
                    chunks = extract_text_from_pdf(
                        pdf_path, 
                        chunk_size=self.chunk_size, 
                        chunk_overlap=self.chunk_overlap
                    )
                    
                    if not chunks:
                        print(f"No text extracted from {rel_path}. Skipping.")
                        continue
                    
                    # Add to batch
                    for chunk in chunks:
                        batch_texts.append(chunk["text"])
                        batch_metadatas.append({
                            "source": rel_path,
                            "document_type": document_type,
                            "vehicle": vehicle_name,
                            "pages": chunk.get("pages", []),
                            "chunk_index": len(batch_texts) - 1,
                            "total_chunks": len(chunks)
                        })
                
                if batch_texts:
                    # Create embeddings for the batch
                    print(f"Generating embeddings for batch of {len(batch_texts)} chunks...")
                    index, embeddings = create_index(batch_texts)
                    
                    if index is not None:
                        # Add to existing index
                        if self.index is None:
                            self.index = index
                        else:
                            self.index.add(embeddings)
                        
                        # Update metadata
                        self.metadata["texts"].extend(batch_texts)
                        self.metadata["metadata"].extend(batch_metadatas)
                        
                        # Update document hashes
                        for pdf_path in batch_files:
                            file_hash = self._get_file_hash(pdf_path)
                            rel_path = os.path.relpath(pdf_path, start=os.path.dirname(docs_dir))
                            self.metadata["document_hashes"][file_hash] = {
                                "path": rel_path,
                                "last_modified": os.path.getmtime(pdf_path),
                                "chunk_count": len(batch_texts),
                                "processed_date": time.time()
                            }
                        
                        # Save after each batch
                        self._save_index()
                        processed_count += len(batch_files)
                        print(f"Successfully processed batch of {len(batch_files)} documents")
                    else:
                        print("Failed to create embeddings for batch")
            
            print(f"Processed {processed_count} new documents out of {len(pdf_files)} total")
            return True
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "document_processing", {"context": "batch_processing"})
            print(f"Error processing documents: {str(e)}")
            return False
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file to track whether it's been indexed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            String hash of the file content
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {file_path}: {str(e)}")
            # Fallback to file name and size as pseudo-hash
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            return f"{os.path.basename(file_path)}_{file_size}"
    
    def _extract_vehicle_name(self, filename: str, filepath: str) -> str:
        """Extract vehicle name from filename or content.
        
        Args:
            filename: Name of the PDF file
            filepath: Path to the PDF file
            
        Returns:
            Vehicle name or generic value if not determinable
        """
        # Try to extract from filename first
        filename = filename.lower()
        
        # Common vehicle names to check for
        vehicle_models = {
            "venue": "Hyundai Venue",
            "elantra": "Hyundai Elantra",
            "sonata": "Hyundai Sonata",
            "santa": "Hyundai Santa Fe",
            "tucson": "Hyundai Tucson",
            "kona": "Hyundai Kona",
            "accent": "Hyundai Accent",
            "palisade": "Hyundai Palisade",
            "ioniq": "Hyundai Ioniq",
            "civic": "Honda Civic",
            "accord": "Honda Accord",
            "cr-v": "Honda CR-V",
            "camry": "Toyota Camry",
            "corolla": "Toyota Corolla",
            "rav4": "Toyota RAV4",
            "cruze": "Chevrolet Cruze",
            "malibu": "Chevrolet Malibu",
            "impala": "Chevrolet Impala",
            "focus": "Ford Focus",
            "fusion": "Ford Fusion",
            "escape": "Ford Escape",
            "altima": "Nissan Altima",
            "sentra": "Nissan Sentra",
            "rogue": "Nissan Rogue"
        }
        
        # Check for vehicle model names in the filename
        for model, full_name in vehicle_models.items():
            if model in filename:
                return full_name
        
        # Extract from parent directory name as fallback
        parent_dir = os.path.basename(os.path.dirname(filepath))
        for model, full_name in vehicle_models.items():
            if model in parent_dir.lower():
                return full_name
        
        # Generic fallback
        return "Generic Vehicle"
    
    def _extract_document_type(self, filename: str) -> str:
        """Extract document type from filename.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Document type (manual, guide, etc.)
        """
        filename = filename.lower()
        
        # Check for common document types in the filename
        if "manual" in filename:
            return "manual"
        elif "guide" in filename:
            return "guide"
        elif "handbook" in filename:
            return "handbook"
        elif "spec" in filename:
            return "specifications"
        elif "maintenance" in filename:
            return "maintenance guide"
        elif "warranty" in filename:
            return "warranty information"
        elif "quick" in filename and ("start" in filename or "guide" in filename):
            return "quick start guide"
        else:
            return "documentation"
    
    def load_venue_pdf(self, force_reload=False, chunk_size=1000, chunk_overlap=200) -> bool:
        """Load and process the Hyundai Venue PDF manual if not already in the index.
        
        Args:
            force_reload: Whether to force reloading even if chunks exist
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_path = "data/manuals/venue.pdf"
            
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                print(f"Error: PDF file not found at {pdf_path}")
                return False
            
            # Check if Venue PDF chunks already exist in the index
            if not force_reload and self._check_venue_chunks_exist():
                print("Venue manual chunks already exist in the index. Skipping processing.")
                return True
            
            print(f"Processing Venue PDF manual from {pdf_path}...")
            
            # Extract text from PDF using pdfplumber
            pdf_text = self._extract_text_from_pdf(pdf_path)
            if not pdf_text:
                print("Failed to extract text from PDF.")
                return False
            
            # Create chunks with appropriate size and overlap
            chunks = self._create_text_chunks(pdf_text, chunk_size, chunk_overlap)
            print(f"Created {len(chunks)} text chunks from Venue PDF manual.")
            
            # Prepare metadata for each chunk
            metadatas = []
            for i, _ in enumerate(chunks):
                metadatas.append({
                    "source": "venue.pdf",
                    "document_type": "manual",
                    "vehicle": "Hyundai Venue",
                    "chunk_index": i
                })
            
            # Add chunks to the FAISS index
            success = self.add_documents(chunks, metadatas)
            if success:
                print(f"Successfully added {len(chunks)} Venue manual chunks to the index.")
            else:
                print("Failed to add Venue manual chunks to the index.")
            
            return success
        
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "pdf_loading", {"context": "venue_manual"})
            print(f"Error loading Venue PDF: {str(e)}")
            return False
        
    def _check_venue_chunks_exist(self) -> bool:
        """Check if Venue PDF chunks already exist in the index.
        
        Returns:
            True if chunks exist, False otherwise
        """
        if not self.metadata or "metadata" not in self.metadata:
            return False
        
        # Check for any chunks that have source=venue.pdf
        for meta in self.metadata["metadata"]:
            if isinstance(meta, dict) and meta.get("source") == "venue.pdf":
                return True
        
        return False

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Extracted text as a single string
        """
        try:
            text_content = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                # Process each page
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        # Add page number reference and append to content
                        page_number = page.page_number
                        text_content += f"[Page {page_number}]\n{page_text}\n\n"
            
            return text_content
        
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _create_text_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks of specified size.
        
        Args:
            text: The text to split into chunks
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position of current chunk
            end = min(start + chunk_size, text_length)
            
            # Adjust chunk boundaries to avoid breaking sentences if possible
            if end < text_length:
                # Try to find a period, question mark, or exclamation point followed by space/newline
                for punct in [". ", "? ", "! ", ".\n", "?\n", "!\n"]:
                    punct_pos = text.rfind(punct, start, end)
                    if punct_pos != -1:
                        end = punct_pos + len(punct)
                        break
            
            # Extract the chunk and add to results
            chunks.append(text[start:end])
            
            # Move to next chunk position with overlap
            start = end - chunk_overlap if end < text_length else text_length
        
        return chunks
    
    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context using SentenceTransformer and FAISS with boosting for Venue content.
        
        Args:
            query: The user's query
            top_k: Number of top results to retrieve (uses config value if not specified)
            
        Returns:
            List of contexts with text, metadata, and relevance scores
        """
        if not self.embedding_model or not self.index:
            print("RAG components not properly initialized. Cannot retrieve context.")
            return []
        
        try:
            # Get retrieval parameters with potential adjustments for Venue-related queries
            if not top_k:
                top_k = self.top_k
                
            # Determine if this is a Venue-related query
            is_venue_query = self._is_venue_related_query(query)
            
            # Adjust parameters for Venue queries
            venue_boost = 0.15  # Boost amount for Venue content
            original_threshold = self.relevance_threshold
            
            # Lower threshold for Venue queries to include more relevant results
            if is_venue_query:
                relevance_threshold = max(0.4, original_threshold - 0.1)  # Lower threshold but not below 0.4
                print(f"Venue-related query detected. Using adjusted threshold: {relevance_threshold}")
                # Retrieve more results initially for re-ranking
                search_top_k = min(top_k * 2, 20)  # Get more results but cap at 20
            else:
                relevance_threshold = original_threshold
                search_top_k = top_k
                
            print(f"Performing semantic search for query: '{query}'")
            
            # Preprocess query - convert to lowercase and remove excessive whitespace
            query = " ".join(query.lower().split())
            
            # Embed the query - convert to dense vector representation
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
            
            # Check if index is empty
            if self.index.ntotal == 0:
                print("FAISS index is empty. No context to retrieve.")
                return []
            
            # Perform semantic search with FAISS
            # This returns L2 distances (lower is better) and indices
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), search_top_k)
            
            # Process search results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.metadata["texts"]):
                    continue  
                
                # Get text and metadata
                text = self.metadata["texts"][idx]
                chunk_metadata = self.metadata["metadata"][idx] if idx < len(self.metadata["metadata"]) else {}
                
                # Convert L2 distance to similarity score (smaller distance = higher similarity)
                similarity_score = 1 / (1 + dist)
                
                # Apply boosting for Venue content
                is_venue_content = self._is_venue_content(chunk_metadata)
                
                # Boost score for Venue content in Venue queries, slightly boost venue content in all queries
                if is_venue_query and is_venue_content:
                    boosted_score = min(1.0, similarity_score + venue_boost)
                    print(f"Boosting Venue content score from {similarity_score:.4f} to {boosted_score:.4f}")
                    similarity_score = boosted_score
                elif is_venue_content:
                    # Small boost for Venue content even in non-Venue queries
                    similarity_score = min(1.0, similarity_score + (venue_boost * 0.3))
                
                # Skip results below adjusted relevance threshold
                if similarity_score < relevance_threshold:
                    continue
                
                results.append({
                    "text": text,
                    "metadata": chunk_metadata,
                    "score": float(similarity_score),
                    "index": int(idx),
                    "is_venue": is_venue_content
                })
            
            # Sort by score in descending order
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            # Prioritize Venue results for Venue queries while maintaining diversity
            if is_venue_query and results:
                results = self._prioritize_venue_results(results, top_k)
            else:
                # Limit to top_k
                results = results[:top_k]
            
            print(f"Retrieved {len(results)} context chunks from vector database")
            if is_venue_query:
                venue_count = sum(1 for r in results if r.get("is_venue", False))
                print(f"Including {venue_count} Venue-specific chunks")
                
            return results
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "rag_retrieval", {"context": "retrieval", "query": query})
            print(f"Error retrieving context: {str(e)}")
            return []
    
    def _is_venue_related_query(self, query: str) -> bool:
        """Determine if a query is related to the Hyundai Venue.
        
        Args:
            query: The query text
            
        Returns:
            True if query is likely Venue-related, False otherwise
        """
        # Keywords that strongly indicate a Venue-related query
        venue_keywords = [
            "venue", "hyundai venue", "venue car", "venue suv", "venue manual",
            "venue features", "venue specifications", "venue spec", "venue problem",
            "venue engine", "venue interior", "venue exterior", "hyundai suv", 
            "my venue", "venue dashboard", "venue settings"
        ]
        
        # Check if any keywords are in the query
        query_lower = query.lower()
        for keyword in venue_keywords:
            if keyword in query_lower:
                return True
                
        # More complex pattern matching could be added here
        
        return False
    
    def _is_venue_content(self, metadata: Dict[str, Any]) -> bool:
        """Determine if content is from the Venue manual based on metadata.
        
        Args:
            metadata: Metadata dictionary for a chunk
            
        Returns:
            True if content is from Venue manual, False otherwise
        """
        # Check for explicit Venue markers in metadata
        if metadata.get("source") == "venue.pdf":
            return True
        if metadata.get("vehicle") == "Hyundai Venue":
            return True
        if isinstance(metadata.get("source"), str) and "venue" in metadata.get("source").lower():
            return True
        
        # Could add more complex checks here
        
        return False
    
    def _prioritize_venue_results(self, results: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Prioritize Venue results while maintaining diversity.
        
        Args:
            results: List of retrieved results
            max_results: Maximum number of results to return
            
        Returns:
            Re-ranked list of results with Venue content prioritized
        """
        if not results:
            return []
        
        # Separate Venue results from other results
        venue_results = [r for r in results if r.get("is_venue", False)]
        other_results = [r for r in results if not r.get("is_venue", False)]
        
        # Create final list with prioritized venue content
        final_results = []
        
        # If we have very few results, just return all of them (up to max_results)
        if len(results) <= max_results:
            return results[:max_results]
        
        # If we have some venue results, interleave them with other results
        # but ensure venue results get priority
        if venue_results:
            # Calculate how many venue vs other results to include
            venue_count = min(len(venue_results), int(max_results * 0.7))
            other_count = min(len(other_results), max_results - venue_count)
            
            # Take top venue results
            final_results.extend(venue_results[:venue_count])
            
            # Add some other results for diversity
            final_results.extend(other_results[:other_count])
            
            # Sort once more by score to get proper ordering
            final_results = sorted(final_results, key=lambda x: x["score"], reverse=True)
            
        else:
            # No venue results, just take top results
            final_results = results[:max_results]
        
        return final_results

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]] = None) -> str:
        """Generate a response using the Groq API based on the query and retrieved context.
        
        Args:
            query: The user's query
            context_chunks: Retrieved context chunks (if None, will retrieve context first)
            
        Returns:
            Generated response to the query
        """
        try:
            # Retrieve context if not provided
            if context_chunks is None:
                context_chunks = self.retrieve_context(query)
            
            # Get Groq configuration
            groq_config = self.config.get("API_SETTINGS", {})
            
            # Check if we have a valid API key
            if not self.groq_api_key:
                print("No Groq API key found. Using fallback response.")
                return self._generate_simple_response(query, context_chunks)
            
            system_prompt = groq_config.get("system_prompt", 
                "You are a helpful automotive assistant that answers questions about vehicles. Provide accurate, clear, and concise responses based on the context provided."
            )
            
            # Prepare user message based on available context
            if context_chunks:
                # We have context - include it in the prompt
                context_text = "\n\n".join([f"Context {i+1}: {chunk['text']}" for i, chunk in enumerate(context_chunks)])
                user_message = (
                    f"Based on the following information, please answer my question.\n\n"
                    f"{context_text}\n\n"
                    f"My question is: {query}"
                )
            else:
                # No specific context available - ask model to use general knowledge
                user_message = (
                    f"I have a question, but I don't have specific information in my database about this. "
                    f"Please answer based on your general knowledge.\n\n"
                    f"My question is: {query}\n\n"
                    f"If you can't provide a precise answer due to lack of specific information, please explain "
                    f"what information might be needed and suggest how I could find it."
                )
            
            # Prepare messages for the API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Prepare request parameters
            request_data = {
                "model": self.groq_model,
                "messages": messages,
                "temperature": groq_config.get("temperature", 0.7),
                "max_tokens": groq_config.get("max_tokens", 1024),
                "top_p": groq_config.get("top_p", 0.9)
            }
            
            # Send request to Groq API
            print(f"Sending request to Groq API with model: {self.groq_model}")
            print(f"API Base URL: {self.groq_base_url}")
            print(f"Request data: {json.dumps(request_data, indent=2)}")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.groq_api_key}"
            }
            
            response = requests.post(
                f"{self.groq_base_url}/chat/completions",
                headers=headers,
                json=request_data
            )
            
            # Handle API response
            if response.status_code == 200:
                response_json = response.json()
                answer = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if answer:
                    print("Successfully generated response using Groq API")
                    return answer
                else:
                    print("Received empty response from Groq API. Using fallback response.")
                    return self._generate_simple_response(query, context_chunks)
            else:
                print(f"Error from Groq API: Status code {response.status_code}")
                print(f"Response text: {response.text}")
                print("Using fallback response.")
                return self._generate_simple_response(query, context_chunks)
            
        except Exception as e:
            print(f"Error generating response with Groq: {str(e)}")
            if self.error_handler:
                self.error_handler.log_error(e, "rag_generation", {"context": "response_generation", "query": query})
            return self._generate_simple_response(query, context_chunks)
    
    def _generate_simple_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate a simple response without using the Groq API.
        
        Args:
            query: The user's query
            context_chunks: Retrieved context chunks
            
        Returns:
            A simple generated response
        """
        # Check if we have any context chunks
        if not context_chunks:
            return (
                "I don't have specific information about that in my database. "
                "I can try to help you find this information through other means, "
                "or you could try asking a different question. "
                "Would you like to try asking something else?"
            )
        
        # Simple response generation by combining context information
        relevant_texts = [chunk["text"] for chunk in context_chunks]
        
        # Build a simple response with the most relevant information
        if len(relevant_texts) == 1:
            response = f"Here's what I found: {relevant_texts[0]}"
        else:
            response = "Here's what I found:\n\n"
            for i, text in enumerate(relevant_texts[:3], 1):  # Limit to top 3
                response += f"{i}. {text}\n\n"
        
        # Add source information if available
        sources = []
        for chunk in context_chunks:
            if "source" in chunk["metadata"] and chunk["metadata"]["source"] not in sources:
                sources.append(chunk["metadata"]["source"])
        
        if sources:
            response += "\nThis information comes from: " + ", ".join(sources)
        
        return response
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Add documents to the FAISS index.
        
        Args:
            texts: List of text documents to add
            metadatas: List of metadata dictionaries for each text
            
        Returns:
            True if successful, False otherwise
        """
        if not self.embedding_model or not self.index:
            print("RAG components not properly initialized. Cannot add documents.")
            return False
        
        try:
            if not texts:
                return False
            
            if not metadatas:
                metadatas = [{} for _ in texts]
            
            # Encode texts
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # Get current index id
            start_id = len(self.metadata["texts"])
            
            # Add to FAISS index
            self.index.add(np.array(embeddings).astype('float32'))
            
            # Update metadata
            self.metadata["texts"].extend(texts)
            self.metadata["ids"].extend(list(range(start_id, start_id + len(texts))))
            self.metadata["metadata"].extend(metadatas)
            
            # Save index and metadata
            self._save_index()
            
            print(f"Added {len(texts)} documents to the index. Total: {len(self.metadata['texts'])}")
            return True
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "rag_index", {"context": "document_addition"})
            print(f"Error adding documents to index: {str(e)}")
            return False
    
    def _save_index(self) -> bool:
        """Save the FAISS index and metadata to disk."""
        try:
            if not self.index:
                return False
                
            # Create directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
                
            print(f"Saved index with {self.index.ntotal} vectors and metadata.")
            return True
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "rag_save", {"context": "index_saving"})
            print(f"Error saving index: {str(e)}")
            return False
    
    def listen(self) -> str:
        """Listen for and recognize speech input.
        
        Returns:
            Recognized speech text or empty string if recognition fails
        """
        try:
            with self.microphone as source:
                print("Listening...")
                recognition_config = self.config.get("recognition", {})
                timeout = recognition_config.get("timeout", 10)
                phrase_time_limit = recognition_config.get("phrase_time_limit", 5)
                
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print(f"User said: {text}")
            return text.lower()
        
        except sr.WaitTimeoutError:
            if self.error_handler:
                message, _ = self.error_handler.handle_error(
                    Exception("Listening timed out"), "recognition", {"context": "timeout"}
                )
                print(message)
            else:
                print("Listening timed out. Please try again.")
            return ""
        
        except sr.UnknownValueError:
            if self.error_handler:
                message, _ = self.error_handler.handle_error(
                    Exception("Could not understand audio"), "recognition", {"context": "unclear"}
                )
                print(message)
            else:
                print("Sorry, I didn't catch that. Could you please repeat?")
            return ""
        
        except Exception as e:
            if self.error_handler:
                message, _ = self.error_handler.handle_error(
                    e, "recognition", {"context": "general"}
                )
                print(message)
            else:
                print(f"Error in speech recognition: {str(e)}")
            return ""
    
    def speak(self, text: str) -> None:
        """Convert text to speech and output it.
        
        Args:
            text: Text to be spoken
        """
        try:
            # Format text for natural-sounding speech
            formatted_text = self.formatter.format_response(text)
            print(f"Assistant: {formatted_text}")
            
            self.engine.say(formatted_text)
            self.engine.runAndWait()
        
        except Exception as e:
            if self.error_handler:
                message, _ = self.error_handler.handle_error(
                    e, "synthesis", {"context": "speech_output"}
                )
                print(message)
            else:
                print(f"Error in speech synthesis: {str(e)}")
            # Fall back to just printing the text
            print(f"Assistant (text only): {text}")
    
    def wait_for_wake_word(self) -> bool:
        """Listen for the wake word to activate the assistant.
        
        Returns:
            True if wake word detected, False otherwise
        """
        try:
            with self.microphone as source:
                print(f"Waiting for wake word: '{self.wake_word}'...")
                wake_config = self.config.get("API_SETTINGS", {})
                timeout = wake_config.get("timeout", 10)
                phrase_time_limit = wake_config.get("phrase_time_limit", 3)
                
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            text = self.recognizer.recognize_google(audio).lower()
            print(f"Heard: {text}")
            
            # Check if wake word is in the recognized text
            return self.wake_word in text
        
        except (sr.UnknownValueError, sr.RequestError):
            # Just keep listening on common errors
            return False
        
        except Exception as e:
            if self.error_handler:
                self.error_handler.log_error(e, "wake_word", {"context": "detection"})
            print(f"Error in wake word detection: {str(e)}")
            return False
    
    def adjust_for_ambient_noise(self, duration=1) -> None:
        """Adjust the recognizer for ambient noise to improve recognition.
        
        Args:
            duration: Duration in seconds to sample ambient noise
        """
        try:
            print("Adjusting for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            print("Ambient noise adjustment complete.")
        
        except Exception as e:
            if self.error_handler:
                message, _ = self.error_handler.handle_error(
                    e, "recognition", {"context": "ambient_adjustment"}
                )
                print(message)
            else:
                print(f"Error adjusting for ambient noise: {str(e)}")
    
    def process_query(self, query: str) -> str:
        """Process a query through the RAG pipeline and generate a response.
        
        Args:
            query: The user's query
            
        Returns:
            Generated response to the query
        """
        # Step 1: Retrieve relevant context
        context_chunks = self.retrieve_context(query)
        
        # Step 2: Generate response using retrieved context
        # Always try to use Groq first, even if no context is found
        if self.groq_api_key:
            response = self.generate_response(query, context_chunks)
        else:
            # Fall back to simple response if no Groq API key
            response = self._generate_simple_response(query, context_chunks)
        
        return response
    
    def run(self) -> None:
        """Run the voice assistant in a continuous loop, activated by wake word."""
        print("Voice Assistant initialized. Say the wake word to begin.")
        
        while True:
            try:
                # Wait for wake word
                if self.wait_for_wake_word():
                    self.speak("How can I help you with your vehicle?")
                    
                    # Now actively listen for a command
                    user_input = self.listen()
                    
                    if user_input:
                        # Process the query through our RAG pipeline
                        response = self.process_query(user_input)
                        self.speak(response)
                
                # Small delay to prevent CPU usage from spiking
                time.sleep(0.1)
            
            except KeyboardInterrupt:
                print("\nExiting Voice Assistant...")
                break
            
            except Exception as e:
                if self.error_handler:
                    message, _ = self.error_handler.handle_error(
                        e, "assistant", {"context": "main_loop"}
                    )
                    print(message)
                else:
                    print(f"Error in voice assistant: {str(e)}")
                # Continue the loop unless it's a critical error
                continue


if __name__ == "__main__":
    # Example usage
    error_handler = ErrorHandler()
    
    voice_assistant = VoiceAssistant(
        config_path="config.json",
        error_handler=error_handler,
        wake_word="car assistant"
    )
    
    # Process all PDF documents in the manuals directory
    voice_assistant.process_documents()
    
    # Example of adding documents to the index
    sample_docs = [
        "To change the oil in your car, first locate the drain plug under the vehicle. Place an oil pan under it, then remove the plug to drain the old oil.",
        "The recommended tire pressure for most passenger vehicles is between 32 and 35 PSI (pounds per square inch).",
        "If your check engine light is on, it could indicate various issues from a loose gas cap to serious engine problems. Consider having it diagnosed with an OBD-II scanner."
    ]
    
    sample_metadata = [
        {"source": "Maintenance Manual", "section": "Oil Change", "page": 42},
        {"source": "Tire Specifications", "section": "Pressure Guidelines", "page": 18},
        {"source": "Troubleshooting Guide", "section": "Dashboard Indicators", "page": 103}
    ]
    
    voice_assistant.add_documents(sample_docs, sample_metadata)
    
    # Run the assistant
    voice_assistant.run()