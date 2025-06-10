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
import logging
from datetime import datetime
import sys
import traceback

# Add tqdm for progress bars with fallback
try:
    from tqdm import tqdm
except ImportError:
    # Create a simple fallback if tqdm is not available
    def tqdm(iterable=None, **kwargs):
        if iterable is not None:
            # Simple fallback that just returns the iterable
            return iterable
        else:
            # If used as a context manager
            class DummyTqdmContextManager:
                def __init__(self, **kwargs): 
                    self.total = kwargs.get('total', 0)
                    self.desc = kwargs.get('desc', '')
                    print(f"{self.desc}: Started processing {self.total} items")
                
                def update(self, n=1): pass
                def __enter__(self): return self
                def __exit__(self, *args, **kwargs): 
                    print(f"{self.desc}: Completed processing")
            
            return DummyTqdmContextManager(**kwargs)
    print("Warning: tqdm module not found. Progress bars will not be displayed.")
    print("Install tqdm for progress bars: pip install tqdm")

# Import for direct RAG implementation
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber

from utils.error_handler import ErrorHandler
from utils.embedding_cache import EmbeddingCache
from utils.logger import VoiceAssistantLogger
from utils.config_manager import ConfigManager
from utils.command_handler import CommandHandler
from utils.helpers import PDFProcessor

from speech.formatting import VoiceResponseFormatter


def extract_text_from_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """Extract text from PDF and split into chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries containing text chunks and metadata
    """
    try:
        print(f"\nExtracting text from PDF: {pdf_path}")
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Show total number of pages
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            # Extract text from each page with progress bar
            print("\nExtracting text from pages:")
            for page_num, page in tqdm(enumerate(pdf.pages, 1), total=total_pages, desc="Pages", unit="page"):
                text = page.extract_text()
                if not text:
                    continue
                    
                # Split page text into chunks
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    if end > len(text):
                        end = len(text)
                        
                    chunk_text = text[start:end]
                    if chunk_text.strip():  # Only add non-empty chunks
                        chunks.append({
                            "text": chunk_text,
                            "pages": [page_num],
                            "start_char": start,
                            "end_char": end
                        })
                    
                    start = end - chunk_overlap
        
        print(f"\nCreated {len(chunks)} chunks from {pdf_path}")
        return chunks
        
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def create_index(texts: List[str], embedding_model_name: str = "all-MiniLM-L6-v2", embedding_cache: Optional["EmbeddingCache"] = None) -> Tuple[Any, np.ndarray]:
    """Create a FAISS index from a list of texts with optimized embedding generation.
    
    Args:
        texts: List of text chunks to index
        embedding_model_name: Name of the SentenceTransformer model to use
        embedding_cache: Optional cache for storing and reusing embeddings
    
    Returns:
        Tuple of (FAISS index, numpy array of embeddings)
    """
    try:
        if not texts:
            print("No texts provided for indexing")
            return None, None
            
        print(f"\nCreating index with {len(texts)} text chunks using {embedding_model_name}")
        
        # Load the embedding model with optimized settings
        model = SentenceTransformer(
            embedding_model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            cache_folder='./model_cache'  # Cache the model locally
        )
        
        # Set batch size based on available memory
        batch_size = 32 if torch.cuda.is_available() else 16
        
        # Generate embeddings - either using cache or direct encoding
        print("\nGenerating embeddings...")
        if embedding_cache is not None:
            # Update embedder if needed
            if embedding_cache.embedder is None:
                embedding_cache.set_embedder(model)
                
            # Use cache for embeddings with progress bar
            # First, check how many items are not in cache
            uncached_count = 0
            for text in texts:
                if text.strip() not in embedding_cache.cache:
                    uncached_count += 1
            
            if uncached_count > 0:
                print(f"Generating embeddings for {uncached_count} uncached chunks...")
            
            # Get embeddings with progress tracking for uncached items
            with tqdm(total=len(texts), desc="Embeddings", unit="chunk") as pbar:
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:min(i+batch_size, len(texts))]
                    batch_embeddings = embedding_cache.get_embeddings(batch)
                    embeddings.extend(batch_embeddings)
                    pbar.update(len(batch))
            
            # Get cache stats for logging
            cache_stats = embedding_cache.get_cache_stats()
            print(f"\nEmbedding cache stats: {cache_stats}")
        else:
            # Generate embeddings without cache but with progress bar
            print(f"\nProcessing in batches of {batch_size} chunks...")
            embeddings = []
            
            with tqdm(total=len(texts), desc="Embeddings", unit="chunk") as pbar:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:min(i+batch_size, len(texts))]
                    batch_embeddings = model.encode(
                        batch,
                        show_progress_bar=False,  # We're handling our own progress bar
                        convert_to_tensor=False,
                        normalize_embeddings=True
                    )
                    embeddings.extend(batch_embeddings)
                    pbar.update(len(batch))
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index with optimized settings
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use Inner Product for normalized vectors
        
        # Add vectors to the index with progress bar
        print("\nAdding vectors to FAISS index...")
        index.add(embeddings_np)
        
        print(f"\nSuccessfully created index with {index.ntotal} vectors of dimension {dimension}")
        return index, embeddings_np
        
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        return None, None


class VoiceAssistant:
    """Consolidated class for speech recognition, synthesis, and RAG using SentenceTransformer and FAISS."""
    
    def __init__(self):
        """Initialize the voice assistant with speech recognition and synthesis."""
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Default speech rate
        self.engine.setProperty('volume', 1.0)  # Default volume
        
        # Get available voices
        self.voices = self.engine.getProperty('voices')
        self.current_voice_index = 0
        self.engine.setProperty('voice', self.voices[self.current_voice_index].id)
        
        # Initialize command handler with profiles directory
        self.command_handler = CommandHandler(profiles_dir="profiles")
        
        # Store the last response for repeat command
        self.last_response = ""
        
        # Initialize error handler
        self.error_handler = ErrorHandler()
        
        # Initialize document processor
        self.doc_processor = None  # Initialize as None, will be set when processing a PDF
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize wake word detector
        self.wake_word = "hey assistant"
        
        # Initialize voice settings
        self.voice_settings = {
            "rate": 150,
            "volume": 1.0,
            "voice_id": self.voices[self.current_voice_index].id
        }
        
        # Load default profile if available
        self._load_default_profile()
        
        # Use ConfigManager to load configuration
        self.config_manager = ConfigManager("config.json")
        self.config = self.config_manager.config
        
        # Initialize logger first so it's available for other components
        self._initialize_logger()
        
        # Log system startup
        self.logger.log_system("initialization", f"Initializing VoiceAssistant with wake word: '{self.wake_word}'")
        
        # Initialize components that depend on configuration
        with self.logger.time_operation("init_speech_components"):
            self._initialize_speech_components()
            
        with self.logger.time_operation("init_embedding_cache"):
            self._initialize_embedding_cache()
            
        with self.logger.time_operation("init_api_clients"):
            self._initialize_api_clients()
            
        with self.logger.time_operation("init_rag_components"):
            self._initialize_rag_components()
        
        # Adjust for ambient noise when initialized
        with self.logger.time_operation("adjust_ambient_noise"):
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self.logger.log_system("initialization_complete", "VoiceAssistant initialization completed successfully")
    
    def _initialize_logger(self):
        """Initialize the logging system."""
        log_config = self.config.get("logging", {})
        log_dir = log_config.get("log_dir", "logs")
        log_level_name = log_config.get("log_level", "INFO")
        
        # Convert log level name to logging constant
        log_level = getattr(logging, log_level_name, logging.INFO)
        
        # Initialize the logger
        self.logger = VoiceAssistantLogger(
            log_dir=log_dir,
            app_name="voice_assistant",
            log_level=log_level,
            enable_console=log_config.get("enable_console", True),
            max_log_files=log_config.get("max_log_files", 10),
            max_file_size_mb=log_config.get("max_file_size_mb", 10)
        )

    def _initialize_speech_components(self):
        """Initialize speech recognition and synthesis components."""
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
    
    def _initialize_embedding_cache(self):
        """Initialize the embedding cache for RAG."""
        cache_cfg = self.config.get("embedding_cache", {})
        limit = cache_cfg.get("size_limit", 5000)
        self.embedding_cache = EmbeddingCache(size_limit=limit, model_name=None)
        print(f"Embedding cache initialized (limit={limit})")
    
    def _initialize_api_clients(self):
        """Initialize API clients and settings."""
        print("\nInitializing API clients...")
        
        # Get Groq API configuration
        groq_config = self.config.get("API_SETTINGS", {})
        
        # Try to get API key from environment first, then config
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            self.groq_api_key = groq_config.get("GROQ_API_KEY")
            
        if not self.groq_api_key:
            print("ERROR: No Groq API key found! Please set GROQ_API_KEY environment variable or in config.json")
            self.client = None
            return
            
        print("Groq API key found")
        self.groq_model = groq_config.get("GROQ_MODEL", "llama3-8b-8192")
        self.groq_base_url = groq_config.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        
        # Initialize Groq client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.groq_api_key,
                base_url=self.groq_base_url
            )
            # Test the client with a simple request
            test_response = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            print("Successfully initialized and tested Groq client")
        except Exception as e:
            print(f"Error initializing Groq client: {str(e)}")
            import traceback
            traceback.print_exc()
            self.client = None
    
    def _initialize_rag_components(self):
        """Initialize RAG components (SentenceTransformer and FAISS)."""
        print("\nInitializing RAG components...")
        
        try:
            # Get RAG configuration
            rag_config = self.config.get("rag", {})
            
            # Initialize with empty values for graceful degradation in case of errors
            self.embedding_model = None
            self.index = None
            self.metadata = {"texts": [], "ids": [], "metadata": [], "document_hashes": {}}
            
            # Initialize embedding model
            model_name = rag_config.get("model_name", "all-MiniLM-L6-v2")
            print(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            print("Embedding model loaded successfully")
            
            # Set the embedding model in the cache
            if hasattr(self, 'embedding_cache'):
                self.embedding_cache.set_embedder(self.embedding_model)
            
            # FAISS index settings
            self.index_path = rag_config.get("index_path", "data/index/manual_index")
            self.metadata_path = os.path.join(self.index_path, "metadata.json")
            
            # Set default retrieval parameters
            self.top_k = rag_config.get("top_k", 3)
            self.relevance_threshold = rag_config.get("relevance_threshold", 0.6)
            self.chunk_size = rag_config.get("chunk_size", 1000)
            self.chunk_overlap = rag_config.get("chunk_overlap", 200)
            
            print("Loading or creating FAISS index...")
            # Load or create FAISS index
            self._load_or_create_index()
            
            print("RAG components initialized successfully")
            
        except Exception as e:
            print(f"Error initializing RAG components: {str(e)}")
            import traceback
            traceback.print_exc()
            if self.error_handler:
                self.error_handler.log_error(e, "rag_init", {"context": "initialization"})
    
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
        """Generate a hash for a file based on its content and modification time.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash of the file
        """
        try:
            # Get file stats
            stats = os.stat(file_path)
            # Combine file size and modification time
            file_info = f"{stats.st_size}_{stats.st_mtime}"
            # Generate MD5 hash
            return hashlib.md5(file_info.encode()).hexdigest()
        except Exception as e:
            print(f"Error generating file hash: {str(e)}")
            return ""
    
    def _extract_vehicle_name(self, file_name: str, pdf_path: str) -> str:
        """Extract vehicle name from file name or path.
        
        Args:
            file_name: Name of the PDF file
            pdf_path: Full path to the PDF file
            
        Returns:
            Extracted vehicle name
        """
        # Try to get vehicle name from parent directory first
        parent_dir = os.path.basename(os.path.dirname(pdf_path))
        if parent_dir and parent_dir != ".":
            return parent_dir
            
        # Otherwise try to extract from filename
        name = file_name.replace(".pdf", "")
        # Remove common prefixes/suffixes
        name = name.replace("manual", "").replace("guide", "").strip()
        return name if name else "Unknown Vehicle"
        
    def _extract_document_type(self, file_name: str) -> str:
        """Extract document type from file name.
        
        Args:
            file_name: Name of the PDF file
            
        Returns:
            Extracted document type
        """
        file_name = file_name.lower()
        if "manual" in file_name:
            return "Manual"
        elif "guide" in file_name:
            return "Guide"
        elif "spec" in file_name:
            return "Specification"
        elif "warranty" in file_name:
            return "Warranty"
        else:
            return "Document"
    
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
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a query using the RAG system."""
        try:
            print(f"Retrieving context for query: {query}")
            
            if not self.embedding_model or not self.index:
                print("RAG components not initialized")
                return []
            
            # Get query embedding
            print("Generating query embedding...")
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            
            # Search the index
            print("Searching index...")
            distances, indices = self.index.search(
                np.array([query_embedding]).astype('float32'),
                self.top_k
            )
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                # Convert distance to similarity score
                similarity = 1.0 - (distance / 2.0)  # FAISS uses L2 distance
                
                if similarity >= self.relevance_threshold:
                    results.append({
                        "text": self.metadata["texts"][idx],
                        "metadata": self.metadata["metadata"][idx],
                        "similarity": float(similarity)
                    })
            
            print(f"Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            print(f"Error in context retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def process_specific_pdf(self, pdf_path: str, force_reload: bool = False) -> bool:
        """Process a specific PDF document and add it to the index.
        
        Args:
            pdf_path: Path to the PDF file to process
            force_reload: Whether to force reprocessing even if already indexed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Log the start of processing
            print(f"\n{'='*50}")
            print(f"Processing PDF: {os.path.basename(pdf_path)}")
            print(f"{'='*50}\n")
            
            if hasattr(self, 'logger'):
                self.logger.log_system("pdf_processing", f"Processing {pdf_path}", {
                    "force_reload": force_reload
                })
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                return False
            
            # Initialize PDFProcessor for this specific PDF
            self.doc_processor = PDFProcessor(pdf_path)
            
            # Check if already processed
            file_hash = self._get_file_hash(pdf_path)
            file_name = os.path.basename(pdf_path)
            
            if not force_reload and file_hash in self.metadata.get("document_hashes", {}):
                print(f"Skipping already indexed document: {file_name}")
                print(f"Use force_reload=True to reprocess this file.")
                return True
            
            # Extract document info
            vehicle_name = self._extract_vehicle_name(file_name, pdf_path)
            document_type = self._extract_document_type(file_name)
            
            print(f"\n[1/4] Extracting and chunking text from PDF...")
            # Extract and chunk text with progress bar (implemented in extract_text_from_pdf)
            chunks = extract_text_from_pdf(
                pdf_path, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            if not chunks:
                print(f"No text extracted from {file_name}")
                return False
            
            print(f"\n[2/4] Preparing {len(chunks)} chunks for indexing...")
            # Prepare batch data
            batch_texts = []
            batch_metadatas = []
            
            # Show progress for metadata preparation
            with tqdm(total=len(chunks), desc="Metadata", unit="chunk") as pbar:
                for chunk in chunks:
                    batch_texts.append(chunk["text"])
                    batch_metadatas.append({
                        "source": file_name,
                        "document_type": document_type,
                        "vehicle": vehicle_name,
                        "pages": chunk.get("pages", []),
                        "chunk_index": len(batch_texts) - 1,
                        "total_chunks": len(chunks)
                    })
                    pbar.update(1)
            
            if not batch_texts:
                print("No valid text chunks to process")
                return False
                
            print(f"\n[3/4] Generating embeddings for {len(batch_texts)} chunks...")
            
            # Make sure embedding_cache has the embedder set
            if hasattr(self, 'embedding_cache') and self.embedding_cache.embedder is None:
                self.embedding_cache.set_embedder(self.embedding_model)
                
            # Generate embeddings
            if hasattr(self, 'embedding_cache'):
                # Use create_index function with progress info
                _, embeddings_np = create_index(batch_texts, embedding_cache=self.embedding_cache)
                
                if embeddings_np is None:
                    print("Failed to generate embeddings")
                    return False
            else:
                # Fallback if embedding cache is not available
                print("Embedding cache not available, falling back to direct encoding")
                batch_size = 32 if torch.cuda.is_available() else 16
                
                embeddings = []
                with tqdm(total=len(batch_texts), desc="Embeddings", unit="chunk") as pbar:
                    for i in range(0, len(batch_texts), batch_size):
                        batch = batch_texts[i:min(i+batch_size, len(batch_texts))]
                        batch_embeddings = self.embedding_model.encode(
                            batch, 
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                        embeddings.extend(batch_embeddings)
                        pbar.update(len(batch))
                        
                embeddings_np = np.array(embeddings).astype('float32')
            
            print(f"\n[4/4] Adding vectors to the FAISS index and saving...")
            
            # Add to the FAISS index
            if self.index is not None:
                # Get current index size
                start_id = len(self.metadata["texts"])
                
                # Add vectors to the index
                self.index.add(embeddings_np)
                
                # Update metadata
                self.metadata["texts"].extend(batch_texts)
                self.metadata["metadata"].extend(batch_metadatas)
                
                # Add IDs (needed for proper metadata tracking)
                new_ids = list(range(start_id, start_id + len(batch_texts)))
                self.metadata["ids"].extend(new_ids)
                
                # Update document hash information
                self.metadata["document_hashes"][file_hash] = {
                    "path": file_name,
                    "last_modified": os.path.getmtime(pdf_path),
                    "chunk_count": len(batch_texts),
                    "processed_date": time.time()
                }
                
                # Save the index and metadata
                save_success = self._save_index()
                if not save_success:
                    print("Warning: Failed to save index after adding PDF chunks")
                
                print(f"\n{'='*50}")    
                print(f"Successfully processed {file_name}:")
                print(f"- {len(batch_texts)} chunks added to index")
                print(f"- {self.index.ntotal} total vectors in index")
                print(f"{'='*50}\n")
                return True
            else:
                print("FAISS index not initialized")
                return False
            
        except Exception as e:
            error_type = "pdf_processing"
            if "audio" in str(e).lower():
                error_type = "audio_device"
            elif "embedding" in str(e).lower():
                error_type = "embedding"
            elif "index" in str(e).lower():
                error_type = "index"
            
            message, should_retry = self.error_handler.handle_error(
                e, 
                error_type,
                {
                    "file": pdf_path,
                    "context": "pdf_processing",
                    "retryable": error_type in ["network", "timeout", "rate_limit"]
                }
            )
            print(message)
            return False

    def _save_index(self) -> bool:
        """Save the FAISS index and metadata to disk."""
        try:
            # Log the start of index saving
            self.logger.log_system("index_saving", "start", {
                "index_path": self.index_path,
                "total_vectors": self.index.ntotal if self.index else 0
            })
            
            if not self.index:
                self.logger.log_error(
                    ValueError("No index to save"),
                    "index_saving",
                    {"stage": "validation"}
                )
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save FAISS index
            with self.logger.time_operation("faiss_save"):
                faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
            
            # Save metadata
            with self.logger.time_operation("metadata_save"):
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.metadata, f)
            
            # Log successful save
            self.logger.log_system("index_saving", "success", {
                "index_path": self.index_path,
                "total_vectors": self.index.ntotal,
                "metadata_size": len(self.metadata["texts"])
            })
            
            return True
            
        except Exception as e:
            self.logger.log_error(e, "index_saving", {
                "index_path": self.index_path
            })
            return False

    def generate_response(self, query: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the LLM with context from RAG."""
        try:
            # Log the start of response generation
            self.logger.log_interaction("response_generation", "start", {
                "query": query,
                "has_context": context is not None,
                "context_length": len(context) if context else 0
            })
            
            # Prepare the prompt with context if available
            if context:
                context_text = "\n".join([f"Context {i+1}: {item['text']}" for i, item in enumerate(context)])
                prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {query}

Answer:"""
            else:
                prompt = f"Question: {query}\nAnswer:"

            # Generate response using Groq
            with self.logger.time_operation("llm_generation"):
                response = self.client.chat.completions.create(
                    model=self.config["API_SETTINGS"]["GROQ_MODEL"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
            
            answer = response.choices[0].message.content.strip()
            
            # Log successful response generation
            self.logger.log_interaction("response_generation", "success", {
                "query": query,
                "response_length": len(answer),
                "has_context": context is not None
            })
            
            return answer
            
        except Exception as e:
            self.logger.log_error(e, "response_generation", {
                "query": query,
                "has_context": context is not None
            })
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def listen(self) -> Optional[str]:
        """Listen for user input and return transcribed text."""
        try:
            print("\nListening...")
            with self.microphone as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=5
                )
                
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"\nI heard: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            error_type = "network"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            
            message, should_retry = self.error_handler.handle_error(
                e,
                error_type,
                {
                    "context": "speech_recognition",
                    "retryable": error_type in ["timeout", "network"]
                }
            )
            print(message)
            return None
        except Exception as e:
            error_type = "audio_device"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            elif "network" in str(e).lower():
                error_type = "network"
            
            message, should_retry = self.error_handler.handle_error(
                e,
                error_type,
                {
                    "context": "audio_listening",
                    "retryable": error_type in ["timeout", "network"]
                }
            )
            print(message)
            return None

    def speak(self, text: str) -> None:
        """Convert text to speech and speak it.
        
        Args:
            text: The text to speak
        """
        try:
            # Store the response for repeat command
            self.last_response = text
            self.command_handler.set_last_response(text)
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
        except Exception as e:
            self.error_handler.handle_error(
                ErrorType.SPEECH,
                str(e),
                "Failed to speak the response",
                {"text": text}
            )
    
    def process_command(self, text: str) -> Tuple[bool, Optional[str]]:
        """Process a voice command.
        
        Args:
            text: The command text
            
        Returns:
            Tuple of (is_command, response)
        """
        try:
            # Try to process as a command
            is_command, response, updated_settings = self.command_handler.process_command(
                text,
                self.voice_settings
            )
            
            if is_command:
                # Update voice settings if they changed
                if updated_settings != self.voice_settings:
                    self.voice_settings = updated_settings
                    
                    # Apply new settings
                    self.engine.setProperty('rate', updated_settings.get("rate", 150))
                    self.engine.setProperty('volume', updated_settings.get("volume", 1.0))
                    
                    # Handle voice selection
                    if "requested_voice" in updated_settings:
                        self._handle_voice_selection(updated_settings["requested_voice"])
                        # Remove the temporary flag
                        del updated_settings["requested_voice"]
                
                return True, response
            
            return False, None
            
        except Exception as e:
            self.error_handler.handle_error(
                ErrorType.GENERAL,
                str(e),
                "Error processing voice command",
                {"command": text}
            )
            return False, None
    
    def _handle_voice_selection(self, voice_type: str) -> None:
        """Handle voice selection request.
        
        Args:
            voice_type: The type of voice requested ("male", "female", or "different")
        """
        try:
            if voice_type == "male":
                # Find a male voice
                for i, voice in enumerate(self.voices):
                    if "male" in voice.name.lower():
                        self.current_voice_index = i
                        self.engine.setProperty('voice', voice.id)
                        self.voice_settings["voice_id"] = voice.id
                        return
                
                # If no male voice found, just switch to next
                self.current_voice_index = (self.current_voice_index + 1) % len(self.voices)
                self.engine.setProperty('voice', self.voices[self.current_voice_index].id)
                self.voice_settings["voice_id"] = self.voices[self.current_voice_index].id
                
            elif voice_type == "female":
                # Find a female voice
                for i, voice in enumerate(self.voices):
                    if "female" in voice.name.lower():
                        self.current_voice_index = i
                        self.engine.setProperty('voice', voice.id)
                        self.voice_settings["voice_id"] = voice.id
                        return
                
                # If no female voice found, just switch to next
                self.current_voice_index = (self.current_voice_index + 1) % len(self.voices)
                self.engine.setProperty('voice', self.voices[self.current_voice_index].id)
                self.voice_settings["voice_id"] = self.voices[self.current_voice_index].id
                
            else:  # "different"
                # Just switch to next voice
                self.current_voice_index = (self.current_voice_index + 1) % len(self.voices)
                self.engine.setProperty('voice', self.voices[self.current_voice_index].id)
                self.voice_settings["voice_id"] = self.voices[self.current_voice_index].id
            
        except Exception as e:
            self.error_handler.handle_error(
                ErrorType.SPEECH,
                str(e),
                "Error changing voice",
                {"voice_type": voice_type}
            )
    
    def wait_for_wake_word(self) -> bool:
        """Listen for the wake word to activate the assistant.
        
        Returns:
            True if wake word detected, False otherwise
        """
        try:
            with self.microphone as source:
                print(f"Waiting for wake word: '{self.wake_word}'...")
                # Adjust for ambient noise before listening
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Set timeout and phrase time limit
                timeout = 5  # Reduced timeout to 5 seconds
                phrase_time_limit = 3
                
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    text = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                    return self.wake_word in text
                except sr.WaitTimeoutError:
                    print("No speech detected within timeout period")
                    return False
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return False
                except sr.RequestError as e:
                    error_type = "network"
                    if "timeout" in str(e).lower():
                        error_type = "timeout"
                    
                    message, should_retry = self.error_handler.handle_error(
                        e,
                        error_type,
                        {
                            "context": "wake_word_recognition",
                            "retryable": error_type in ["timeout", "network"]
                        }
                    )
                    print(message)
                    return False
                
        except Exception as e:
            error_type = "audio_device"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            elif "network" in str(e).lower():
                error_type = "network"
            
            message, should_retry = self.error_handler.handle_error(
                e,
                error_type,
                {
                    "context": "wake_word_detection",
                    "retryable": error_type in ["timeout", "network"]
                }
            )
            print(message)
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
        try:
            print(f"\n{'='*50}")
            print(f"Processing query: {query}")
            print(f"{'='*50}")
            
            # Step 1: Try to retrieve relevant context from RAG
            print("\nSearching for relevant context...")
            context_chunks = self.retrieve_context(query)
            print(f"Found {len(context_chunks)} relevant context chunks")
            
            # Step 2: Generate response using Groq
            if self.client:
                print("\nGenerating response using Groq...")
                # Prepare the prompt with context if available
                if context_chunks:
                    context_text = "\n".join([f"Context {i+1}: {item['text']}" for i, item in enumerate(context_chunks)])
                    prompt = f"""Based on the following context, answer the question. If the context doesn't contain relevant information, say so.

Context:
{context_text}

Question: {query}

Answer:"""
                else:
                    prompt = f"""Answer the following question about vehicles. If you're not sure about something, say so.

Question: {query}

Answer:"""

                print("\nSending request to Groq...")
                # Generate response using Groq
                response = self.client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                print("\nReceived response from Groq:")
                print(f"\n{'='*50}")
                print(response.choices[0].message.content.strip())
                print(f"{'='*50}\n")
                return response.choices[0].message.content.strip()
            else:
                print("\nGroq client not initialized, using fallback response")
                # Fallback response if Groq is not available
                if context_chunks:
                    return f"I found some relevant information, but I'm unable to generate a proper response at the moment. Please try again later."
                else:
                    return "I'm sorry, I'm unable to process your request at the moment. Please try again later."
                
        except Exception as e:
            error_type = "api"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            elif "network" in str(e).lower():
                error_type = "network"
            elif "rate limit" in str(e).lower():
                error_type = "rate_limit"
            elif "context" in str(e).lower():
                error_type = "context"
            elif "retrieval" in str(e).lower():
                error_type = "retrieval"
            
            message, should_retry = self.error_handler.handle_error(
                e,
                error_type,
                {
                    "query": query,
                    "has_context": context_chunks is not None and len(context_chunks) > 0,
                    "retryable": error_type in ["timeout", "network", "rate_limit"]
                }
            )
            print(f"\nError: {message}")
            return "I encountered an error while processing your request. Please try again."
    
    def handle_voice_command(self, command_text: str) -> bool:
        """Handle a voice command to control the assistant's behavior.
        
        Args:
            command_text: The command from the user
            
        Returns:
            True if command was handled, False if not recognized as a command
        """
        try:
            # Get current voice settings
            current_settings = {
                "rate": self.engine.getProperty('rate'),
                "volume": self.engine.getProperty('volume'),
                "voice_id": self.engine.getProperty('voice').id if hasattr(self.engine.getProperty('voice'), 'id') else None
            }
            
            # Process the command
            is_command, response, new_settings = self.command_handler.process_command(
                command_text, current_settings
            )
            
            if not is_command:
                return False
                
            # Handle special voice selection request
            if "requested_voice" in new_settings:
                voice_type = new_settings.pop("requested_voice")
                # Change to requested voice type
                voices = self.engine.getProperty('voices')
                if voices:
                    # Find appropriate voice
                    target_voices = [v for v in voices if (
                        (voice_type == "male" and not "female" in v.name.lower()) or
                        (voice_type == "female" and "female" in v.name.lower()) or
                        (voice_type == "different" and v.id != current_settings["voice_id"])
                    )]
                    
                    if target_voices:
                        # Set the first matching voice
                        self.engine.setProperty('voice', target_voices[0].id)
                        self.config["voice"]["voice_id"] = target_voices[0].id
                        # Save the updated configuration
                        if hasattr(self, 'config_manager'):
                            self.config_manager.update_section("voice", self.config["voice"])
            
            # Apply new settings
            for setting, value in new_settings.items():
                if hasattr(self.engine, 'setProperty'):
                    self.engine.setProperty(setting, value)
                    # Update config to persist settings
                    self.config["voice"][setting] = value
            
            # If settings were changed, save the updated configuration
            if hasattr(self, 'config_manager'):
                self.config_manager.update_section("voice", self.config["voice"])
                
            # Log the command if we have a logger
            if hasattr(self, 'logger'):
                self.logger.log_system("voice_command", "Command processed", {
                    "command": command_text,
                    "new_settings": new_settings
                })
            
            # Speak the response
            self.speak(response)
            
            return True
            
        except Exception as e:
            # Handle any errors
            if self.error_handler:
                self.error_handler.log_error(e, "voice_command", {"command": command_text})
            print(f"Error handling voice command: {str(e)}")
            return False

    def run(self):
        """Run the voice assistant main loop."""
        print("Voice Assistant is running. Say 'hey assistant' to begin.")
        
        while True:
            try:
                # Listen for wake word
                if not self.wait_for_wake_word():
                    continue
                
                # Listen for command or query
                text = self.listen()
                if not text:
                    continue
                
                # Try to process as a command first
                is_command, response = self.process_command(text)
                if is_command:
                    self.speak(response)
                    continue
                
                # Process as a normal query
                response = self.process_query(text)
                if response:
                    self.speak(response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
                
            except Exception as e:
                self.error_handler.handle_error(
                    ErrorType.GENERAL,
                    str(e),
                    "Error in main loop",
                    {"error": str(e)}
                )
                continue

    def list_voice_commands(self) -> str:
        """Return a list of available voice commands as a string."""
        commands = [
            "speak slower/faster",
            "volume up/down",
            "repeat that/say again",
            "use male/female voice",
            "reset voice settings",
            "what commands can I say"
        ]
        
        return "Available voice commands: " + ", ".join(commands)

    def _load_default_profile(self):
        """Load the default profile if it exists."""
        try:
            # Try to load the default profile
            if self.command_handler.profile_manager.set_current_profile("default"):
                # Get profile settings
                settings = self.command_handler.profile_manager.get_profile_settings("default")
                self.voice_settings.update(settings)
                
                # Apply settings
                self.engine.setProperty('rate', settings.get("rate", 150))
                self.engine.setProperty('volume', settings.get("volume", 1.0))
                self.engine.setProperty('voice', settings.get("voice_id", self.voices[0].id))
                
                print("Loaded default voice profile.")
        except Exception as e:
            print(f"Error loading default profile: {str(e)}")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data/manuals", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler("logs/startup.log"),
            logging.StreamHandler()
        ]
    )
    
    startup_logger = logging.getLogger("startup")
    startup_logger.info("Application starting")
    
    try:
        # Initialize error handler
        error_handler = ErrorHandler()
        
        # Process command-line arguments
        if len(sys.argv) > 1:
            # Get the PDF filename from the command-line argument
            pdf_filename = sys.argv[1]
            
            # Ensure .pdf extension
            if not pdf_filename.lower().endswith('.pdf'):
                pdf_filename += '.pdf'
            
            pdf_path = f"data/manuals/{pdf_filename}"
            
            startup_logger.info(f"Processing PDF file: {pdf_filename}")
            
            # Check if the file exists
            if not os.path.exists(pdf_path):
                startup_logger.error(f"File not found: {pdf_path}")
                print(f"ERROR: File not found: {pdf_path}")
                print("Make sure the file exists in the data/manuals/ directory.")
                sys.exit(1)
                
            # Create the voice assistant
            startup_logger.info("Initializing VoiceAssistant...")
            assistant = VoiceAssistant()
            
            # Process the specified PDF file
            print(f"\n=== Processing PDF file: {pdf_path} ===\n")
            
            # Use the process_specific_pdf method
            success = assistant.process_specific_pdf(pdf_path)
            
            if success:
                startup_logger.info(f"Successfully processed {pdf_filename}")
                print(f"\n=== Successfully processed {pdf_filename} ===\n")
                
                # Test a query related to the PDF content
                base_filename = os.path.basename(pdf_filename).replace('.pdf', '')
                test_query = f"Tell me about {base_filename}"
                
                startup_logger.info(f"Testing with query: {test_query}")
                print(f"\nQuery: '{test_query}'")
                
                response = assistant.process_query(test_query)
                print(f"\nResponse: {response}\n")
                
                # Display embedding cache stats
                if hasattr(assistant, 'embedding_cache'):
                    cache_stats = assistant.embedding_cache.get_cache_stats()
                    print(f"Embedding cache stats: {cache_stats}\n")
                
                # Interactive mode
                print("\n=== Interactive mode ===")
                print("Type 'exit' to quit or ask a question about the PDF content:")
                
                while True:
                    user_input = input("\nYour question: ")
                    if user_input.lower() in ('exit', 'quit', 'q'):
                        break
                        
                    response = assistant.process_query(user_input)
                    print(f"\nResponse: {response}")
                
                startup_logger.info("Application ended normally")
                
            else:
                startup_logger.error(f"Failed to process {pdf_filename}")
                print(f"\n=== Failed to process {pdf_filename} ===\n")
                sys.exit(1)
        else:
            # No specific PDF provided, run in normal mode
            startup_logger.info("Starting voice assistant in normal mode")
            
            # Create and run the voice assistant
            assistant = VoiceAssistant()
            print("Voice Assistant is starting up...")
            assistant.run()
            
    except Exception as e:
        startup_logger.exception(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")
        
        if 'error_handler' in locals():
            error_handler.handle_error(e, "main", {"context": "startup"})
        
        traceback.print_exc()
        sys.exit(1)