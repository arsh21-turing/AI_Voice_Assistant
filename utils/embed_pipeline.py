"""
Embedding pipeline utilities for processing PDF documents.
This module provides functions to load, process and chunk PDF documents
for use in the voice-powered car assistant.
"""
from utils.helpers import PDFProcessor
import logging
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_chunks(pdf_path):
    """
    Load a PDF document and extract semantic chunks.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        
    Returns:
        list: A list of semantic chunks with their metadata
        
    Raises:
        FileNotFoundError: If the PDF file cannot be found
        Exception: If there's an error processing the PDF
    """
    try:
        logger.info(f"Loading PDF from: {pdf_path}")
        pdf_processor = PDFProcessor(pdf_path)
        
        logger.info("Extracting semantic chunks from PDF")
        chunks = pdf_processor.get_semantic_chunks()
        
        logger.info(f"Successfully extracted {len(chunks)} chunks from the PDF")
        return chunks
    except FileNotFoundError:
        logger.error(f"PDF file not found at path: {pdf_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for a list of semantic chunks using a sentence-transformer model.
    
    Args:
        chunks (list): List of semantic chunks with text and metadata
        model_name (str, optional): Name of the sentence-transformers model to use
            Defaults to 'all-MiniLM-L6-v2'
            
    Returns:
        tuple: (embeddings, metadata, texts) where embeddings is a numpy array of shape 
               (n_chunks, embedding_dim), metadata is a list of dictionaries, and
               texts is a list containing the chunk text content
               
    Raises:
        ValueError: If chunks is empty or doesn't contain expected format
        Exception: If there's an error during embedding generation
    """
    try:
        if not chunks or not isinstance(chunks, list):
            raise ValueError("Chunks must be a non-empty list")
        
        # Extract text and metadata from chunks
        texts = []
        metadata = []
        
        for chunk in chunks:
            if isinstance(chunk, dict) and 'text' in chunk:
                texts.append(chunk['text'])
                # Extract metadata, removing the text to avoid duplication
                chunk_meta = {k: v for k, v in chunk.items() if k != 'text'}
                metadata.append(chunk_meta)
            else:
                # For simple text chunks without metadata
                texts.append(chunk)
                metadata.append({})
        
        logger.info(f"Loading sentence-transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        logger.info(f"Successfully generated embeddings of shape {embeddings.shape}")
        return embeddings, metadata, texts
    
    except ValueError as ve:
        logger.error(f"Value error in generate_embeddings: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings for efficient similarity search.
    
    Args:
        embeddings (numpy.ndarray): Matrix of embeddings to index,
            shape (n_chunks, embedding_dim)
            
    Returns:
        faiss.Index: FAISS index built with IndexFlatL2 for L2 distance search
        
    Raises:
        ValueError: If embeddings is empty or not a numpy array
        Exception: If there's an error creating the FAISS index
    """
    try:
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array")
        
        if embeddings.size == 0:
            raise ValueError("Embeddings array cannot be empty")
        
        # Get the dimensionality of the embeddings
        embedding_dim = embeddings.shape[1]
        
        logger.info(f"Creating FAISS index for {embeddings.shape[0]} embeddings with dimension {embedding_dim}")
        
        # Ensure embeddings are in the correct format (float32)
        embeddings_float32 = embeddings.astype(np.float32)
        
        # Create the FAISS index using L2 distance
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Add the embeddings to the index
        index.add(embeddings_float32)
        
        logger.info(f"FAISS index created successfully with {index.ntotal} vectors")
        return index
        
    except ValueError as ve:
        logger.error(f"Value error in create_faiss_index: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise

def save_index_and_metadata(index, metadata, chunks_text, base_path):
    """
    Save FAISS index and associated metadata to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        metadata (list): List of metadata dictionaries for each chunk
        chunks_text (list): List of text chunks corresponding to the embeddings
        base_path (str): Base file path for saving (without extension)
        
    Returns:
        tuple: (index_path, metadata_path) with the paths to the saved files
        
    Raises:
        ValueError: If inputs are invalid
        OSError: If there's an error writing to disk
        Exception: For any other errors
    """
    try:
        if not index or not isinstance(index, faiss.Index):
            raise ValueError("Index must be a valid FAISS index")
        
        if not isinstance(metadata, list):
            raise ValueError("Metadata must be a list")
            
        if not isinstance(chunks_text, list):
            raise ValueError("chunks_text must be a list")
            
        if len(metadata) != index.ntotal or len(chunks_text) != index.ntotal:
            raise ValueError(
                f"Metadata ({len(metadata)}) and chunks_text ({len(chunks_text)}) " 
                f"must have same length as index vectors ({index.ntotal})"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
        
        # Define file paths
        index_path = f"{base_path}.index"
        metadata_path = f"{base_path}.json"
        
        # Save the FAISS index
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(index, index_path)
        
        # Combine metadata with text chunks
        full_metadata = []
        for i, (meta, text) in enumerate(zip(metadata, chunks_text)):
            entry = {
                "id": i,
                "text": text,
                **meta  # Include all other metadata
            }
            full_metadata.append(entry)
        
        # Save metadata and chunks to JSON file
        logger.info(f"Saving metadata and chunk texts to {metadata_path}")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": full_metadata,
                "index_info": {
                    "vector_count": index.ntotal,
                    "vector_dim": index.d,
                    "index_type": type(index).__name__
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved index and metadata to disk")
        return index_path, metadata_path
        
    except ValueError as ve:
        logger.error(f"Value error in save_index_and_metadata: {str(ve)}")
        raise
    except OSError as oe:
        logger.error(f"Error writing files to disk: {str(oe)}")
        raise
    except Exception as e:
        logger.error(f"Error saving index and metadata: {str(e)}")
        raise

def load_index_and_metadata(base_path):
    """
    Load FAISS index and metadata from disk.
    
    Args:
        base_path (str): Base file path for loading (without extension)
        
    Returns:
        tuple: (faiss.Index, list) containing the loaded index and a list of dictionaries 
               with metadata and chunk text
               
    Raises:
        FileNotFoundError: If index or metadata files don't exist
        ValueError: If the loaded data has unexpected format
        Exception: For any other errors during loading
    """
    try:
        # Define file paths
        index_path = f"{base_path}.index"
        metadata_path = f"{base_path}.json"
        
        # Check if files exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load the FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        
        # Load metadata and chunk texts
        logger.info(f"Loading metadata and chunk texts from {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate the loaded data
        if not isinstance(data, dict) or "chunks" not in data or "index_info" not in data:
            raise ValueError("Invalid metadata file format")
        
        chunks_metadata = data["chunks"]
        index_info = data["index_info"]
        
        # Verify index consistency
        if index.ntotal != index_info.get("vector_count", 0):
            logger.warning(
                f"Index vector count ({index.ntotal}) doesn't match metadata "
                f"vector count ({index_info.get('vector_count', 0)})"
            )
        
        if index.d != index_info.get("vector_dim", 0):
            logger.warning(
                f"Index dimension ({index.d}) doesn't match metadata "
                f"dimension ({index_info.get('vector_dim', 0)})"
            )
        
        logger.info(f"Successfully loaded index with {index.ntotal} vectors and {len(chunks_metadata)} metadata entries")
        return index, chunks_metadata
        
    except FileNotFoundError as fe:
        logger.error(f"File not found: {str(fe)}")
        raise
    except ValueError as ve:
        logger.error(f"Value error in load_index_and_metadata: {str(ve)}")
        raise
    except json.JSONDecodeError as je:
        logger.error(f"Error parsing metadata JSON file: {str(je)}")
        raise ValueError(f"Invalid JSON format in metadata file: {metadata_path}")
    except Exception as e:
        logger.error(f"Error loading index and metadata: {str(e)}")
        raise

def search_similar_chunks(query, index, chunks_metadata, model_name='all-MiniLM-L6-v2', top_k=5):
    """
    Search for chunks similar to a given query using the FAISS index.
    
    Args:
        query (str): User query to search for
        index (faiss.Index): FAISS index to search in
        chunks_metadata (list): List of dictionaries containing chunk metadata and text
        model_name (str, optional): Name of the sentence-transformers model to use
            Defaults to 'all-MiniLM-L6-v2'
        top_k (int, optional): Number of top results to return
            Defaults to 5
            
    Returns:
        list: List of dictionaries containing similar chunks with:
            - text: The text content of the chunk
            - metadata: Any associated metadata (page numbers, etc.)
            - score: Similarity score (lower is better for L2 distance)
            
    Raises:
        ValueError: If inputs are invalid
        Exception: For any other errors during search
    """
    try:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        if not index or not isinstance(index, faiss.Index):
            raise ValueError("Index must be a valid FAISS index")
            
        if not chunks_metadata or not isinstance(chunks_metadata, list):
            raise ValueError("chunks_metadata must be a non-empty list")
            
        if len(chunks_metadata) < index.ntotal:
            logger.warning(
                f"Chunks metadata length ({len(chunks_metadata)}) is less than "
                f"index vector count ({index.ntotal})"
            )
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
            
        # Cap top_k to the number of vectors in the index
        top_k = min(top_k, index.ntotal)
        
        # Load the sentence-transformer model
        logger.info(f"Loading sentence-transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # Generate embedding for the query
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = model.encode([query])[0]
        
        # Ensure the query embedding is in the correct format
        query_embedding_float32 = np.array([query_embedding], dtype=np.float32)
        
        # Search the index
        logger.info(f"Searching index for top {top_k} similar chunks")
        distances, indices = index.search(query_embedding_float32, top_k)
        
        # Get the results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # Check if the index is valid
            if idx < 0 or idx >= len(chunks_metadata):
                logger.warning(f"Search returned invalid index: {idx}")
                continue
                
            # Get the chunk metadata and text
            chunk_info = chunks_metadata[idx]
            
            # Create result entry
            result = {
                "text": chunk_info.get("text", ""),
                "score": float(dist),  # Convert numpy float to Python float
                "rank": i + 1,
            }
            
            # Add all other metadata
            metadata = {k: v for k, v in chunk_info.items() 
                       if k not in ["text", "id"]}
            result["metadata"] = metadata
            
            results.append(result)
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
        
    except ValueError as ve:
        logger.error(f"Value error in search_similar_chunks: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error searching for similar chunks: {str(e)}")
        raise