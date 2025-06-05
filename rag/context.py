# voice_car_assistant/rag/context.py

"""
Context retrieval and conversation management for the RAG pipeline.

This module provides:
1. ContextRetriever - Vector-based semantic search for retrieval
2. ContextManager - Conversation history management
"""

import time
import datetime
import logging
import json 
import re
import os
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import deque
import uuid
import difflib
from pathlib import Path

# For document processing
import numpy as np

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

# Import from project modules
try:
    from voice_car_assistant.config import ConfigManager
    from voice_car_assistant.utils.vector_store import VectorStore
    from voice_car_assistant.utils.scoring import RelevanceScorer
    from voice_car_assistant.utils.error_handler import ErrorHandler
except ImportError:
    # For standalone usage or testing
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from config import ConfigManager
        from utils.vector_store import VectorStore
        from utils.scoring import RelevanceScorer
        from utils.error_handler import ErrorHandler
    except ImportError:
        # Minimal fallback for isolated testing
        class ConfigManager:
            def get(self, *args, **kwargs):
                return kwargs.get('default', None)
            
            def update(self, config_dict):
                pass
                
        class ErrorHandler:
            def handle_error(self, *args, **kwargs):
                return "Error occurred", False
            
            def log_error(self, *args, **kwargs):
                pass

# For document processing capabilities
try:
    # Try to import document processing libraries
    from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAVE_DOCUMENT_PROCESSING = True
except ImportError:
    HAVE_DOCUMENT_PROCESSING = False


class ContextSerializationError(Exception):
    """Exception raised for context serialization/deserialization errors."""
    pass


class DocumentProcessingError(Exception):
    """Exception raised for document processing errors."""
    pass


class ContextRetriever:
    """
    Handles context retrieval and ranking for the RAG pipeline.
    Integrates with vector store for similarity search and scoring.
    
    This class is responsible for:
    1. Initializing and managing the vector store
    2. Performing semantic search on documents
    3. Ranking and filtering context chunks
    4. Processing and loading documents (PDFs, text files)
    5. Integrating with conversation history
    """
    
    def __init__(self, config_manager: ConfigManager, error_handler: ErrorHandler):
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector store
        self.vector_store = VectorStore(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            index_path="data/vector_store"
        )
        
        # Initialize document processor
        self._initialize_document_processor()
        
        # Process initial documents if vector store is empty
        self._process_initial_documents()
        
        # Initialize relevance scorer
        self.relevance_scorer = RelevanceScorer()
        
        # Get configuration values with defaults
        self.default_top_k = self.config_manager.get('context.default_top_k', 5)
        self.relevance_threshold = self.config_manager.get('context.relevance_threshold', 0.5)
        self.use_reranking = self.config_manager.get('context.use_reranking', True)
        self.max_history_size = self.config_manager.get('context.max_history_size', 10)
        
        # Initialize recent contexts
        self.recent_contexts = []
    
    def _initialize_document_processor(self):
        """Initialize document processor with text splitter and PDF loader."""
        try:
            # Get chunk size and overlap from config with defaults
            chunk_size = int(self.config_manager.get('context.chunk_size', 1000))
            chunk_overlap = int(self.config_manager.get('context.chunk_overlap', 200))
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize PDF loader
            self.pdf_loader = PyPDFLoader
            
        except Exception as e:
            self.logger.error(f"Error initializing document processor: {e}")
            self.text_splitter = None
            self.pdf_loader = None
    
    def _process_initial_documents(self):
        """Process documents from manuals directory if vector store is empty."""
        try:
            if not self.vector_store or not self.text_splitter:
                return
                
            manuals_dir = "data/manuals"
            if not os.path.exists(manuals_dir):
                self.logger.warning(f"Manuals directory not found: {manuals_dir}")
                return
                
            # Process each PDF file
            for filename in os.listdir(manuals_dir):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(manuals_dir, filename)
                    try:
                        # Load and split PDF
                        loader = self.pdf_loader(file_path)
                        pages = loader.load()
                        chunks = self.text_splitter.split_documents(pages)
                        
                        # Add to vector store
                        self.vector_store.add_documents(chunks)
                        self.logger.info(f"Processed {filename}: {len(chunks)} chunks")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {filename}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error in initial document processing: {e}")
    
    def retrieve_context(self, 
                        query: str, 
                        top_k: Optional[int] = None, 
                        filter_criteria: Optional[Dict[str, Any]] = None,
                        use_reranking: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks from vector store using similarity search.
        
        Args:
            query (str): The preprocessed query to search for.
            top_k (int, optional): Number of results to retrieve. If None, uses default.
            filter_criteria (Dict[str, Any], optional): Metadata filters for retrieval.
            use_reranking (bool, optional): Whether to apply reranking to initial results.
                If None, uses the value from configuration.
            
        Returns:
            List[Dict[str, Any]]: Relevant context chunks with metadata and scores.
                Each chunk has 'text', 'metadata', and 'score' keys.
        """
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                self.logger.warning("Invalid query provided")
                return []
                
            # Check if vector store is available
            if not self.vector_store:
                self.logger.error("Vector store is not initialized.")
                return []
                
            # Use default top_k if not provided or invalid
            if not isinstance(top_k, int) or top_k <= 0:
                top_k = self.default_top_k
                
            # Use default reranking setting if not specified
            if not isinstance(use_reranking, bool):
                use_reranking = self.use_reranking
                
            # Ensure top_k is within reasonable limits
            top_k = min(max(1, top_k), 20)  # Between 1 and 20
            
            # Add headroom for filtering (retrieve more than needed initially)
            search_k = min(top_k * 3, 50)  # Get up to 3x but max 50
            
            self.logger.debug(f"Retrieving context for query: '{query}', top_k={top_k}")
            
            # Perform similarity search
            raw_results = self.vector_store.search(
                query=query,
                top_k=search_k
            )
            
            if not raw_results:
                self.logger.info("No results found for query")
                return []
            
            # Convert raw results to a consistent format
            formatted_results = []
            for result in raw_results:
                if not isinstance(result, dict):
                    continue
                    
                formatted_result = {
                    'text': str(result.get('text', '')),
                    'metadata': dict(result.get('metadata', {})),
                    'score': float(result.get('score', 0.0))
                }
                formatted_results.append(formatted_result)
            
            if not formatted_results:
                self.logger.info("No valid results after formatting")
                return []
            
            # Apply reranking if enabled and have relevance scorer
            if use_reranking and self.relevance_scorer and len(formatted_results) > top_k:
                reranked_results = self._rerank_results(formatted_results, query)
            else:
                reranked_results = formatted_results
            
            # Filter by relevance threshold and limit to top_k
            filtered_results = [
                result for result in reranked_results 
                if isinstance(result.get('score'), (int, float)) and result['score'] >= self.relevance_threshold
            ][:top_k]
            
            # Update recent contexts
            self._update_recent_contexts(query, filtered_results)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            if self.error_handler:
                self.error_handler.handle_error(
                    e,
                    error_type="context_retrieval",
                    context={
                        "query": query,
                        "top_k": top_k,
                        "filter": filter_criteria
                    }
                )
            return []
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Rerank results using the relevance scorer.
        
        Args:
            results: List of results to rerank
            query: The query string
            
        Returns:
            Reranked list of results
        """
        if not results or not self.relevance_scorer:
            return results
            
        try:
            # Score each result
            scored_results = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                    
                text = str(result.get('text', ''))
                if not text:
                    continue
                    
                # Get relevance score
                score = self.relevance_scorer.score(query, text)
                if not isinstance(score, (int, float)):
                    continue
                    
                # Update result with new score
                result_copy = result.copy()
                result_copy['score'] = float(score)
                scored_results.append(result_copy)
            
            # Sort by score
            scored_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            return scored_results
            
        except Exception as e:
            self.logger.error(f"Error reranking results: {e}")
            return results
    
    def _update_recent_contexts(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Update the list of recent contexts.
        
        Args:
            query: The query string
            results: List of retrieved results
        """
        if not results:
            return
            
        try:
            # Add new contexts
            self.recent_contexts.extend(results)
            
            # Trim to max size
            if len(self.recent_contexts) > self.max_history_size:
                self.recent_contexts = self.recent_contexts[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"Error updating recent contexts: {e}")
    
    def get_recent_contexts(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get the most recent contexts.
        
        Args:
            count: Number of recent contexts to return
            
        Returns:
            List of recent contexts
        """
        try:
            return self.recent_contexts[-count:]
        except Exception as e:
            self.logger.error(f"Error getting recent contexts: {e}")
            return []
    
    def clear_recent_contexts(self) -> None:
        """Clear the recent contexts history."""
        try:
            self.recent_contexts = []
        except Exception as e:
            self.logger.error(f"Error clearing recent contexts: {e}")


# Keep the original ContextManager class for backward compatibility
class ContextManager:
    """
    Manager for conversation history and contextual information.
    
    This class maintains the conversation history, tracks session state,
    and provides relevant context for query processing and response generation.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the context manager.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        # Get configuration
        try:
            from voice_car_assistant.config import ConfigManager
            self.config = config_manager if config_manager else ConfigManager()
        except ImportError:
            # Fallback for backward compatibility
            try:
                from config import get_config
                self.config = config_manager if config_manager else get_config()
            except ImportError:
                # Minimal default config if all else fails
                self.config = config_manager or ConfigManager()
        
        # Initialize conversation history
        self.history = []
        self.interaction_counter = 0
        self.session_id = str(uuid.uuid4())
        self.session_start_time = time.time()
        
        # Initialize session state
        self.session_state = {
            'vehicle_info': {},     # Information about the user's vehicle
            'preferences': {},      # User preferences
            'current_topic': None,  # Current conversation topic
            'flags': {},            # Session flags
        }
        
        # Load context settings from configuration
        self.max_history_size = self.config.get(
            'context.max_history_size', 
            default=5
        )
        self.relevance_threshold = self.config.get(
            'context.relevance_threshold', 
            default=0.5
        )
        self.include_timestamps = self.config.get(
            'context.include_timestamps', 
            default=True
        )
        self.retention_days = self.config.get(
            'context.retention_days', 
            default=1
        )
        self.store_context_chunks = self.config.get(
            'context.store_context_chunks', 
            default=True
        )
        
        # Initialize embedding model for semantic similarity if available
        self.embedding_model = None
        if HAVE_SENTENCE_TRANSFORMERS:
            embedding_model_name = self.config.get(
                'embedding.model_name', 
                default='all-MiniLM-L6-v2'
            )
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                logging.info(f"Initialized embedding model: {embedding_model_name}")
            except Exception as e:
                logging.warning(f"Could not load embedding model: {str(e)}")
        
        logging.info(f"Initialized context manager with session ID: {self.session_id}")
    
    def add_interaction(self, 
                       query: str, 
                       response: str,
                       context_chunks: Optional[List[Dict]] = None,
                       metadata: Optional[Dict] = None) -> int:
        """
        Add a user interaction to the conversation history.
        
        Args:
            query: The user's query
            response: The system's response
            context_chunks: Retrieved context chunks used (optional)
            metadata: Additional metadata about the interaction (optional)
            
        Returns:
            The interaction ID
        """
        # Generate a unique interaction ID
        self.interaction_counter += 1
        interaction_id = self.interaction_counter
        
        # Get current timestamp
        timestamp = time.time()
        formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create interaction record
        interaction = {
            'id': interaction_id,
            'timestamp': timestamp,
            'formatted_time': formatted_time,
            'query': query,
            'response': response,
            'session_id': self.session_id,
        }
        
        # Add context chunks if provided and enabled
        if self.store_context_chunks and context_chunks:
            # Store only essential information from context chunks to save space
            simplified_chunks = []
            for chunk in context_chunks:
                simplified_chunk = {
                    'text': chunk.get('text', ''),
                    'metadata': chunk.get('metadata', {})
                }
                simplified_chunks.append(simplified_chunk)
            
            interaction['context_chunks'] = simplified_chunks
        
        # Add metadata if provided
        if metadata:
            interaction['metadata'] = metadata
        
        # Add to history
        self.history.append(interaction)
        
        # Apply retention policy
        self.prune_history()
        
        logging.debug(f"Added interaction #{interaction_id} to history")
        return interaction_id
    
    def get_recent_interactions(self, count: Optional[int] = None) -> List[Dict]:
        """
        Get the most recent interactions from conversation history.
        
        Args:
            count: Number of recent interactions to retrieve 
                   (defaults to max_history_size from config)
                   
        Returns:
            List of recent interactions with their data
        """
        # Use configured max size if count not specified
        if count is None:
            count = self.max_history_size
            
        # Return the most recent interactions up to count
        return list(self.history[-count:]) if count > 0 else []
    
    def get_relevant_interactions(self, 
                                 query: str, 
                                 threshold: float = 0.0) -> List[Dict]:
        """
        Get interactions relevant to the current query.
        
        Args:
            query: Current user query
            threshold: Minimum relevance score (0-1), defaults to config value if 0
            
        Returns:
            Relevant past interactions with relevance scores added
        """
        # Use configured threshold if not specified
        if threshold <= 0:
            threshold = self.relevance_threshold
            
        relevant_interactions = []
        
        for interaction in self.history:
            # Skip if no past query
            past_query = interaction.get('query', '')
            if not past_query:
                continue
                
            # Compute similarity between current query and past query
            similarity = self.compute_query_similarity(query, past_query)
            
            # If similarity exceeds threshold, add to relevant interactions
            if similarity >= threshold:
                # Create a copy with similarity score added
                relevant_interaction = interaction.copy()
                relevant_interaction['relevance_score'] = similarity
                relevant_interactions.append(relevant_interaction)
        
        # Sort by relevance score (most relevant first)
        relevant_interactions.sort(
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        
        return relevant_interactions
    
    def get_conversation_context(self, 
                                query: str, 
                                format: str = 'text') -> Union[str, Dict, List]:
        """
        Get formatted conversation context for the current query.
        
        Args:
            query: Current user query
            format: Output format ('text', 'dict', or 'structured')
            
        Returns:
            Formatted conversation context in the requested format
        """
        # Get recent and relevant interactions
        recent = self.get_recent_interactions()
        relevant = self.get_relevant_interactions(query)
        
        # Combine and deduplicate (recent interactions take precedence)
        recent_ids = set(r['id'] for r in recent)
        combined = list(recent)  # Start with recent interactions
        
        # Add relevant interactions that aren't already in recent
        for interaction in relevant:
            if interaction['id'] not in recent_ids:
                combined.append(interaction)
        
        # Sort by original timestamp order
        combined.sort(key=lambda x: x.get('timestamp', 0))
        
        # Format according to requested output format
        if format == 'text':
            return self._format_context_as_text(combined)
        elif format == 'dict':
            return combined
        elif format == 'structured':
            return self._format_context_as_structured(combined)
        else:
            logging.warning(f"Unknown context format '{format}', using 'text'")
            return self._format_context_as_text(combined)
    
    def update_session_state(self, key: str, value: Any) -> None:
        """
        Update session state with a new value.
        
        Args:
            key: State parameter name
            value: State parameter value
        """
        # Handle nested keys with dot notation (e.g., "vehicle_info.make")
        if '.' in key:
            parts = key.split('.')
            parent = self.session_state
            
            # Navigate to the correct nested dictionary
            for part in parts[:-1]:
                if part not in parent or not isinstance(parent[part], dict):
                    parent[part] = {}
                parent = parent[part]
            
            # Set the value in the deepest level
            parent[parts[-1]] = value
        else:
            # Set top-level key
            self.session_state[key] = value
            
        logging.debug(f"Updated session state: {key} = {value}")
    
    def get_session_state(self, key: str, default: Any = None) -> Any:
        """
        Get a session state value.
        
        Args:
            key: State parameter name
            default: Default value if key not found
            
        Returns:
            The stored state value or default
        """
        # Handle nested keys with dot notation
        if '.' in key:
            parts = key.split('.')
            current = self.session_state
            
            # Navigate through the nested dictionaries
            for part in parts:
                if part not in current:
                    return default
                current = current[part]
                
            return current
        else:
            # Get top-level key
            return self.session_state.get(key, default)
    
    def clear_history(self, retain_count: int = 0) -> None:
        """
        Clear conversation history.
        
        Args:
            retain_count: Number of most recent interactions to retain
        """
        if retain_count <= 0:
            # Clear all history
            self.history = []
        else:
            # Retain the specified number of most recent interactions
            self.history = self.history[-retain_count:] if retain_count < len(self.history) else self.history
            
        logging.info(f"Cleared history (retained {retain_count} most recent interactions)")
    
    def prune_history(self) -> None:
        """
        Apply retention policy to conversation history.
        
        This removes old interactions based on both count and time constraints.
        """
        # Apply maximum history size constraint
        if len(self.history) > self.max_history_size:
            # Keep only the most recent interactions
            self.history = self.history[-self.max_history_size:]
        
        # Apply time-based retention policy
        if self.retention_days > 0:
            cutoff_time = time.time() - (self.retention_days * 24 * 60 * 60)
            
            # Keep only interactions newer than the cutoff time
            self.history = [
                interaction for interaction in self.history
                if interaction.get('timestamp', 0) >= cutoff_time
            ]
    
    def compute_query_similarity(self, query1: str, query2: str) -> float:
        """
        Compute similarity between two queries.
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Similarity score between 0 and 1
        """
        # If the queries are identical, return 1.0
        if query1 == query2:
            return 1.0
            
        # Try using semantic similarity if embedding model is available
        if self.embedding_model is not None:
            try:
                embedding1 = self.embedding_model.encode(query1)
                embedding2 = self.embedding_model.encode(query2)
                
                # Compute cosine similarity
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                norm1 = sum(a * a for a in embedding1) ** 0.5
                norm2 = sum(b * b for b in embedding2) ** 0.5
                
                if norm1 * norm2 == 0:
                    return 0.0
                    
                return dot_product / (norm1 * norm2)
                
            except Exception as e:
                logging.warning(f"Error computing semantic similarity: {str(e)}")
                # Fall back to lexical similarity
        
        # Fallback: Use lexical similarity (difflib)
        return difflib.SequenceMatcher(None, query1, query2).ratio()
    
    def serialize(self) -> Dict:
        """
        Serialize context history and state for persistent storage.
        
        Returns:
            Serialized context data as a dictionary
            
        Raises:
            ContextSerializationError: If serialization fails
        """
        try:
            serialized = {
                'session_id': self.session_id,
                'session_start_time': self.session_start_time,
                'interaction_counter': self.interaction_counter,
                'history': self.history,
                'session_state': self.session_state
            }
            
            # Test JSON serialization to ensure it's serializable
            json.dumps(serialized)
            
            return serialized
            
        except Exception as e:
            logging.error(f"Error serializing context: {str(e)}")
            raise ContextSerializationError(f"Could not serialize context: {str(e)}")
    
    def deserialize(self, data: Dict) -> None:
        """
        Load context history and state from serialized data.
        
        Args:
            data: Serialized context data
            
        Raises:
            ContextSerializationError: If deserialization fails
        """
        try:
            # Load basic session information
            self.session_id = data.get('session_id', str(uuid.uuid4()))
            self.session_start_time = data.get('session_start_time', time.time())
            self.interaction_counter = data.get('interaction_counter', 0)
            
            # Load history and state
            self.history = data.get('history', [])
            self.session_state = data.get('session_state', {
                'vehicle_info': {},
                'preferences': {},
                'current_topic': None,
                'flags': {},
            })
            
            logging.info(f"Deserialized context with {len(self.history)} interactions")
            
        except Exception as e:
            logging.error(f"Error deserializing context: {str(e)}")
            raise ContextSerializationError(f"Could not deserialize context: {str(e)}")
    
    def get_vehicle_info(self) -> Dict:
        """
        Get information about the user's vehicle.
        
        Returns:
            Dictionary of vehicle information
        """
        return self.session_state.get('vehicle_info', {})
    
    def set_vehicle_info(self, info_dict: Dict) -> None:
        """
        Set information about the user's vehicle.
        
        Args:
            info_dict: Dictionary of vehicle information
        """
        vehicle_info = self.session_state.get('vehicle_info', {})
        vehicle_info.update(info_dict)
        self.session_state['vehicle_info'] = vehicle_info
        
        logging.debug(f"Updated vehicle information: {info_dict}")
    
    def _format_context_as_text(self, interactions: List[Dict]) -> str:
        """
        Format context as a text string for prompt inclusion.
        
        Args:
            interactions: List of interaction records
            
        Returns:
            Formatted context as text
        """
        context_parts = []
        
        for i, interaction in enumerate(interactions):
            # Format timestamp if enabled
            timestamp_str = ""
            if self.include_timestamps and 'formatted_time' in interaction:
                timestamp_str = f" [{interaction['formatted_time']}]"
                
            # Format the current interaction
            context_parts.append(
                f"User{timestamp_str}: {interaction.get('query', '')}\n"
                f"Assistant: {interaction.get('response', '')}"
            )
        
        # Combine all parts with separators
        return "\n\n".join(context_parts)
    
    def _format_context_as_structured(self, interactions: List[Dict]) -> List[Dict]:
        """
        Format context as a structured format for prompt inclusion.
        
        Args:
            interactions: List of interaction records
            
        Returns:
            List of structured message objects
        """
        structured_context = []
        
        for interaction in interactions:
            # Add user message
            structured_context.append({
                "role": "user",
                "content": interaction.get('query', '')
            })
            
            # Add assistant message
            structured_context.append({
                "role": "assistant",
                "content": interaction.get('response', '')
            })
        
        return structured_context
    
    def extract_vehicle_info_from_query(self, query: str) -> Dict:
        """
        Extract potential vehicle information from a user query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted vehicle information
        """
        info = {}
        
        # Simple pattern matching for common vehicle information
        # Year pattern (1990-2030)
        year_match = re.search(r'\b(19[9][0-9]|20[0-2][0-9]|2030)\b', query)
        if year_match:
            info['year'] = year_match.group(1)
            
        # Common car makes
        make_pattern = r'\b(toyota|honda|ford|chevrolet|chevy|bmw|audi|lexus|mercedes|benz|hyundai|kia|subaru|volkswagen|vw|nissan|mazda|tesla|volvo)\b'
        make_match = re.search(make_pattern, query.lower())
        if make_match:
            info['make'] = make_match.group(1)
            
        # If we found any information, update the session
        if info:
            self.set_vehicle_info(info)
            
        return info

# For testing or direct usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Example test of ContextRetriever
    try:
        from voice_car_assistant.config import ConfigManager
        config_manager = ConfigManager()
        
        # Set test configuration
        config_manager.update({
            'paths': {
                'vector_store_index': './data/test_index'
            },
            'embedding': {
                'model_name': 'all-MiniLM-L6-v2'
            },
            'retrieval': {
                'top_k': 3,
                'relevance_threshold': 0.6,
                'use_reranking': True
            },
            'document_processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200
            }
        })
        
        retriever = ContextRetriever(config_manager=config_manager)
        
        # Test PDF processing if available
        if HAVE_DOCUMENT_PROCESSING:
            pdf_path = './data/sample_manual.pdf'
            if os.path.exists(pdf_path):
                print(f"Testing PDF processing with {pdf_path}")
                chunks = retriever.process_pdf(pdf_path)
                print(f"Processed PDF into {len(chunks)} chunks")
                
                # Add chunks to vector store
                retriever.vector_store.add_texts(
                    [chunk['text'] for chunk in chunks],
                    metadatas=[chunk['metadata'] for chunk in chunks]
                )
                print("Added chunks to vector store")
                
                # Test retrieval
                query = "How to change oil in my car?"
                results = retriever.retrieve_context(query)
                print(f"\nRetrieved {len(results)} chunks for query: {query}")
                for i, result in enumerate(results):
                    print(f"\nResult {i+1} (Score: {result['score']:.2f}):")
                    print(f"Text: {result['text'][:100]}...")
                
        else:
            print("Document processing not available. Install langchain package.")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
    
    # Test backward compatibility with ContextManager
    print("\nTesting ContextManager for backward compatibility:")
    context_manager = ContextManager()
    
    # Add some test interactions
    context_manager.add_interaction(
        "How do I check my oil level?",
        "To check your oil level, make sure your car is on level ground and the engine is cool. "
        "Locate the dipstick, pull it out and wipe it clean. Then insert it back fully and pull it out again. "
        "The oil level should be between the two marks on the dipstick."
    )
    
    # Extract vehicle info
    vehicle_info = context_manager.extract_vehicle_info_from_query("I have a 2019 Toyota Camry, how do I change the wiper blades?")
    print(f"Extracted vehicle info: {vehicle_info}")
    
    # Test relevance
    query = "What type of oil should I use in my car?"
    relevant = context_manager.get_relevant_interactions(query)
    print(f"\nRelevant interactions for '{query}':")
    for interaction in relevant:
        score = interaction.get('relevance_score', 0)
        print(f" - Score: {score:.2f}, Query: '{interaction.get('query')}'")