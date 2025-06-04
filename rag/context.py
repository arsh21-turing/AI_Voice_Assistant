"""
Conversation context management for the RAG pipeline.

This module provides the ContextManager class for maintaining conversation
history, tracking session state, and providing contextual information for
query understanding and response generation.
"""

import time
import datetime
import logging
import json 
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import deque
import uuid
import difflib
from pathlib import Path

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False

# Import from project modules
try:
    from config import get_config
except ImportError:
    # For standalone usage or testing
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config

class ContextSerializationError(Exception):
    """Exception raised for context serialization/deserialization errors."""
    pass

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
        self.config = config_manager if config_manager else get_config()
        
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
            'CONTEXT_SETTINGS', 
            'MAX_CONVERSATION_TURNS', 
            5
        )
        self.relevance_threshold = self.config.get(
            'CONTEXT_SETTINGS', 
            'RELEVANCE_THRESHOLD', 
            0.5
        )
        self.include_timestamps = self.config.get(
            'CONTEXT_SETTINGS', 
            'INCLUDE_TIMESTAMPS', 
            True
        )
        self.retention_days = self.config.get(
            'CONTEXT_SETTINGS', 
            'RETENTION_DAYS', 
            1
        )
        self.store_context_chunks = self.config.get(
            'CONTEXT_SETTINGS', 
            'STORE_CONTEXT_CHUNKS', 
            True
        )
        
        # Initialize embedding model for semantic similarity if available
        self.embedding_model = None
        if HAVE_SENTENCE_TRANSFORMERS:
            embedding_model_name = self.config.get(
                'RAG_SETTINGS', 
                'EMBEDDING_MODEL', 
                'all-MiniLM-L6-v2'
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
            
        # Try to detect models - this is very simplistic and would need to be expanded
        # with a more comprehensive database of vehicle makes and models
        
        # If we found any information, update the session
        if info:
            self.set_vehicle_info(info)
            
        return info


# For testing or direct usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Example test
    context_manager = ContextManager()
    
    # Add some test interactions
    context_manager.add_interaction(
        "How do I check my oil level?",
        "To check your oil level, make sure your car is on level ground and the engine is cool. "
        "Locate the dipstick, pull it out and wipe it clean. Then insert it back fully and pull it out again. "
        "The oil level should be between the two marks on the dipstick."
    )
    
    context_manager.add_interaction(
        "When should I change my oil?",
        "For most modern vehicles, it's recommended to change your oil every 7,500 to 10,000 miles or every 6 months, "
        "whichever comes first. However, if you drive in severe conditions like extreme temperatures, dusty environments, "
        "or primarily short trips, you should change it more frequently, around every 3,000 to 5,000 miles."
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
    
    # Get formatted context
    context = context_manager.get_conversation_context(query)
    print("\nFormatted context:")
    print(context)
    
    # Test serialization
    serialized = context_manager.serialize()
    print(f"\nSerialized {len(serialized['history'])} interactions")
    
    # Create a new context manager and deserialize
    new_context_manager = ContextManager()
    new_context_manager.deserialize(serialized)
    print(f"Deserialized context has {len(new_context_manager.history)} interactions")