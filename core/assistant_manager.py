"""
core/assistant_manager.py
Conversation state and context management for the Voice-Powered Car Assistant.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import os
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class AssistantManager:
    """
    Maintains conversation state and context between user interactions.
    Provides context management for more intelligent and continuous conversations.
    """

    def __init__(
        self,
        max_context_entries: int = 10,
        context_file_path: Optional[str] = None,
        save_context: bool = True,
        include_timestamps: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the AssistantManager.

        Args:
            max_context_entries: Maximum number of conversation entries to maintain (default: 10)
            context_file_path: Path to save/load context from (default: None, does not persist)
            save_context: Whether to save context to disk (default: True)
            include_timestamps: Whether to include timestamps in context entries (default: True)
            log_level: Logging level (default: logging.INFO)
        """
        # Set up logger
        self._configure_logger(log_level)
        
        logger.info("Initializing AssistantManager")
        
        # Configure context settings
        self.max_context_entries = max_context_entries
        self.save_context = save_context
        self.include_timestamps = include_timestamps
        
        # Set context file path or use default
        if context_file_path:
            self.context_file_path = context_file_path
        else:
            # Default to a context file in the data directory
            data_dir = Path(os.path.join(os.path.dirname(__file__), '..' ,'data'))
            data_dir.mkdir(exist_ok=True)
            self.context_file_path = os.path.join(data_dir, 'assistant_context.json')
            
        logger.debug(f"Context file path set to: {self.context_file_path}")
        
        # Initialize context state
        self.conversation_history = []  # List of conversation entries
        self.current_context = {
            "session_id": self._generate_session_id(),
            "session_start": self._get_timestamp(),
            "vehicle": {},            # Vehicle-specific context
            "user": {},               # User preferences and information
            "current_topic": None,    # Current conversation topic
            "conversation_history": self.conversation_history,
            "system_status": {
                "started": self._get_timestamp()
            }
        }
        
        # Try to load existing context if available
        if self.save_context:
            self._load_context()
        
        logger.info("AssistantManager initialized successfully")

    def _configure_logger(self, log_level: int) -> None:
        """
        Configure logger for this class.
        
        Args:
            log_level: Logging level to use
        """
        global logger
        logger.setLevel(log_level)
        
        # Create handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Prevent log propagation to avoid duplicate logs
            logger.propagate = False
            
        logger.debug("Logger configured for AssistantManager")

    def _generate_session_id(self) -> str:
        """
        Generate a unique session ID.
        
        Returns:
            str: Unique session ID
        """
        timestamp = int(time.time())
        return f"session_{timestamp}"

    def _get_timestamp(self) -> str:
        """
        Get the current timestamp in ISO format.
        
        Returns:
            str: Current timestamp
        """
        return datetime.now().isoformat()

    def update_context(self, new_context: Dict[str, Any]) -> None:
        """
        Update the current context with new information.
        
        Args:
            new_context: Dictionary of context data to update/add
        """
        logger.debug(f"Updating context with new data: {new_context}")
        
        # Update context with new information
        for key, value in new_context.items():
            if key == "conversation_history":
                # Skip directly updating conversation history through this method
                continue
            elif isinstance(value, dict) and key in self.current_context and isinstance(self.current_context[key], dict):
                # Merge dictionaries for existing dict fields
                self.current_context[key].update(value)
            else:
                # Direct update for other fields
                self.current_context[key] = value
        
        # Save context if enabled
        if self.save_context:
            self._save_context()
        
        logger.debug("Context updated successfully")

    def add_conversation_entry(
        self, 
        query: str, 
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new query-response pair to the conversation history.
        
        Args:
            query: User query
            response: Assistant response
            metadata: Additional metadata about the conversation turn
        """
        # Create conversation entry
        entry = {
            "query": query,
            "response": response
        }
        
        # Add timestamp if enabled
        if self.include_timestamps:
            entry["timestamp"] = self._get_timestamp()
            
        # Add metadata if provided
        if metadata:
            entry["metadata"] = metadata
            
        # Add to history
        self.conversation_history.append(entry)
        logger.debug(f"Added conversation entry: query='{query[:30]}{'...' if len(query) > 30 else ''}'")
        
        # Trim history if exceeding max entries
        if len(self.conversation_history) > self.max_context_entries:
            removed = self.conversation_history.pop(0)
            logger.debug(f"Removed oldest conversation entry to maintain history limit")
        
        # Save context if enabled
        if self.save_context:
            self._save_context()

    def get_context(self) -> Dict[str, Any]:
        """
        Get the current conversation context.
        
        Returns:
            Dict: Current context including conversation history
        """
        # Update the conversation history field in case it was modified directly
        self.current_context["conversation_history"] = self.conversation_history
        return self.current_context

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent conversation history.
        
        Args:
            limit: Maximum number of history entries to return (default: all)
            
        Returns:
            List[Dict]: Recent conversation history
        """
        if limit is None or limit >= len(self.conversation_history):
            return self.conversation_history
        else:
            # Return the most recent entries up to the limit
            return self.conversation_history[-limit:]

    def clear_context(self, new_session: bool = True) -> None:
        """
        Clear the current conversation context.
        
        Args:
            new_session: Whether to create a new session ID (default: True)
        """
        logger.info("Clearing conversation context")
        
        # Reset conversation history
        self.conversation_history = []
        
        # Reset current context to initial state
        self.current_context = {
            "session_id": self._generate_session_id() if new_session else self.current_context.get("session_id"),
            "session_start": self._get_timestamp() if new_session else self.current_context.get("session_start"),
            "vehicle": {},
            "user": {},
            "current_topic": None,
            "conversation_history": self.conversation_history,
            "system_status": {
                "started": self._get_timestamp() if new_session else self.current_context.get("system_status", {}).get("started")
            }
        }
        
        # Save empty context if enabled
        if self.save_context:
            self._save_context()
            
        logger.info("Context cleared successfully" + (" with new session" if new_session else ""))

    def set_vehicle_info(self, vehicle_info: Dict[str, Any]) -> None:
        """
        Set vehicle information in the context.
        
        Args:
            vehicle_info: Dictionary of vehicle information
        """
        logger.info("Updating vehicle information")
        self.current_context["vehicle"] = vehicle_info
        
        # Save context if enabled
        if self.save_context:
            self._save_context()

    def set_user_info(self, user_info: Dict[str, Any]) -> None:
        """
        Set user information in the context.
        
        Args:
            user_info: Dictionary of user information
        """
        logger.info("Updating user information")
        self.current_context["user"] = user_info
        
        # Save context if enabled
        if self.save_context:
            self._save_context()

    def get_current_topic(self) -> Optional[str]:
        """
        Get the current conversation topic.
        
        Returns:
            Optional[str]: Current topic or None
        """
        return self.current_context.get("current_topic")

    def set_current_topic(self, topic: str) -> None:
        """
        Set the current conversation topic.
        
        Args:
            topic: Current conversation topic
        """
        logger.debug(f"Setting current topic to: {topic}")
        self.current_context["current_topic"] = topic
        
        # Save context if enabled
        if self.save_context:
            self._save_context()

    def get_session_duration(self) -> float:
        """
        Get the current session duration in seconds.
        
        Returns:
            float: Session duration in seconds
        """
        start_time = datetime.fromisoformat(self.current_context["session_start"])
        current_time = datetime.now()
        return (current_time - start_time).total_seconds()

    def _save_context(self) -> bool:
        """
        Save the current context to a file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.save_context:
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.context_file_path), exist_ok=True)
            
            # Prepare context for saving (ensure conversation_history is updated)
            save_data = self.get_context()
            
            # Write to file
            with open(self.context_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            logger.debug(f"Context saved to {self.context_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving context: {e}", exc_info=True)
            return False

    def _load_context(self) -> bool:
        """
        Load context from a file if it exists.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(self.context_file_path):
            logger.debug(f"No existing context file found at {self.context_file_path}")
            return False
            
        try:
            with open(self.context_file_path, 'r') as f:
                loaded_context = json.load(f)
                
            # Update current context with loaded data
            self.current_context.update(loaded_context)
            
            # Extract conversation history
            if "conversation_history" in loaded_context:
                self.conversation_history = loaded_context["conversation_history"]
                
            logger.info(f"Context loaded from {self.context_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading context: {e}", exc_info=True)
            return False

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update AssistantManager settings.
        
        Args:
            settings: Dictionary of settings to update
        """
        logger.info("Updating AssistantManager settings")
        
        if 'max_context_entries' in settings:
            self.max_context_entries = settings['max_context_entries']
            logger.debug(f"Updated max_context_entries to {self.max_context_entries}")
            
        if 'save_context' in settings:
            self.save_context = settings['save_context']
            logger.debug(f"Updated save_context to {self.save_context}")
            
        if 'include_timestamps' in settings:
            self.include_timestamps = settings['include_timestamps']
            logger.debug(f"Updated include_timestamps to {self.include_timestamps}")
            
        if 'context_file_path' in settings:
            self.context_file_path = settings['context_file_path']
            logger.debug(f"Updated context_file_path to {self.context_file_path}")
            
        if 'log_level' in settings:
            self._configure_logger(settings['log_level'])
            
        # Save with new settings if enabled
        if self.save_context:
            self._save_context()


if __name__ == "__main__":
    # Set up basic logging configuration for the test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple test
    manager = AssistantManager(save_context=True)
    
    # Set vehicle and user info
    manager.set_vehicle_info({
        "make": "Toyota",
        "model": "Corolla",
        "year": 2020,
        "vin": "ABC123XYZ456789",
        "last_service": "2023-09-15"
    })
    
    manager.set_user_info({
        "name": "John",
        "preferred_temperature": 72,
        "preferred_music_genre": "Rock"
    })
    
    # Add some conversation entries
    manager.add_conversation_entry(
        "What's the recommended tire pressure for my car?",
        "The recommended tire pressure for your 2020 Toyota Corolla is 32 PSI for front tires and 30 PSI for rear tires.",
        {"topic": "maintenance", "subtopic": "tire_pressure"}
    )
    
    manager.add_conversation_entry(
        "When is my next oil change due?",
        "Based on your last service date of September 15, 2023, and average driving habits, your next oil change should be scheduled for December 15, 2023.",
        {"topic": "maintenance", "subtopic": "oil_change"}
    )
    
    # Get and print context
    context = manager.get_context()
    print("\nCurrent context:")
    print(json.dumps(context, indent=2))
    
    # Test context clearing
    manager.clear_context()
    print("\nAfter clearing context:")
    print(json.dumps(manager.get_context(), indent=2))