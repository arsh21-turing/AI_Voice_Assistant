import os
import json
import logging
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration management for the voice assistant application."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger("voice_assistant.config")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default if it doesn't exist.
        
        Returns:
            Loaded configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"Loaded configuration from {self.config_path}")
                return config
            except json.JSONDecodeError as e:
                print(f"Error parsing {self.config_path}: {str(e)}. Using default configuration.")
                return self._get_default_config()
            except Exception as e:
                print(f"Error loading {self.config_path}: {str(e)}. Using default configuration.")
                return self._get_default_config()
        else:
            print(f"Configuration file not found at {self.config_path}. Creating with default settings.")
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings.
        
        Returns:
            Default configuration dictionary
        """
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
            "embedding_cache": {
                "size_limit": 10000,
                "enable_analytics": True
            },
            "API_SETTINGS": {
                "GROQ_API_KEY": "",
                "GROQ_MODEL": "llama3-8b-8192",
                "GROQ_API_BASE": "https://api.groq.com/openai/v1"
            },
            "logging": {
                "log_dir": "logs",
                "log_level": "INFO",
                "enable_console": True,
                "max_log_files": 10,
                "max_file_size_mb": 10
            }
        }
    
    def _save_config(self, config: Dict[str, Any] = None) -> bool:
        """Save configuration to file.
        
        Args:
            config: Configuration to save (uses self.config if None)
            
        Returns:
            True if successful, False otherwise
        """
        if config is None:
            config = self.config
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2, sort_keys=False)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get a value from the configuration.
        
        Args:
            section: Section name
            key: Key within the section (if None, returns entire section)
            default: Default value if section or key doesn't exist
            
        Returns:
            Value from configuration or default
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def set(self, section: str, key: str, value: Any, save: bool = True) -> bool:
        """Set a value in the configuration.
        
        Args:
            section: Section name
            key: Key within the section
            value: Value to set
            save: Whether to save the configuration to file
            
        Returns:
            True if successful, False otherwise
        """
        # Create section if it doesn't exist
        if section not in self.config:
            self.config[section] = {}
        
        # Set the value
        self.config[section][key] = value
        
        # Save if requested
        if save:
            return self._save_config()
        return True
    
    def update_section(self, section: str, values: Dict[str, Any], save: bool = True) -> bool:
        """Update or create a section in the configuration.
        
        Args:
            section: Section name
            values: Dictionary of values to set
            save: Whether to save the configuration to file
            
        Returns:
            True if successful, False otherwise
        """
        # Create section if it doesn't exist
        if section not in self.config:
            self.config[section] = {}
        
        # Update with new values
        self.config[section].update(values)
        
        # Save if requested
        if save:
            return self._save_config()
        return True
    
    def reload(self) -> bool:
        """Reload configuration from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config = self._load_config()
            return True
        except:
            return False 