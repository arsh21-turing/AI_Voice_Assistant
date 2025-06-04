"""
Configuration module for the Voice-Powered Car Assistant.

This module provides access to the application's configuration settings.
"""

from typing import Any, Dict, Optional
from .settings import ConfigManager

# Global configuration instance
_config_instance = None

def get_config() -> ConfigManager:
    """
    Get the global configuration instance.
    
    Returns:
        ConfigManager: The global configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance

def initialize_config(env_file: str = ".env", config_file: str = "config.json") -> ConfigManager:
    """
    Initialize the global configuration with custom parameters.
    
    Args:
        env_file: Path to environment variables file
        config_file: Path to optional JSON configuration file
        
    Returns:
        ConfigManager: The initialized configuration instance
    """
    global _config_instance
    _config_instance = ConfigManager(env_file=env_file, config_file=config_file)
    return _config_instance

# Default export for easy access
config = get_config()

# Version information
__version__ = "0.1.0"