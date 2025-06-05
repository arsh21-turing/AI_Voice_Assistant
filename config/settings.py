"""
Configuration management module for the Voice-Powered Car Assistant.

This module provides centralized configuration management for all components
of the application, including API keys, speech parameters, RAG settings,
prompt templates, error handling, and logging configuration.
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dotenv import load_dotenv

class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass

class ConfigManager:
    """
    Centralized configuration management for the entire application.
    
    This class handles loading configuration from environment variables and/or
    configuration files, provides access to configuration values, and validates
    the configuration to ensure all required settings are available.
    """
    
    # Required configuration keys by section
    REQUIRED_CONFIG = {
        'API_SETTINGS': ['GROQ_API_KEY', 'GROQ_MODEL'],
        'SPEECH_SETTINGS': ['RECOGNITION_ENGINE', 'SYNTHESIS_ENGINE'],
        'RAG_SETTINGS': ['EMBEDDING_MODEL', 'INDEX_PATH'],
    }
    
    def __init__(self, env_file: str = ".env", config_file: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            env_file: Path to environment variables file
            config_file: Path to optional JSON configuration file
        """
        # Initialize empty configuration
        self.config = {
            'API_SETTINGS': {},
            'SPEECH_SETTINGS': {},
            'RAG_SETTINGS': {},
            'PROMPT_TEMPLATES': {},
            'LOGGING_SETTINGS': {},
            'ERROR_HANDLING': {},
            'CONTEXT_SETTINGS': {},
            'APP_SETTINGS': {},
        }
        
        # Load environment variables first
        load_dotenv(env_file)
        
        # Load configurations
        env_config = self.load_from_env()
        file_config = self.load_from_file(config_file)
        
        # Merge configurations (environment takes precedence over file)
        self._merge_configs(file_config)
        self._merge_configs(env_config)
        
        # Set default values
        self._set_defaults()
        
        # Validate the configuration
        self.validate_config()

    def _merge_configs(self, config_to_merge: Dict[str, Dict[str, Any]]) -> None:
        """
        Merge another configuration into the current configuration.
        
        Args:
            config_to_merge: Configuration dictionary to merge
        """
        for section, values in config_to_merge.items():
            if section in self.config:
                self.config[section].update(values)
            else:
                self.config[section] = values
    
    def _set_defaults(self) -> None:
        """Set default values for optional configuration settings."""
        # API defaults
        api_defaults = {
            'GROQ_API_BASE': 'https://api.groq.com/openai/v1',
            'GROQ_MODEL': 'llama3-8b-8192',
            'MAX_TOKENS': 1024,
            'TEMPERATURE': 0.7,
        }
        for key, value in api_defaults.items():
            if key not in self.config['API_SETTINGS']:
                self.config['API_SETTINGS'][key] = value
        
        # Speech defaults
        speech_defaults = {
            'RECOGNITION_ENGINE': 'google',
            'SYNTHESIS_ENGINE': 'pyttsx3',
            'LANGUAGE': 'en-US',
            'WAKE_WORD': 'assistant',
            'VOICE_RATE': 150,
            'VOICE_VOLUME': 1.0,
            'ENERGY_THRESHOLD': 300,
            'PAUSE_THRESHOLD': 0.5,
            'DYNAMIC_ENERGY_THRESHOLD': True,
        }
        for key, value in speech_defaults.items():
            if key not in self.config['SPEECH_SETTINGS']:
                self.config['SPEECH_SETTINGS'][key] = value
        
        # RAG defaults
        rag_defaults = {
            'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
            'INDEX_PATH': './data/index/manual_index',
            'TOP_K': 5,
            'CHUNK_SIZE': 1000,
            'CHUNK_OVERLAP': 200,
        }
        for key, value in rag_defaults.items():
            if key not in self.config['RAG_SETTINGS']:
                self.config['RAG_SETTINGS'][key] = value
        
        # Prompt templates
        self.config['PROMPT_TEMPLATES'].setdefault('SYSTEM_PROMPT', (
            "You are an automotive assistant helping drivers with questions about "
            "their vehicle. Your answers should be concise, accurate, and helpful. "
            "If you don't have enough information to answer, say so clearly rather "
            "than making up details."
        ))
        
        self.config['PROMPT_TEMPLATES'].setdefault('RESPONSE_PROMPT', (
            "Answer the following question based on the context information "
            "provided. If the question cannot be answered using the information "
            "provided, please indicate that you don't have enough information.\n\n"
            "Context: {context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ))
        
        # Logging defaults
        logging_defaults = {
            'LOG_LEVEL': 'INFO',
            'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'LOG_FILE': 'car_assistant.log',
            'LOG_MAX_BYTES': 10485760,  # 10MB
            'LOG_BACKUP_COUNT': 5,
        }
        for key, value in logging_defaults.items():
            if key not in self.config['LOGGING_SETTINGS']:
                self.config['LOGGING_SETTINGS'][key] = value
        
        # Error handling
        error_defaults = {
            'MAX_RETRY_ATTEMPTS': 3,
            'RETRY_DELAY': 1.0,  # seconds
            'DEFAULT_ERROR_MESSAGE': 'Sorry, I encountered an error while processing your request.',
            'NETWORK_ERROR_MESSAGE': "Sorry, I'm having trouble connecting to the network.",
            'API_ERROR_MESSAGE': "Sorry, I'm having trouble accessing the necessary services.",
            'RECOGNITION_ERROR_MESSAGE': "Sorry, I didn't catch what you said.",
        }
        for key, value in error_defaults.items():
            if key not in self.config['ERROR_HANDLING']:
                self.config['ERROR_HANDLING'][key] = value
        
        # Context management
        context_defaults = {
            'MAX_CONVERSATION_TURNS': 5,
            'CONTEXT_WINDOW_SIZE': 10,
            'INCLUDE_TIMESTAMPS': True,
            'STORE_RAW_AUDIO': False,
            'ABBREVIATE_LONG_RESPONSES': True,
        }
        for key, value in context_defaults.items():
            if key not in self.config['CONTEXT_SETTINGS']:
                self.config['CONTEXT_SETTINGS'][key] = value
        
        # App settings
        app_defaults = {
            'DEBUG_MODE': False,
            'TIMEOUT': 30,  # seconds
            'CONTINUOUS_LISTENING': True,
            'RESPONSE_MAX_LENGTH': 250,
            'DATA_DIR': './data',
        }
        for key, value in app_defaults.items():
            if key not in self.config['APP_SETTINGS']:
                self.config['APP_SETTINGS'][key] = value

    def load_from_env(self) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary with settings from environment variables
        """
        config = {
            'API_SETTINGS': {},
            'SPEECH_SETTINGS': {},
            'RAG_SETTINGS': {},
            'PROMPT_TEMPLATES': {},
            'LOGGING_SETTINGS': {},
            'ERROR_HANDLING': {},
            'CONTEXT_SETTINGS': {},
            'APP_SETTINGS': {},
        }
        
        # API settings
        if os.getenv('GROQ_API_KEY'):
            config['API_SETTINGS']['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
        if os.getenv('GROQ_MODEL'):
            config['API_SETTINGS']['GROQ_MODEL'] = os.getenv('GROQ_MODEL')
        if os.getenv('GROQ_API_BASE'):
            config['API_SETTINGS']['GROQ_API_BASE'] = os.getenv('GROQ_API_BASE')
        if os.getenv('MAX_TOKENS'):
            config['API_SETTINGS']['MAX_TOKENS'] = int(os.getenv('MAX_TOKENS'))
        if os.getenv('TEMPERATURE'):
            config['API_SETTINGS']['TEMPERATURE'] = float(os.getenv('TEMPERATURE'))
        
        # Speech settings
        if os.getenv('RECOGNITION_ENGINE'):
            config['SPEECH_SETTINGS']['RECOGNITION_ENGINE'] = os.getenv('RECOGNITION_ENGINE')
        if os.getenv('SYNTHESIS_ENGINE'):
            config['SPEECH_SETTINGS']['SYNTHESIS_ENGINE'] = os.getenv('SYNTHESIS_ENGINE')
        if os.getenv('LANGUAGE'):
            config['SPEECH_SETTINGS']['LANGUAGE'] = os.getenv('LANGUAGE')
        if os.getenv('WAKE_WORD'):
            config['SPEECH_SETTINGS']['WAKE_WORD'] = os.getenv('WAKE_WORD')
        if os.getenv('VOICE_RATE'):
            config['SPEECH_SETTINGS']['VOICE_RATE'] = int(os.getenv('VOICE_RATE'))
        if os.getenv('VOICE_VOLUME'):
            config['SPEECH_SETTINGS']['VOICE_VOLUME'] = float(os.getenv('VOICE_VOLUME'))
        if os.getenv('ENERGY_THRESHOLD'):
            config['SPEECH_SETTINGS']['ENERGY_THRESHOLD'] = int(os.getenv('ENERGY_THRESHOLD'))
        if os.getenv('PAUSE_THRESHOLD'):
            config['SPEECH_SETTINGS']['PAUSE_THRESHOLD'] = float(os.getenv('PAUSE_THRESHOLD'))
        
        # RAG settings
        if os.getenv('EMBEDDING_MODEL'):
            config['RAG_SETTINGS']['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL')
        if os.getenv('INDEX_PATH'):
            config['RAG_SETTINGS']['INDEX_PATH'] = os.getenv('INDEX_PATH')
        if os.getenv('TOP_K'):
            config['RAG_SETTINGS']['TOP_K'] = int(os.getenv('TOP_K'))
        if os.getenv('CHUNK_SIZE'):
            config['RAG_SETTINGS']['CHUNK_SIZE'] = int(os.getenv('CHUNK_SIZE'))
        if os.getenv('CHUNK_OVERLAP'):
            config['RAG_SETTINGS']['CHUNK_OVERLAP'] = int(os.getenv('CHUNK_OVERLAP'))
        
        # Logging settings
        if os.getenv('LOG_LEVEL'):
            config['LOGGING_SETTINGS']['LOG_LEVEL'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            config['LOGGING_SETTINGS']['LOG_FILE'] = os.getenv('LOG_FILE')
        
        # App settings
        if os.getenv('DEBUG_MODE'):
            config['APP_SETTINGS']['DEBUG_MODE'] = os.getenv('DEBUG_MODE').lower() in ('true', 'yes', '1')
        if os.getenv('DATA_DIR'):
            config['APP_SETTINGS']['DATA_DIR'] = os.getenv('DATA_DIR')
            
        return config

    def load_from_file(self, config_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to the JSON configuration file
            
        Returns:
            Configuration dictionary with settings from the file
        """
        # Initialize empty configuration
        config = {}
        
        # Check if file exists
        if not os.path.exists(config_file):
            return config
        
        # Load configuration from file
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logging.warning(f"Error loading configuration file: {str(e)}")
            
        return config

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key within section
            default: Default value if key not found
            
        Returns:
            The configuration value or default if not found
        """
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default

    def get_config_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Configuration section name
            
        Returns:
            Dictionary of all configuration values for the section
        """
        return self.config.get(section, {})

    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            section: Configuration section name
            key: Configuration key to update
            value: New value to set
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value

    def validate_config(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigValidationError: If required configuration is missing
        """
        validation_errors = []
        
        # Check for required configuration keys
        for section, keys in self.REQUIRED_CONFIG.items():
            if section not in self.config:
                validation_errors.append(f"Missing configuration section: {section}")
                continue
                
            for key in keys:
                if key not in self.config[section]:
                    validation_errors.append(f"Missing required configuration: {section}.{key}")
        
        # Check API key specifically (it's especially important)
        if 'API_SETTINGS' in self.config:
            if not self.config['API_SETTINGS'].get('GROQ_API_KEY'):
                validation_errors.append("GROQ_API_KEY is not set. Set it in the .env file or environment variables.")
        
        # Raise exception if there are validation errors
        if validation_errors:
            error_message = "Configuration validation failed:\n" + "\n".join(validation_errors)
            raise ConfigValidationError(error_message)
        
        return True

    def setup_logging(self) -> None:
        """Configure logging based on the logging settings."""
        log_level_str = self.get('LOGGING_SETTINGS', 'LOG_LEVEL', 'INFO')
        log_format = self.get('LOGGING_SETTINGS', 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = self.get('LOGGING_SETTINGS', 'LOG_FILE')
        
        # Convert string log level to logging constant
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),  # Console handler
            ]
        )
        
        # Add file handler if log file is specified
        if log_file:
            log_max_bytes = self.get('LOGGING_SETTINGS', 'LOG_MAX_BYTES', 10485760)  # 10MB
            log_backup_count = self.get('LOGGING_SETTINGS', 'LOG_BACKUP_COUNT', 5)
            
            # Create directory if it doesn't exist
            log_path = Path(log_file)
            if not log_path.parent.exists():
                log_path.parent.mkdir(parents=True)
            
            # Add rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=log_max_bytes,
                backupCount=log_backup_count
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
            
        logging.info("Logging configured successfully")

    def generate_sample_env(self, output_file: str = "example.env") -> None:
        """
        Generate a sample .env file with all configurable options.
        
        Args:
            output_file: Path to the output file
        """
        lines = [
            "# Voice-Powered Car Assistant - Configuration",
            "# Copy this file to .env and fill in your values",
            "",
            "# API Settings",
            "GROQ_API_KEY=your_api_key_here",
            "GROQ_MODEL=llama3-8b-8192",
            "GROQ_API_BASE=https://api.groq.com/openai/v1",
            "MAX_TOKENS=1024",
            "TEMPERATURE=0.7",
            "",
            "# Speech Settings",
            "RECOGNITION_ENGINE=google",
            "SYNTHESIS_ENGINE=pyttsx3",
            "LANGUAGE=en-US",
            "WAKE_WORD=assistant",
            "VOICE_RATE=150",
            "VOICE_VOLUME=1.0",
            "ENERGY_THRESHOLD=300",
            "PAUSE_THRESHOLD=0.5",
            "",
            "# RAG Settings",
            "EMBEDDING_MODEL=all-MiniLM-L6-v2",
            "INDEX_PATH=./data/index/manual_index",
            "TOP_K=5",
            "CHUNK_SIZE=1000",
            "CHUNK_OVERLAP=200",
            "",
            "# Logging Settings",
            "LOG_LEVEL=INFO",
            "LOG_FILE=car_assistant.log",
            "",
            "# App Settings",
            "DEBUG_MODE=False",
            "DATA_DIR=./data",
            "",
        ]
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Sample configuration written to {output_file}")


# Create a default configuration instance
config = ConfigManager()