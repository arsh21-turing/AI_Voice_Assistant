#!/usr/bin/env python3
# main.py

"""
Entry point for the Voice-Powered Car Assistant.

This script initializes all components and runs the main voice interaction loop.
"""

import os
import time
import json
import logging
from pathlib import Path
import importlib.util

# Set up logging first
def setup_logging():
    """Configure logging for the application."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"car_assistant_{timestamp}.log"
    
    # Set up file and console logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Import components with direct imports
from speech.recognition import VoiceRecognizer
from speech.synthesis import VoiceSynthesizer
from core.voice_engine import VoiceEngine
from core.assistant_manager import AssistantManager
from rag import QueryProcessor, ContextRetriever, ResponseGenerator, ManualRetriever
from config.settings import ConfigManager
from utils.error_handler import ErrorHandler

def setup_environment():
    """Set up the required environment for the application."""
    logger.info("Setting up environment")
    
    # Create required directories
    directories = [
        'data/vector_store',       # For FAISS index and metadata
        'data/index/manual_index', # For manual index
        'data/manuals',            # For PDF manuals
        'data/terminology',        # For terminology files
        'data/indices',            # For additional indices
        'logs'                     # For log files
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.info(f"Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.debug(f"Directory already exists: {directory}")
    
    # Initialize vector store if empty
    vector_store_path = Path('data/vector_store')
    index_path = vector_store_path / 'index.faiss'
    metadata_path = vector_store_path / 'metadata.json'
    
    if not index_path.exists():
        logger.info("Vector store not found, creating empty FAISS index")
        try:
            # Check if faiss is available
            faiss_available = importlib.util.find_spec("faiss") is not None
            
            if faiss_available:
                # Create empty FAISS index
                import faiss
                dimension = 384  # Dimension for all-MiniLM-L6-v2
                index = faiss.IndexFlatL2(dimension)
                faiss.write_index(index, str(index_path))
                logger.info(f"Created empty FAISS index at {index_path}")
                
                # Create empty metadata file
                with open(metadata_path, 'w') as f:
                    json.dump([], f)
                logger.info(f"Created empty metadata file at {metadata_path}")
            else:
                logger.warning("FAISS not available, skipping index creation")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}", exc_info=True)
    else:
        logger.info(f"Found existing vector store at {vector_store_path}")
    
    logger.info("Environment setup complete")

def check_configuration(config_manager):
    """
    Check if the configuration has all required settings.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    logger.info("Checking configuration")
    
    try:
        # Check for required API keys
        api_key = config_manager.get('API_SETTINGS', 'GROQ_API_KEY')
        
        if not api_key or api_key in ['your_groq_api_key_here', 'your-api-key']:
            logger.warning("GROQ_API_KEY not set. Please set it in .env or config.json")
            return False
        
        # Log successful configuration (safely)
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        logger.info(f"Configuration loaded successfully. API Key: {masked_key}")
        
        # Check for other critical config
        embedding_model = config_manager.get('RAG_SETTINGS', 'EMBEDDING_MODEL')
        logger.info(f"Using embedding model: {embedding_model}")
        
        index_path = config_manager.get('RAG_SETTINGS', 'INDEX_PATH')
        logger.info(f"Vector store index path: {index_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}", exc_info=True)
        return False

def initialize_components(config_manager):
    """
    Initialize all components needed for the voice assistant.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        tuple: (recognizer, synthesizer, engine, error_handler) or None on failure
    """
    logger.info("Initializing components")
    
    try:
        # Initialize error handler first for other components to use
        error_handler = ErrorHandler(config_manager)
        logger.info("Initialized error handler")
        
        # Initialize speech components with config values
        recognizer = VoiceRecognizer(
            energy_threshold=config_manager.get('SPEECH_SETTINGS', 'ENERGY_THRESHOLD', 300),
            pause_threshold=config_manager.get('SPEECH_SETTINGS', 'PAUSE_THRESHOLD', 0.8),
            dynamic_energy_threshold=config_manager.get('SPEECH_SETTINGS', 'DYNAMIC_ENERGY_THRESHOLD', True)
        )
        logger.info("Initialized voice recognizer")
        
        synthesizer = VoiceSynthesizer()
        logger.info("Initialized voice synthesizer")
        
        # Initialize assistant manager
        assistant = AssistantManager()
        logger.info("Initialized assistant manager")
        
        # Initialize RAG components
        retriever = ManualRetriever(
            config_manager=config_manager,
            error_handler=error_handler
        )
        logger.info("Initialized manual retriever")
        
        # Initialize voice engine with all components
        engine = VoiceEngine(
            recognizer=recognizer,
            synthesizer=synthesizer,
            retriever=retriever,
            assistant=assistant
        )
        logger.info("Initialized voice engine")
        
        return recognizer, synthesizer, engine, error_handler
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
        return None

def run_voice_interaction(recognizer, synthesizer, engine):
    """
    Run the main voice interaction loop.
    
    Args:
        recognizer: Voice recognizer instance
        synthesizer: Voice synthesizer instance
        engine: Voice engine instance
    """
    logger.info("Starting voice interaction loop")
    
    try:
        # Announce system ready
        startup_message = "Voice-Powered Car Assistant is ready. Say 'Hello Car Assistant' to start."
        logger.info(startup_message)
        synthesizer.speak("Car Assistant is ready. How can I help you with your vehicle?")
        
        # Main interaction loop
        continue_processing = True
        while continue_processing:
            logger.debug("Waiting for voice input...")
            
            # Process voice input (this incorporates the wake word detection)
            response, continue_processing = engine.process_voice_input()
            
            # Small pause to prevent CPU overload
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping voice interaction")
        synthesizer.speak("Shutting down. Goodbye!")
    except Exception as e:
        logger.error(f"Error in voice interaction loop: {str(e)}", exc_info=True)
        synthesizer.speak("I've encountered a problem and need to shut down. Please restart me.")
    finally:
        # Cleanup
        if hasattr(synthesizer, 'cleanup'):
            logger.debug("Cleaning up synthesizer resources")
            synthesizer.cleanup()
            
        if hasattr(recognizer, 'cleanup'):
            logger.debug("Cleaning up recognizer resources")
            recognizer.cleanup()
            
        logger.info("Voice interaction loop ended")

def main():
    """Main entry point for the Voice-Powered Car Assistant."""
    logger.info("Voice-Powered Car Assistant starting")
    
    try:
        # Initialize configuration at the start
        config_manager = ConfigManager()
        logger.info("Configuration manager initialized")
        
        # Set up environment
        setup_environment()
        
        # Check configuration
        if not check_configuration(config_manager):
            logger.error("Invalid configuration. Please check settings.")
            return 1
        
        # Initialize components
        components = initialize_components(config_manager)
        if components is None:
            logger.error("Failed to initialize components")
            return 1
            
        recognizer, synthesizer, engine, error_handler = components
        
        # Run main voice interaction loop
        run_voice_interaction(recognizer, synthesizer, engine)
        
    except Exception as e:
        logger.critical(f"Critical error in application: {str(e)}", exc_info=True)
        return 1
    finally:
        logger.info("Voice-Powered Car Assistant shutting down")
    
    return 0

# Create module logger
logger = setup_logging()

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)