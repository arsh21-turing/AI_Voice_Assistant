"""
speech/synthesis.py
Text-to-speech synthesis module for the Car Assistant using pyttsx3.
"""

import pyttsx3
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


class VoiceSynthesizer:
    """
    Handles text-to-speech synthesis using the pyttsx3 library.
    Provides methods for converting text to speech with customizable voice properties.
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the VoiceSynthesizer with customizable parameters.

        Args:
            rate: Speech rate (words per minute, default: 175)
            volume: Volume level from 0.0 to 1.0 (default: 1.0)
            voice_id: Specific voice identifier to use. When None, uses system default.
            log_level: Logging level (default: logging.INFO)
        """
        # Set up logger
        self._configure_logger(log_level)
        
        logger.info("Initializing VoiceSynthesizer")
        
        try:
            # Initialize the TTS engine
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Set voice if specified, otherwise use default
            if voice_id:
                self.engine.setProperty('voice', voice_id)
            
            # Store current settings
            self.rate = rate
            self.volume = volume
            self.voice_id = voice_id or self._get_current_voice_id()
            
            # Get available voices for later use
            self.available_voices = self._get_available_voices()
            
            logger.info(f"VoiceSynthesizer initialized with rate={rate}, volume={volume}, voice={self.voice_id}")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing speech synthesis engine: {e}", exc_info=True)
            self.is_initialized = False
            self.last_error = str(e)

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
            
        logger.debug("Logger configured for VoiceSynthesizer")

    def speak(self, text: str, block: bool = True) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: The text to convert to speech
            block: Whether to block until speech is complete (default: True)
            
        Returns:
            bool: True if speech was successful, False if an error occurred
        """
        if not text:
            logger.warning("Empty text provided to speak method")
            return False
            
        if not self.is_initialized:
            logger.error("Speech synthesis engine not properly initialized")
            return False
        
        try:
            logger.info(f"Speaking: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            logger.debug(f"Full text: {text}")
            
            # Convert text to speech
            self.engine.say(text)
            
            if block:
                logger.debug("Running engine until speech completes")
                self.engine.runAndWait()
                logger.debug("Speech completed")
            
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error during speech synthesis: {e}", exc_info=True)
            return False

    def _get_current_voice_id(self) -> str:
        """
        Get the currently set voice ID.
        
        Returns:
            str: Current voice ID
        """
        return self.engine.getProperty('voice')

    def _get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get information about available voices.
        
        Returns:
            List[Dict]: List of voice information dictionaries
        """
        voices = []
        for voice in self.engine.getProperty('voices'):
            voice_info = {
                'id': voice.id,
                'name': voice.name,
                'languages': voice.languages,
                'gender': voice.gender,
                'age': voice.age
            }
            voices.append(voice_info)
            logger.debug(f"Available voice: {voice_info['name']} (ID: {voice_info['id']})")
        
        logger.info(f"Found {len(voices)} available voices")
        return voices

    def list_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available voices.
        
        Returns:
            List[Dict]: List of available voice information
        """
        return self.available_voices

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update voice synthesizer settings.
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            logger.info("Updating VoiceSynthesizer settings")
            
            # Update rate if specified
            if 'rate' in settings:
                rate = settings['rate']
                self.engine.setProperty('rate', rate)
                self.rate = rate
                logger.debug(f"Updated rate to {rate}")
                
            # Update volume if specified
            if 'volume' in settings:
                volume = settings['volume']
                self.engine.setProperty('volume', volume)
                self.volume = volume
                logger.debug(f"Updated volume to {volume}")
                
            # Update voice if specified
            if 'voice_id' in settings:
                voice_id = settings['voice_id']
                self.engine.setProperty('voice', voice_id)
                self.voice_id = voice_id
                logger.debug(f"Updated voice to {voice_id}")
                
            # Update log level if specified
            if 'log_level' in settings:
                self._configure_logger(settings['log_level'])
                logger.debug(f"Updated log_level to {settings['log_level']}")
                
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error updating settings: {e}", exc_info=True)
            return False

    def get_current_settings(self) -> Dict[str, Any]:
        """
        Get current voice settings.
        
        Returns:
            Dict: Dictionary of current settings
        """
        return {
            'rate': self.rate,
            'volume': self.volume,
            'voice_id': self.voice_id
        }

    def get_last_error(self) -> Optional[str]:
        """
        Get the last error message.
        
        Returns:
            str or None: The last error message, or None if no error has occurred
        """
        return getattr(self, 'last_error', None)


if __name__ == "__main__":
    # Set up basic logging configuration for the test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple test
    synthesizer = VoiceSynthesizer(rate=160, volume=0.9)
    
    # List available voices
    voices = synthesizer.list_available_voices()
    logger.info(f"Found {len(voices)} voices")
    
    # Speak a test message
    test_message = "Hello! I am your voice-powered car assistant. How can I help you today?"
    success = synthesizer.speak(test_message)
    
    if success:
        logger.info("Speech synthesis completed successfully")
    else:
        logger.error("Speech synthesis failed")
        
    # Demonstrate changing settings
    synthesizer.update_settings({'rate': 145, 'volume': 1.0})
    synthesizer.speak("I've updated my speaking rate to be slower and volume to be louder.")