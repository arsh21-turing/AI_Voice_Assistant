"""
speech/recognition.py
Voice recognition module for the Car Assistant using SpeechRecognition library.
"""

import speech_recognition as sr
import time
import logging
import os
import subprocess
from typing import Optional, Dict, Any


# Configure logging
logger = logging.getLogger(__name__)

# Suppress ALSA and JACK warnings by redirecting stderr
# This approach prevents the noise warnings from appearing in the terminal
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Also hide pygame message if it's used internally

def setup_audio_environment():
    """
    Set up the audio environment and check for available devices.
    Returns True if setup was successful, False otherwise.
    """
    try:
        # Check if we're on Linux
        if os.name != 'posix':
            return True

        # Check if PulseAudio is running
        try:
            subprocess.run(['pulseaudio', '--check'], capture_output=True)
        except FileNotFoundError:
            logger.warning("PulseAudio not found, trying to start it...")
            subprocess.run(['pulseaudio', '--start'], capture_output=True)

        # Get list of audio devices
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("No audio input devices found")
            return False

        # Set ALSA environment variables
        os.environ['ALSA_CARD'] = 'Generic'
        os.environ['PULSE_LATENCY_MSEC'] = '60'
        
        return True
    except Exception as e:
        logger.error(f"Error setting up audio environment: {e}")
        return False

# Suppress ALSA warnings - must be done before importing speech_recognition
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Filter out ALSA and JACK warnings that go to stderr
    import platform
    if platform.system() == "Linux":
        try:
            # Redirect stderr to /dev/null for ALSA warnings
            import sys
            stderr_fd = sys.stderr.fileno()
            # Save the original stderr fd
            original_stderr_fd = os.dup(stderr_fd)
            # Open devnull for writing
            devnull = os.open(os.devnull, os.O_WRONLY)
            # Replace stderr with devnull
            os.dup2(devnull, stderr_fd)
            # Import library that produces the warnings
            import pyaudio
            # Close devnull
            os.close(devnull)
            # Restore original stderr
            os.dup2(original_stderr_fd, stderr_fd)
            os.close(original_stderr_fd)
        except Exception as e:
            warnings.warn(f"Failed to suppress ALSA warnings: {e}")


class VoiceRecognizer:
    """
    Handles voice recognition using the SpeechRecognition library.
    Provides methods for initializing microphone, adjusting for ambient noise,
    and listening for voice commands.
    """

    def __init__(
        self,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        dynamic_energy_threshold: bool = True,
        adjust_for_ambient_noise: bool = True,
        timeout: int = 5,
        phrase_time_limit: int = 10,
        log_level: int = logging.INFO
    ):
        """
        Initialize the VoiceRecognizer with customizable parameters.

        Args:
            energy_threshold: Minimum audio energy to detect (default: 300)
            pause_threshold: Seconds of non-speaking before a phrase is considered complete (default: 0.8)
            dynamic_energy_threshold: Automatically adjust energy threshold based on ambient noise (default: True)
            adjust_for_ambient_noise: Whether to adjust for ambient noise on startup (default: True)
            timeout: How long to wait for speech before timing out (default: 5 seconds)
            phrase_time_limit: Maximum length of a phrase (default: 10 seconds)
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logging if not already configured
        if not logger.handlers:
            self._setup_logging(log_level)
        
        logger.info("Initializing VoiceRecognizer with parameters: energy_threshold=%d, pause_threshold=%.2f, "
                   "dynamic_energy_threshold=%s, timeout=%d, phrase_time_limit=%d", 
                   energy_threshold, pause_threshold, dynamic_energy_threshold, timeout, phrase_time_limit)
        
        # Set up audio environment
        if not setup_audio_environment():
            logger.error("Failed to set up audio environment")
            self.is_initialized = False
            self.last_error = "Audio environment setup failed"
            return
        
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Configure recognizer
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        
        # Store other parameters
        self.do_ambient_adjust = adjust_for_ambient_noise
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        
        # Track initialization status
        self.is_initialized = False
        self.last_error = None

    def _setup_logging(self, level=logging.INFO):
        """
        Set up logging configuration.
        
        Args:
            level: logging level (default: logging.INFO)
        """
        # Create handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs

    def initialize_microphone(self) -> bool:
        """
        Initialize the microphone source.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing microphone")
            
            # Try to get list of available microphones
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                info = p.get_host_api_info_by_index(0)
                numdevices = info.get('deviceCount')
                if numdevices == 0:
                    logger.error("No audio input devices found")
                    return False
                p.terminate()
            except Exception as e:
                logger.warning(f"Could not check audio devices: {e}")
            
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise if enabled
            if self.do_ambient_adjust:
                logger.info("Adjusting for ambient noise (duration: 2s)")
                try:
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=2)
                    logger.info("Ambient noise adjustment complete. Energy threshold: %d", 
                               self.recognizer.energy_threshold)
                except Exception as e:
                    logger.warning(f"Ambient noise adjustment failed: {e}")
                    # Continue even if ambient noise adjustment fails
            
            self.is_initialized = True
            logger.info("Microphone initialization successful")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error("Error initializing microphone: %s", str(e), exc_info=True)
            self.is_initialized = False
            return False

    def adjust_for_ambient_noise(self, duration: int = 2) -> None:
        """
        Manually adjust for ambient noise.
        
        Args:
            duration: Duration in seconds to sample ambient noise (default: 2)
        """
        if not self.is_initialized:
            logger.info("Microphone not initialized, initializing now")
            self.initialize_microphone()
            
        try:
            logger.info("Manually adjusting for ambient noise (duration: %ds)", duration)
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            logger.info("Manual ambient noise adjustment complete. Energy threshold: %d", 
                       self.recognizer.energy_threshold)
        except Exception as e:
            self.last_error = str(e)
            logger.error("Error adjusting for ambient noise: %s", str(e), exc_info=True)

    def listen(
        self, 
        timeout: Optional[int] = None, 
        phrase_time_limit: Optional[int] = None,
        show_listening_indicator: bool = True
    ) -> Optional[str]:
        """
        Listen for voice input and convert to text.
        
        Args:
            timeout: How long to wait for speech before timing out (overrides instance default if provided)
            phrase_time_limit: Maximum length of a phrase (overrides instance default if provided)
            show_listening_indicator: Whether to log a listening indicator
            
        Returns:
            str or None: Recognized text, or None if no speech was detected or an error occurred
        """
        if not self.is_initialized:
            logger.info("Microphone not initialized, initializing now before listening")
            success = self.initialize_microphone()
            if not success:
                logger.error("Failed to initialize microphone, can't listen")
                return None
        
        # Use provided parameters or fall back to instance defaults
        timeout = timeout if timeout is not None else self.timeout
        phrase_time_limit = phrase_time_limit if phrase_time_limit is not None else self.phrase_time_limit
        
        try:
            if show_listening_indicator:
                logger.info("Listening for speech (timeout: %ds, phrase_time_limit: %ds)", 
                           timeout, phrase_time_limit)
                
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            logger.debug("Audio captured, processing speech...")
                
            # Convert speech to text
            text = self.recognizer.recognize_google(audio)
            logger.info("Speech recognized: '%s'", text)
            return text
            
        except sr.WaitTimeoutError:
            # No speech detected within timeout
            logger.info("Listening timed out after %ds. No speech detected.", timeout)
            return None
            
        except sr.UnknownValueError:
            # Speech was unintelligible
            logger.warning("Speech detected but could not be understood")
            return None
            
        except Exception as e:
            self.last_error = str(e)
            logger.error("Error in speech recognition: %s", str(e), exc_info=True)
            return None

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update recognizer settings with a dictionary of parameters.
        
        Args:
            settings: Dictionary of settings to update
        """
        logger.info("Updating voice recognizer settings: %s", str(settings))
        
        # Update recognizer properties
        if 'energy_threshold' in settings:
            self.recognizer.energy_threshold = settings['energy_threshold']
        if 'pause_threshold' in settings:
            self.recognizer.pause_threshold = settings['pause_threshold']
        if 'dynamic_energy_threshold' in settings:
            self.recognizer.dynamic_energy_threshold = settings['dynamic_energy_threshold']
            
        # Update instance properties
        if 'timeout' in settings:
            self.timeout = settings['timeout']
        if 'phrase_time_limit' in settings:
            self.phrase_time_limit = settings['phrase_time_limit']
        if 'adjust_for_ambient_noise' in settings:
            self.do_ambient_adjust = settings['adjust_for_ambient_noise']
        
        # Update log level if provided
        if 'log_level' in settings:
            logger.setLevel(settings['log_level'])
            logger.info("Log level updated to %d", settings['log_level'])
            
        # Re-initialize if needed
        if settings.get('reinitialize', False):
            logger.info("Re-initializing microphone due to setting update")
            self.initialize_microphone()

    def get_last_error(self) -> Optional[str]:
        """
        Get the last error message.
        
        Returns:
            str or None: The last error message, or None if no error has occurred
        """
        return self.last_error


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simple test
    recognizer = VoiceRecognizer(adjust_for_ambient_noise=True)
    recognizer.initialize_microphone()
    
    logger.info("Ready for voice input. Say something!")
    text = recognizer.listen(show_listening_indicator=True)
    
    if text:
        logger.info("You said: %s", text)
    else:
        logger.warning("No speech detected or an error occurred.")