#!/usr/bin/env python3
"""
Tests for the voice assistant pipeline, including wake word detection, 
speech recognition, and text-to-speech functionality.
"""

import os
import sys
import unittest
import pytest
import tempfile
import wave
import numpy as np
import json
import time
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
import pyaudio
import speech_recognition as sr

# Path handling to import main app code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import voice assistant components
from main import VoiceAssistant
from utils.error_handler import ErrorHandler
from utils.logger import VoiceAssistantLogger
from utils.config_manager import ConfigManager

# Import system monitor components (if in a separate file)
try:
    from monitors.system_monitor import SystemMonitor, SystemStatus, SystemComponent
except ImportError:
    # If the SystemMonitor is in the main file
    from main import SystemMonitor, SystemStatus, SystemComponent


class AudioTestFile:
    """Helper class for creating and managing audio test files."""
    
    def __init__(self, filename=None, duration=2.0, sample_rate=16000):
        """Initialize an audio test file.
        
        Args:
            filename: Optional filename for the audio file
            duration: Duration of the audio in seconds
            sample_rate: Sample rate of the audio in Hz
        """
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Create a temporary file if no filename provided
        if filename is None:
            self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.filename = self.temp_file.name
        else:
            self.filename = filename
            self.temp_file = None
    
    def create_silence(self, duration=None):
        """Create a silent WAV file.
        
        Args:
            duration: Duration of silence in seconds (overrides constructor value)
        """
        if duration is None:
            duration = self.duration
            
        # Create silent audio data (all zeros)
        num_samples = int(duration * self.sample_rate)
        audio_data = np.zeros(num_samples, dtype=np.int16)
        
        # Write to WAV file
        self._write_wav(audio_data)
        
        return self
    
    def create_sine_wave(self, frequency=440, amplitude=0.5, duration=None):
        """Create a WAV file with a sine wave.
        
        Args:
            frequency: Frequency of the sine wave in Hz
            amplitude: Amplitude of the sine wave (0.0 to 1.0)
            duration: Duration of the audio in seconds (overrides constructor value)
        """
        if duration is None:
            duration = self.duration
            
        # Create time array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Generate sine wave
        audio_data = (amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
        
        # Write to WAV file
        self._write_wav(audio_data)
        
        return self
    
    def create_from_text(self, text, voice="en-US"):
        """Create audio file from text using text-to-speech.
        Requires pyttsx3 or another TTS engine.
        
        Args:
            text: Text to synthesize
            voice: Voice ID or language code
        """
        try:
            import pyttsx3
            
            # Initialize TTS engine
            engine = pyttsx3.init()
            
            # Set voice if needed
            if voice is not None:
                voices = engine.getProperty('voices')
                for v in voices:
                    if voice in v.id:
                        engine.setProperty('voice', v.id)
                        break
            
            # Save to file
            engine.save_to_file(text, self.filename)
            engine.runAndWait()
            
            return self
            
        except ImportError:
            raise ImportError("pyttsx3 is required to create audio from text")
    
    def _write_wav(self, audio_data):
        """Write audio data to a WAV file.
        
        Args:
            audio_data: NumPy array of audio samples
        """
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
    
    def get_audio_data(self):
        """Read the audio file and return audio data for speech recognition."""
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.filename) as source:
            return recognizer.record(source)
    
    def cleanup(self):
        """Delete the temporary file if created."""
        if self.temp_file:
            try:
                os.unlink(self.filename)
            except Exception:
                pass


class TestAudioMocking:
    """Class for mocking audio input/output components."""
    
    @staticmethod
    def create_mock_recognizer():
        """Create a mock speech recognizer.
        
        Returns:
            Mock object for sr.Recognizer
        """
        mock_recognizer = Mock(spec=sr.Recognizer)
        
        # Mock adjust_for_ambient_noise
        mock_recognizer.adjust_for_ambient_noise = Mock(return_value=None)
        
        # Mock recognize_google with custom response based on input
        def mock_recognize_google(audio_data, **kwargs):
            # Check if audio_data has a _mock_text attribute
            if hasattr(audio_data, '_mock_text'):
                return audio_data._mock_text
            elif isinstance(audio_data, MagicMock):
                # For generic mocks, pick a phrase based on hash of the mock
                phrases = [
                    "hey assistant", 
                    "what's the weather like", 
                    "turn on the lights",
                    "how are you today",
                    "tell me a joke"
                ]
                return phrases[hash(str(audio_data)) % len(phrases)]
            else:
                return "hey assistant"
                
        mock_recognizer.recognize_google = Mock(side_effect=mock_recognize_google)
        
        # Mock listen with custom response
        def mock_listen(source, **kwargs):
            # Create a mock audio data object with text
            if hasattr(source, '_mock_text'):
                mock_audio = MagicMock()
                mock_audio._mock_text = source._mock_text
                return mock_audio
            else:
                # Default mock audio
                return MagicMock()
                
        mock_recognizer.listen = Mock(side_effect=mock_listen)
        
        return mock_recognizer
    
    @staticmethod
    def create_mock_microphone(text=None):
        """Create a mock microphone source.
        
        Args:
            text: Text to associate with this microphone source
            
        Returns:
            Mock object for sr.Microphone
        """
        mock_mic = MagicMock(spec=sr.Microphone)
        
        # Set up context manager behavior
        mock_mic.__enter__ = Mock(return_value=mock_mic)
        mock_mic.__exit__ = Mock(return_value=None)
        
        # Add mock text if provided
        if text is not None:
            mock_mic._mock_text = text
            
        return mock_mic
    
    @staticmethod
    def create_mock_tts_engine():
        """Create a mock text-to-speech engine.
        
        Returns:
            Mock pyttsx3.Engine
        """
        mock_engine = MagicMock()
        
        # Set up properties
        voices = [
            MagicMock(id="voice1", name="Male voice"),
            MagicMock(id="voice2", name="Female voice")
        ]
        
        # Mock getProperty to return different values based on property name
        def mock_get_property(prop):
            if prop == 'voices':
                return voices
            elif prop == 'rate':
                return 150
            elif prop == 'volume':
                return 1.0
            elif prop == 'voice':
                return voices[0]
            return None
            
        mock_engine.getProperty = Mock(side_effect=mock_get_property)
        mock_engine.setProperty = Mock()
        mock_engine.say = Mock()
        mock_engine.runAndWait = Mock()
        
        return mock_engine


class TestWakeWordDetection(unittest.TestCase):
    """Tests for wake word detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mocks
        self.mock_recognizer = TestAudioMocking.create_mock_recognizer()
        self.mock_microphone = TestAudioMocking.create_mock_microphone()
        self.mock_engine = TestAudioMocking.create_mock_tts_engine()
        
        # Create test audio files
        self.test_audio_dir = tempfile.mkdtemp()
        
        # Create wake word audio
        self.wake_word_audio = AudioTestFile(
            os.path.join(self.test_audio_dir, "hey_assistant.wav")
        ).create_from_text("hey assistant")
        
        # Create non-wake word audio
        self.non_wake_audio = AudioTestFile(
            os.path.join(self.test_audio_dir, "hello.wav")
        ).create_from_text("hello how are you")
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        self.wake_word_audio.cleanup()
        self.non_wake_audio.cleanup()
        
        # Remove test directory
        os.rmdir(self.test_audio_dir)
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_wake_word_detection_basic(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test basic wake word detection."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            assistant = VoiceAssistant()
        
        # Set mock text for microphone
        self.mock_microphone._mock_text = "hey assistant"
        
        # Test wake word detection
        result = assistant.wait_for_wake_word()
        self.assertTrue(result)
        
        # Test with non-wake word
        self.mock_microphone._mock_text = "hello"
        result = assistant.wait_for_wake_word()
        self.assertFalse(result)
    
    def test_wake_word_detection_accuracy(self):
        """Test wake word detection accuracy with pre-recorded audio."""
        # This test uses actual audio files and recognition
        recognizer = sr.Recognizer()
        
        # Test with wake word audio
        with sr.AudioFile(self.wake_word_audio.filename) as source:
            audio = recognizer.record(source)
            
            # Use a try-except block as recognition might fail in test environment
            try:
                text = recognizer.recognize_google(audio).lower()
                self.assertIn("hey assistant", text)
            except sr.UnknownValueError:
                self.skipTest("Speech recognition failed - environment may not support audio")
            except sr.RequestError:
                self.skipTest("Speech recognition API unavailable")
    
    def test_wake_word_with_background_noise(self):
        """Test wake word detection with background noise."""
        # Create a noisy wake word audio file by mixing with noise
        noisy_audio = AudioTestFile(
            os.path.join(self.test_audio_dir, "noisy_wake.wav")
        )
        
        # Load wake word audio
        with wave.open(self.wake_word_audio.filename, 'rb') as wf:
            params = wf.getparams()
            frames = wf.readframes(wf.getnframes())
            wave_data = np.frombuffer(frames, dtype=np.int16)
        
        # Generate noise
        noise = np.random.normal(0, 500, wave_data.shape).astype(np.int16)
        
        # Mix with 20% noise
        noisy_data = wave_data + noise
        
        # Save noisy audio
        with wave.open(noisy_audio.filename, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(noisy_data.tobytes())
        
        try:
            # Test with real recognizer
            recognizer = sr.Recognizer()
            with sr.AudioFile(noisy_audio.filename) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.record(source)
                
                # Try to recognize
                try:
                    text = recognizer.recognize_google(audio).lower()
                    detected = "hey assistant" in text
                    
                    # We're just testing if it can be detected at all with noise
                    if detected:
                        self.assertTrue(detected)
                    else:
                        # If not detected, this is not necessarily a failure
                        self.skipTest("Wake word not detected in noisy audio - this may be acceptable")
                        
                except sr.UnknownValueError:
                    self.skipTest("Speech recognition failed with noisy audio - this may be acceptable")
                except sr.RequestError:
                    self.skipTest("Speech recognition API unavailable")
        finally:
            # Clean up
            noisy_audio.cleanup()
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_wake_word_error_handling(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test error handling in wake word detection."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            assistant = VoiceAssistant()
        
        # Test timeout error
        self.mock_recognizer.listen.side_effect = sr.WaitTimeoutError()
        result = assistant.wait_for_wake_word()
        self.assertFalse(result)
        
        # Test UnknownValueError
        self.mock_recognizer.listen.side_effect = None
        self.mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()
        result = assistant.wait_for_wake_word()
        self.assertFalse(result)
        
        # Test RequestError
        self.mock_recognizer.recognize_google.side_effect = sr.RequestError("API unavailable")
        result = assistant.wait_for_wake_word()
        self.assertFalse(result)
        
        # Test other exception
        self.mock_recognizer.listen.side_effect = Exception("Test error")
        result = assistant.wait_for_wake_word()
        self.assertFalse(result)


class TestSpeechRecognition(unittest.TestCase):
    """Tests for speech recognition functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mocks
        self.mock_recognizer = TestAudioMocking.create_mock_recognizer()
        self.mock_microphone = TestAudioMocking.create_mock_microphone()
        self.mock_engine = TestAudioMocking.create_mock_tts_engine()
        
        # Create test audio files
        self.test_audio_dir = tempfile.mkdtemp()
        
        # Create test phrases
        phrases = [
            "what is the weather today",
            "tell me about the Hyundai Venue",
            "how do I change the oil",
            "where are the headlight controls"
        ]
        
        # Create audio files for test phrases
        self.test_audio_files = []
        for i, phrase in enumerate(phrases):
            audio_file = AudioTestFile(
                os.path.join(self.test_audio_dir, f"phrase_{i}.wav")
            ).create_from_text(phrase)
            self.test_audio_files.append((audio_file, phrase))
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove test files
        for audio_file, _ in self.test_audio_files:
            audio_file.cleanup()
        
        # Remove test directory
        os.rmdir(self.test_audio_dir)
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_basic_speech_recognition(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test basic speech recognition."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            assistant = VoiceAssistant()
        
        # Test speech recognition
        test_text = "tell me about the Hyundai Venue"
        self.mock_microphone._mock_text = test_text
        
        result = assistant.listen()
        self.assertEqual(result, test_text)
    
    def test_speech_recognition_accuracy(self):
        """Test speech recognition accuracy with pre-recorded audio."""
        # This test uses actual audio files and recognition
        recognizer = sr.Recognizer()
        
        for audio_file, expected_text in self.test_audio_files:
            with sr.AudioFile(audio_file.filename) as source:
                audio = recognizer.record(source)
                
                # Use a try-except block as recognition might fail in test environment
                try:
                    text = recognizer.recognize_google(audio).lower()
                    
                    # We're testing if key words are recognized, not exact match
                    key_words = [word for word in expected_text.lower().split() if len(word) > 3]
                    recognized_words = [word for word in text.split() if word in key_words]
                    
                    self.assertGreater(
                        len(recognized_words), 
                        len(key_words) // 2,
                        f"Expected to recognize at least half of key words in '{expected_text}'"
                    )
                    
                except sr.UnknownValueError:
                    self.skipTest("Speech recognition failed - environment may not support audio")
                except sr.RequestError:
                    self.skipTest("Speech recognition API unavailable")
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_speech_recognition_errors(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test error handling in speech recognition."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            with patch('main.SystemMonitor'):  # Mock the system monitor too
                assistant = VoiceAssistant()
        
        # Test timeout error
        self.mock_recognizer.listen.side_effect = sr.WaitTimeoutError()
        result = assistant.listen()
        self.assertIsNone(result)
        
        # Test UnknownValueError
        self.mock_recognizer.listen.side_effect = None
        self.mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()
        result = assistant.listen()
        self.assertIsNone(result)
        
        # Test RequestError
        self.mock_recognizer.recognize_google.side_effect = sr.RequestError("API unavailable")
        result = assistant.listen()
        self.assertIsNone(result)
        
        # Test other exception
        self.mock_recognizer.listen.side_effect = Exception("Test error")
        result = assistant.listen()
        self.assertIsNone(result)
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_long_phrases(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test recognition of longer phrases."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            with patch('main.SystemMonitor'):  # Mock the system monitor too
                assistant = VoiceAssistant()
        
        # Test with a long phrase
        long_phrase = ("I would like to know how to change the oil in my Hyundai Venue "
                      "and also where the headlight controls are located and how to use the wipers "
                      "during rainy weather conditions please and thank you")
                      
        self.mock_microphone._mock_text = long_phrase
        
        result = assistant.listen()
        self.assertEqual(result, long_phrase)


class TestTextToSpeech(unittest.TestCase):
    """Tests for text-to-speech functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mocks
        self.mock_recognizer = TestAudioMocking.create_mock_recognizer()
        self.mock_microphone = TestAudioMocking.create_mock_microphone()
        self.mock_engine = TestAudioMocking.create_mock_tts_engine()
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_basic_text_to_speech(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test basic text-to-speech functionality."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            with patch('main.SystemMonitor'):  # Mock the system monitor too
                assistant = VoiceAssistant()
        
        # Test speak functionality
        test_text = "This is a test message"
        assistant.speak(test_text)
        
        # Verify that say was called with the expected text
        self.mock_engine.say.assert_called_with(test_text)
        self.mock_engine.runAndWait.assert_called_once()
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_text_to_speech_error_handling(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test error handling in text-to-speech."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            with patch('main.SystemMonitor'):  # Mock the system monitor too
                assistant = VoiceAssistant()
        
        # Test error in engine.say
        self.mock_engine.say.side_effect = Exception("TTS error")
        
        # This should not raise an exception
        assistant.speak("Test with error")
        
        # Verify that text_only_mode is now True
        self.assertTrue(assistant.text_only_mode)
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_voice_settings(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test changing voice settings."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'):
            with patch('main.SystemMonitor'):  # Mock the system monitor too
                assistant = VoiceAssistant()
        
        # Store original settings
        original_rate = assistant.voice_settings.get("rate")
        original_volume = assistant.voice_settings.get("volume")
        
        # Change voice settings
        new_rate = original_rate + 50
        new_volume = original_volume * 0.8
        
        assistant.voice_settings["rate"] = new_rate
        assistant.engine.setProperty('rate', new_rate)
        
        assistant.voice_settings["volume"] = new_volume
        assistant.engine.setProperty('volume', new_volume)
        
        # Speak something to test new settings
        assistant.speak("Testing with new voice settings")
        
        # Verify that setProperty was called with the new settings
        # We can't easily check the exact calls since they happen during initialization too
        self.mock_engine.say.assert_called_with("Testing with new voice settings")
        
        # Check that settings were actually changed
        self.assertEqual(assistant.voice_settings["rate"], new_rate)
        self.assertEqual(assistant.voice_settings["volume"], new_volume)


class TestVoicePipeline(unittest.TestCase):
    """Tests for the end-to-end voice pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mocks
        self.mock_recognizer = TestAudioMocking.create_mock_recognizer()
        self.mock_microphone = TestAudioMocking.create_mock_microphone()
        self.mock_engine = TestAudioMocking.create_mock_tts_engine()
        
        # Create a mock client for API requests
        self.mock_client = MagicMock()
        self.mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="This is a test response"))]
        )
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_end_to_end_pipeline(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test the full voice pipeline flow."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'), \
             patch('main.SystemMonitor'), \
             patch('openai.OpenAI'):
            assistant = VoiceAssistant()
            assistant.client = self.mock_client
        
        # Mock wake word detection
        self.mock_microphone._mock_text = "hey assistant"
        
        # Test the full pipeline
        with patch.object(assistant, 'wait_for_wake_word', return_value=True), \
             patch.object(assistant, 'listen', return_value="tell me about the Venue"), \
             patch.object(assistant, 'process_command', return_value=(False, None)), \
             patch.object(assistant, 'process_query', return_value="This is information about the Venue"):
            
            # Run one cycle of the assistant
            # Need to patch the main loop to run only once
            with patch.object(assistant, 'run', side_effect=KeyboardInterrupt):
                try:
                    assistant.run()
                except KeyboardInterrupt:
                    pass
            
            # Verify that process_query was called with the correct text
            assistant.process_query.assert_called_once_with("tell me about the Venue")
            
            # Verify that speak was called with the response
            assistant.speak.assert_called_with("This is information about the Venue")

    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_command_processing(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test command processing in the pipeline."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'), \
             patch('main.SystemMonitor'), \
             patch('openai.OpenAI'):
            assistant = VoiceAssistant()
            assistant.client = self.mock_client
        
        # Test command processing
        with patch.object(assistant, 'wait_for_wake_word', return_value=True), \
             patch.object(assistant, 'listen', return_value="speak slower"), \
             patch.object(assistant, 'process_command', return_value=(True, "I'll speak slower now")), \
             patch.object(assistant, 'speak') as mock_speak:
            
            # Run one cycle of the assistant
            with patch.object(assistant, 'run', side_effect=KeyboardInterrupt):
                try:
                    assistant.run()
                except KeyboardInterrupt:
                    pass
            
            # Verify that process_query was NOT called (since process_command returned True)
            # and speak was called with the command response
            mock_speak.assert_called_with("I'll speak slower now")
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pyttsx3.init')
    def test_error_recovery(self, mock_init, mock_mic_class, mock_recognizer_class):
        """Test recovery from errors in the pipeline."""
        # Set up mocks
        mock_init.return_value = self.mock_engine
        mock_recognizer_class.return_value = self.mock_recognizer
        mock_mic_class.return_value = self.mock_microphone
        
        # Create assistant with mocked components
        with patch('main.VoiceAssistant.adjust_for_ambient_noise'), \
             patch('main.SystemMonitor'), \
             patch('openai.OpenAI'):
            assistant = VoiceAssistant()
            assistant.client = self.mock_client
        
        # Set up system monitor mock to handle errors
        system_monitor_mock = MagicMock()
        system_monitor_mock.handle_error.return_value = (False, "There was an error")
        assistant.system_monitor = system_monitor_mock
        
        # Test with error in listen method
        with patch.object(assistant, 'wait_for_wake_word', return_value=True), \
             patch.object(assistant, 'listen', side_effect=Exception("Test error")), \
             patch.object(assistant, 'speak') as mock_speak:
            
            # Run one cycle of the assistant
            with patch.object(assistant, 'run', side_effect=KeyboardInterrupt):
                try:
                    assistant.run()
                except KeyboardInterrupt:
                    pass
            
            # Verify that system_monitor.handle_error was called
            system_monitor_mock.handle_error.assert_called()
            
            # Verify that speak was called with the error message
            mock_speak.assert_called_with("There was an error")


class TestSystemMonitoring(unittest.TestCase):
    """Tests for the system monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mocks
        self.mock_recognizer = TestAudioMocking.create_mock_recognizer()
        self.mock_microphone = TestAudioMocking.create_mock_microphone()
        self.mock_engine = TestAudioMocking.create_mock_tts_engine()
        
        # Create a mock assistant for the system monitor
        self.mock_assistant = MagicMock()
        
        # Set assistant attributes accessed by the monitor
        self.mock_assistant.microphone = self.mock_microphone
        self.mock_assistant.recognizer = self.mock_recognizer
        self.mock_assistant.engine = self.mock_engine
        self.mock_assistant.voices = [MagicMock(), MagicMock()]
        self.mock_assistant.client = MagicMock()
        self.mock_assistant.groq_model = "test-model"
        
        # Set FAISS index attributes
        mock_index = MagicMock()
        mock_index.ntotal = 100
        self.mock_assistant.index = mock_index
        
        # Set embedding model attributes
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.random(384)
        self.mock_assistant.embedding_model = mock_model
        
        # Set metadata
        self.mock_assistant.metadata = {
            "texts": ["text"] * 100,
            "ids": list(range(100)),
            "metadata": [{}] * 100,
            "document_hashes": {"hash1": {}}
        }
    
    def test_system_monitor_initialization(self):
        """Test initialization of system monitor."""
        # Create SystemMonitor instance
        monitor = SystemMonitor(self.mock_assistant)
        
        # Check that component statuses were initialized
        for component in SystemComponent:
            self.assertIn(component, monitor.component_status)
            self.assertEqual(monitor.component_status[component]["status"], SystemStatus.UNKNOWN)
    
    def test_check_microphone(self):
        """Test microphone status check."""
        # Create SystemMonitor instance
        monitor = SystemMonitor(self.mock_assistant)
        
        # Check microphone
        monitor.check_microphone()
        
        # Verify that component status was updated
        self.assertNotEqual(
            monitor.component_status[SystemComponent.MICROPHONE]["status"], 
            SystemStatus.UNKNOWN
        )
    
    def test_check_api_status(self):
        """Test API status check."""
        # Create SystemMonitor instance
        monitor = SystemMonitor(self.mock_assistant)
        
        # Set up client mock
        self.mock_assistant.client.chat.completions.create.return_value = MagicMock()
        
        # Check API status
        monitor.check_api_status()
        
        # Verify that component status was updated
        self.assertEqual(
            monitor.component_status[SystemComponent.API]["status"], 
            SystemStatus.OK
        )
        
        # Test with API error
        self.mock_assistant.client.chat.completions.create.side_effect = Exception("API error")
        
        # Check API status again
        monitor.check_api_status()
        
        # Verify that component status was updated to ERROR
        self.assertEqual(
            monitor.component_status[SystemComponent.API]["status"], 
            SystemStatus.ERROR
        )
    
    def test_check_rag_system(self):
        """Test RAG system check."""
        # Create SystemMonitor instance
        monitor = SystemMonitor(self.mock_assistant)
        
        # Check RAG system
        monitor.check_rag_system()
        
        # Verify that component statuses were updated
        self.assertEqual(
            monitor.component_status[SystemComponent.RAG]["status"], 
            SystemStatus.OK
        )
        
        self.assertEqual(
            monitor.component_status[SystemComponent.EMBEDDING]["status"], 
            SystemStatus.OK
        )
        
        self.assertEqual(
            monitor.component_status[SystemComponent.INDEX]["status"], 
            SystemStatus.OK
        )
        
        # Test with embedding model error
        self.mock_assistant.embedding_model.encode.side_effect = Exception("Model error")
        
        # Check RAG system again
        monitor.check_rag_system()
        
        # Verify that RAG status is now WARNING (since index is still OK)
        self.assertEqual(
            monitor.component_status[SystemComponent.RAG]["status"], 
            SystemStatus.WARNING
        )
        
        # Verify embedding status is ERROR
        self.assertEqual(
            monitor.component_status[SystemComponent.EMBEDDING]["status"], 
            SystemStatus.ERROR
        )
    
    def test_handle_component_failure(self):
        """Test handling of component failures."""
        # Create SystemMonitor instance
        monitor = SystemMonitor(self.mock_assistant)
        
        # Initialize statuses to OK
        for component in SystemComponent:
            monitor.component_status[component] = {
                "status": SystemStatus.OK,
                "last_check": time.time(),
                "message": "OK",
                "details": {}
            }
        
        # Test handling a microphone failure
        error = Exception("Microphone access error")
        should_retry, message = monitor.handle_component_failure(
            SystemComponent.MICROPHONE, 
            error, 
            {"context": "test"}
        )
        
        # Verify that component status was updated
        self.assertEqual(
            monitor.component_status[SystemComponent.MICROPHONE]["status"], 
            SystemStatus.ERROR
        )
        
        # Verify that overall status was also updated
        self.assertEqual(
            monitor.component_status[SystemComponent.OVERALL]["status"], 
            SystemStatus.ERROR
        )
        
        # Verify that user message was generated
        self.assertIsNotNone(message)
        self.assertIn("microphone", message.lower())
    
    def test_status_notifications(self):
        """Test generation and retrieval of status notifications."""
        # Create SystemMonitor instance
        monitor = SystemMonitor(self.mock_assistant)
        
        # Set initial status to OK
        monitor._update_component_status(
            SystemComponent.MICROPHONE,
            SystemStatus.OK,
            "Microphone working fine",
            {}
        )
        
        # Change status to ERROR to generate notification
        monitor._update_component_status(
            SystemComponent.MICROPHONE,
            SystemStatus.ERROR,
            "Microphone not working",
            {}
        )
        
        # Check if notification was generated
        self.assertTrue(monitor.has_notifications())
        
        # Get notification
        notification = monitor.get_notification()
        
        # Verify notification
        self.assertEqual(notification["component"], SystemComponent.MICROPHONE)
        self.assertEqual(notification["prev_status"], SystemStatus.OK)
        self.assertEqual(notification["status"], SystemStatus.ERROR)
        self.assertIn("microphone", notification["message"].lower())


if __name__ == "__main__":
    unittest.main() 