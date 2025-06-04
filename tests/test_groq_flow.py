"""
End-to-end tests for the Groq API integration.

This module provides comprehensive tests for the entire flow of the Groq API
integration, including client initialization, API connectivity, response handling,
error handling, and integration with other components.
"""

import unittest
import os
import time
import pytest
import logging
import json
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import re
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Create mock modules for rag package before importing anything else
# This ensures we don't accidentally import the real modules
class MockPromptTemplateManager:
    """Mock implementation of PromptTemplateManager for testing."""
    
    def __init__(self, config_manager=None):
        """Initialize with default templates."""
        self.templates = {
            'SYSTEM_PROMPT': "You are an automotive assistant helping drivers with questions about their vehicle.",
            'RESPONSE_PROMPT': "Based on the context information and the user's question, provide a helpful response.\n\nContext: {context}\n\nUser question: {query}\n\nResponse:",
            'CONTEXT_FORMAT_PROMPT': "Format these context chunks into a coherent document.\n\nChunks: {chunks}\n\nFormatted context:",
            'VOICE_OPTIMIZATION_GUIDELINES': "Optimize responses for voice by avoiding special characters and using clear language."
        }
        self.config = config_manager
        
    def get_template(self, template_name, default=None):
        """Get a template by name."""
        return self.templates.get(template_name, default or f"Default template for {template_name}")
        
    def get_system_prompt(self):
        """Get the system prompt."""
        return self.templates.get('SYSTEM_PROMPT')
        
    def get_response_prompt(self, response_type="general"):
        """Get response prompt based on type."""
        template_name = f"{response_type.upper()}_RESPONSE_PROMPT"
        if template_name in self.templates:
            return self.templates[template_name]
        return self.templates['RESPONSE_PROMPT']
        
    def get_voice_optimization_guidelines(self):
        """Get voice optimization guidelines."""
        return self.templates.get('VOICE_OPTIMIZATION_GUIDELINES')
        
    def format_template(self, template_name, **kwargs):
        """Format a template with variables."""
        template = self.get_template(template_name)
        for key, value in kwargs.items():
            template = template.replace(f"{{{key}}}", str(value))
        return template


class MockContextManager:
    """Mock implementation of ContextManager for testing."""
    
    def __init__(self, config_manager=None):
        """Initialize with empty history."""
        self.history = []
        self.interaction_counter = 0
        self.session_state = {
            'vehicle_info': {},
            'preferences': {},
            'current_topic': None,
            'flags': {},
        }
        self.config = config_manager
        
    def add_interaction(self, query, response, context_chunks=None, metadata=None):
        """Add an interaction to history."""
        self.interaction_counter += 1
        interaction = {
            'id': self.interaction_counter,
            'timestamp': time.time(),
            'query': query,
            'response': response,
            'context_chunks': context_chunks if context_chunks else [],
            'metadata': metadata if metadata else {}
        }
        self.history.append(interaction)
        return self.interaction_counter
        
    def get_recent_interactions(self, count=None):
        """Get recent interactions."""
        if count is None:
            count = 5
        return self.history[-count:] if count > 0 else []
        
    def get_relevant_interactions(self, query, threshold=0.0):
        """Get interactions relevant to query."""
        # Simple mock implementation - return all interactions
        return [{**interaction, 'relevance_score': 0.8} for interaction in self.history]
        
    def get_conversation_context(self, query, format='text'):
        interactions = self.get_recent_interactions()
        if format == 'text':
            return "\n\n".join([
                f"User: {i['query']}\nAssistant: {i['response']}"
                for i in interactions
            ])
        elif format == 'dict':
            return interactions
        else:
            context_messages = []
            for i in interactions:
                context_messages.append({"role": "user", "content": i["query"]})
                context_messages.append({"role": "assistant", "content": i["response"]})
            return context_messages

            
    def update_session_state(self, key, value):
        """Update session state."""
        if '.' in key:
            parts = key.split('.')
            current = self.session_state
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.session_state[key] = value
            
    def get_session_state(self, key, default=None):
        """Get session state value."""
        if '.' in key:
            parts = key.split('.')
            current = self.session_state
            for part in parts:
                if part not in current:
                    return default
                current = current[part]
            return current
        return self.session_state.get(key, default)


# Mock the entire rag package
sys.modules['rag'] = MagicMock()
sys.modules['rag.templates'] = MagicMock()
sys.modules['rag.context'] = MagicMock()
sys.modules['rag.retriever'] = MagicMock()
sys.modules['rag.processor'] = MagicMock()
sys.modules['rag.generator'] = MagicMock()

# Make the mock classes available through the mocked modules
sys.modules['rag.templates'].PromptTemplateManager = MockPromptTemplateManager
sys.modules['rag.context'].ContextManager = MockContextManager

# Import configuration and utils modules
from config import get_config, ConfigManager
from utils.groq_client import GroqClient, GroqAPIError
from utils.error_handler import ErrorHandler, ErrorType
from speech.formatting import VoiceResponseFormatter

# Now import from our mocked rag modules
from rag.templates import PromptTemplateManager
from rag.context import ContextManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine if we're running in a CI environment
IN_CI_ENVIRONMENT = os.environ.get('CI', 'false').lower() == 'true'

# Skip marker for tests that require API access
requires_api = pytest.mark.skipif(
    IN_CI_ENVIRONMENT,
    reason="Skipping test that requires Groq API access in CI environment"
)

class TestGroqFlow(unittest.TestCase):
    """Tests for the Groq API integration flow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - runs once before all tests."""
        # Initialize configuration
        try:
            env_file = Path(__file__).parent / '.env.test'
            if env_file.exists():
                cls.config = ConfigManager(env_file=str(env_file))
            else:
                cls.config = get_config()
                
            # Check if we have API credentials
            cls.has_api_credentials = bool(cls.config.get('API_SETTINGS', 'GROQ_API_KEY', ''))
            
            # Initialize components needed for tests
            cls.error_handler = ErrorHandler(cls.config)
            cls.templates = PromptTemplateManager(cls.config)
            cls.formatter = VoiceResponseFormatter(cls.config)
            
        except Exception as e:
            logger.error(f"Error in setUpClass: {str(e)}")
            cls.has_api_credentials = False
    
    def setUp(self):
        """Set up each test."""
        # Skip tests requiring API credentials if they're not available
        if hasattr(self, 'requires_api') and self.requires_api and not self.has_api_credentials:
            self.skipTest("Groq API credentials not available")
        
        # Create a fresh client for each test
        self.client = GroqClient(self.config)
        self.context_manager = ContextManager(self.config)
    
    def test_groq_client_initialization(self):
        """Test GroqClient initialization with configuration."""
        # Verify client was initialized properly
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.api_base, self.config.get('API_SETTINGS', 'GROQ_API_BASE', 'https://api.groq.com/openai/v1'))
        self.assertEqual(self.client.default_model, self.config.get('API_SETTINGS', 'GROQ_MODEL', 'llama3-8b-8192'))
    
    @requires_api
    def test_groq_api_connection(self):
        """Test basic connectivity to the Groq API."""
        # Simple completion request to test connectivity
        try:
            start_time = time.time()
            response = self.client.completion("Hello, this is a test.")
            duration = time.time() - start_time
            
            # Verify we got a response
            self.assertIsNotNone(response)
            self.assertIn('choices', response)
            self.assertGreater(len(response['choices']), 0)
            
            # Log performance metric
            logger.info(f"API response time: {duration:.2f} seconds")
            
        except GroqAPIError as e:
            self.fail(f"API connection failed: {str(e)}")
    
    @requires_api
    def test_chat_completion(self):
        """Test the chat completion endpoint with a simple prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the primary components in a car's cooling system?"}
        ]
        
        start_time = time.time()
        response = self.client.chat_completion(messages)
        duration = time.time() - start_time
        
        # Extract the response text
        response_text = self.client.extract_response_text(response)
        
        # Verify the response
        self.assertIsNotNone(response_text)
        self.assertTrue(len(response_text) > 50, "Response too short")
        
        # Check for expected content about cooling systems
        cooling_terms = ['radiator', 'coolant', 'water pump', 'thermostat', 'hoses']
        found_terms = [term for term in cooling_terms if term.lower() in response_text.lower()]
        self.assertTrue(len(found_terms) >= 3, f"Expected at least 3 cooling system terms, found {found_terms}")
        
        logger.info(f"Chat completion response time: {duration:.2f} seconds")
    
    def test_error_handling(self):
        """Test error handling for API errors."""
        # Test with invalid API key
        with patch.object(self.client, 'api_key', 'invalid_key'):
            with patch.object(self.error_handler, 'handle_error') as mock_handle_error:
                # Set up mock to return a message and False for retry
                mock_handle_error.return_value = ("API authentication error", False)
                
                # Try to make a request that will fail
                with self.assertRaises(Exception):
                    messages = [{"role": "user", "content": "This should fail"}]
                    self.client.chat_completion(messages)
                
                # Verify handle_error was called
                self.assertTrue(mock_handle_error.called)
                
                # Check the error type passed to handle_error
                args, kwargs = mock_handle_error.call_args
                error, error_type = args[0], args[1]
                self.assertIn(error_type, [ErrorType.API.value, ErrorType.AUTHENTICATION.value])
    
    def test_retry_mechanism(self):
        """Test the retry mechanism with simulated failures."""
        # Create a mock for the _make_request method to simulate failures
        with patch.object(self.client, '_make_request') as mock_request:
            # First call raises timeout, second call succeeds
            mock_request.side_effect = [
                GroqAPIError("Request timed out", None, None),
                {"choices": [{"message": {"content": "Success after retry"}}]}
            ]
            
            # Make a request that will retry
            messages = [{"role": "user", "content": "Test retry"}]
            
            # Configure retry behavior
            with patch.object(self.client, 'max_retries', 2):
                with patch.object(self.client, 'retry_delay', 0.01):  # Small delay for tests
                    response = self.client.chat_completion(messages)
                    
                    # Verify the retry worked
                    self.assertEqual(mock_request.call_count, 2)
                    self.assertIn('choices', response)
                    
                    # Check the response after retry
                    response_text = self.client.extract_response_text(response)
                    self.assertEqual(response_text, "Success after retry")
    
    def test_response_extraction(self):
        """Test extracting content from API responses."""
        # Test different response formats
        test_responses = [
            # Standard chat completion response
            {
                "id": "abc123",
                "choices": [
                    {"message": {"content": "This is a test response"}}
                ]
            },
            # Completion response format
            {
                "id": "def456",
                "choices": [
                    {"text": "This is another format"}
                ]
            }
        ]
        
        expected_extractions = [
            "This is a test response",
            "This is another format"
        ]
        
        for response, expected in zip(test_responses, expected_extractions):
            extracted = self.client.extract_response_text(response)
            self.assertEqual(extracted, expected)
        
        # Test with an unexpected format
        with self.assertLogs(level='WARNING') as log:
            weird_response = {"something": "unexpected"}
            result = self.client.extract_response_text(weird_response)
            self.assertEqual(result, "")
            self.assertIn("WARNING", log.output[0])
    
    def test_context_integration(self):
        """Test using conversation context with the API."""
        # Set up a conversation context
        self.context_manager.add_interaction(
            "What's the recommended oil change interval for my car?",
            "For most modern vehicles, it's recommended to change your oil every 7,500 to 10,000 miles or every 6 months, whichever comes first."
        )
        
        self.context_manager.add_interaction(
            "What type of oil should I use?",
            "Most vehicles use 5W-30 or 5W-20 synthetic oil. Check your owner's manual for the exact specification for your vehicle."
        )
        
        # Get conversation context
        context_text = self.context_manager.get_conversation_context(
            "How do I check my oil level?", format='text'
        )
        
        # Create mock client response
        mock_response = {
            "choices": [
                {"message": {"content": "To check your oil level, make sure your car is on level ground with the engine off and cooled down. Locate the dipstick, pull it out and wipe it clean, then reinsert it fully and pull it out again to check the level."}}
            ]
        }
        
        # Test with patched client
        with patch.object(self.client, 'chat_completion', return_value=mock_response):
            # Create system and user messages with context
            messages = [
                {"role": "system", "content": self.templates.get_system_prompt()},
                {"role": "user", "content": f"Previous conversation:\n{context_text}\n\nNew question: How do I check my oil level?"}
            ]
            
            response = self.client.chat_completion(messages)
            response_text = self.client.extract_response_text(response)
            
            # Verify response
            self.assertIn("dipstick", response_text)
            self.assertIn("level ground", response_text)
    
    def test_template_integration(self):
        """Test using prompt templates with the API."""
        # Create template manager and get templates
        system_prompt = self.templates.get_system_prompt()
        response_prompt = self.templates.get_template('RESPONSE_PROMPT')
        
        # Format template with test data
        formatted_prompt = response_prompt.replace(
            "{context}", "The recommended tire pressure is 32-35 PSI for most passenger cars."
        ).replace(
            "{query}", "What should my tire pressure be?"
        )
        
        # Create mock response
        mock_response = {
            "choices": [
                {"message": {"content": "The recommended tire pressure for most passenger cars is between 32 and 35 PSI (pounds per square inch)."}}
            ]
        }
        
        # Test with patched client
        with patch.object(self.client, 'chat_completion', return_value=mock_response):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_prompt}
            ]
            
            response = self.client.chat_completion(messages)
            response_text = self.client.extract_response_text(response)
            
            # Verify template was correctly used
            self.assertIn("32", response_text)
            self.assertIn("35", response_text)
            self.assertIn("PSI", response_text)
    
    @requires_api
    def test_system_prompt_effect(self):
        """Test how different system prompts affect responses."""
        test_query = "What should I do if my check engine light comes on?"
        
        # Test with two different system prompts
        system_prompts = [
            "You are a helpful assistant.",
            "You are an automotive expert who gives detailed technical advice about cars. Include specific steps in your answers."
        ]
        
        responses = []
        
        for prompt in system_prompts:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": test_query}
            ]
            
            response = self.client.chat_completion(messages)
            response_text = self.client.extract_response_text(response)
            responses.append(response_text)
            
            # Avoid rate limiting
            time.sleep(1)
        
        # Verify difference between responses
        self.assertNotEqual(responses[0], responses[1], "System prompts should yield different responses")
        
        # For the technical prompt, expect more technical details
        technical_response = responses[1]
        technical_terms = ['diagnostic', 'OBD', 'code reader', 'scanner', 'sensor']
        found_terms = [term for term in technical_terms if term.lower() in technical_response.lower()]
        self.assertTrue(len(found_terms) >= 2, f"Technical prompt should yield technical terms, found {found_terms}")
        
        # Check for step-based format in the technical response
        step_patterns = ['first', 'step', '1\.', 'begin by', 'next', 'then', 'finally']
        has_steps = any(re.search(pattern, technical_response, re.IGNORECASE) for pattern in step_patterns)
        self.assertTrue(has_steps, "Technical prompt should yield step-based instructions")
    
    def test_combined_flow(self):
        """Test the end-to-end flow from user input to response."""
        user_query = "How often should I replace my spark plugs?"
        
        # 1. Get conversation context
        context = self.context_manager.get_conversation_context(user_query)
        
        # 2. Get template
        system_prompt = self.templates.get_system_prompt()
        response_prompt = self.templates.get_response_prompt()
        
        # 3. Prepare context information (mock for testing)
        mock_context = "Spark plugs typically need replacement every 30,000 to 90,000 miles depending on the type. Iridium and platinum plugs last longer than copper plugs."
        
        # 4. Format prompt
        formatted_prompt = response_prompt.replace(
            "{context}", mock_context
        ).replace(
            "{query}", user_query
        )
        
        # 5. Create mock API response
        mock_response = {
            "choices": [
                {"message": {"content": "You should replace your spark plugs every 30,000 to 90,000 miles, depending on the type. Copper plugs need replacement more frequently (30,000-40,000 miles), while iridium or platinum plugs can last much longer (60,000-90,000 miles). Check your owner's manual for the specific recommendation for your vehicle."}}
            ]
        }
        
        # 6. Make API call with error handling
        with patch.object(self.client, 'chat_completion', return_value=mock_response):
            with patch.object(self.error_handler, 'execute_with_fallback') as mock_execute:
                # Set up execute_with_fallback to call the original method
                def side_effect(func, args=None, kwargs=None, **options):
                    args = args or []
                    kwargs = kwargs or {}
                    return func(*args, **kwargs)
                    
                mock_execute.side_effect = side_effect
                
                # Execute flow
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt}
                ]
                
                response = self.error_handler.execute_with_fallback(
                    self.client.chat_completion,
                    args=[messages]
                )
                
                # Verify execute_with_fallback was called
                self.assertTrue(mock_execute.called)
                
                # 7. Extract and format response
                response_text = self.client.extract_response_text(response)
                formatted_for_speech = self.formatter.format_response(response_text)
                
                # 8. Verify the flow
                self.assertIn("spark plugs", response_text)
                self.assertIn("miles", response_text)
                self.assertIn("copper", response_text)
                self.assertIn("iridium", response_text)
                
        # 9. Add interaction to context
        self.context_manager.add_interaction(user_query, response_text)
        self.assertTrue(len(self.context_manager.history) > 0)
    
    def test_response_formatting(self):
        """Test formatting API responses for speech output."""
        # Create a response with various elements that need formatting
        test_response = """
        According to the manual, you should change your oil every 5,000-7,500 miles using 5W-30 oil. The ABS system may trigger a code P0300 if oil quality is poor.
        
        Section 7.3 of your manual states that the temperature should be between 195°F and 220°F. Your vehicle has a 3.5L V6 engine that produces 270hp at 5,500rpm.
        
        WARNING: Never drive with less than 1/4 tank of fuel!
        """
        
        # Format for speech
        formatted = self.formatter.format_response(test_response)
        
        # Test number formatting
        self.assertIn("5", formatted)
        
        # Test technical terms
        self.assertIn("ABS", formatted) or self.assertIn("A B S", formatted)
        
        # Test unit formatting
        self.assertIn("degrees", formatted) or self.assertIn("°F", formatted)
        
        # Test section references
        self.assertIn("Section 7", formatted) and (self.assertIn("point 3", formatted) or self.assertIn(".3", formatted))
    
    @requires_api
    def test_automotive_query_answering(self):
        """Test domain-specific automotive queries."""
        # A list of automotive queries to test
        automotive_queries = [
            "What happens if I use the wrong octane fuel in my car?",
            "How do I know when my brakes need to be replaced?",
            "What causes a car to overheat?"
        ]
        
        # Test each query
        for query in automotive_queries:
            # Format with automotive system prompt
            system_prompt = self.templates.get_system_prompt()
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Make the API call
            start_time = time.time()
            response = self.client.chat_completion(messages)
            duration = time.time() - start_time
            
            # Extract response
            response_text = self.client.extract_response_text(response)
            
            # Verify response quality
            self.assertTrue(len(response_text) > 100, f"Response too short for query: {query}")
            self.assertFalse(response_text.startswith("I don't know"), f"Got 'I don't know' response for: {query}")
            
            # Log performance
            logger.info(f"Query: '{query}' - Response time: {duration:.2f}s - Length: {len(response_text)} chars")
            
            # Avoid rate limiting
            time.sleep(1)
    
    def test_mock_responses(self):
        """Test with mock responses to avoid API calls."""
        mock_data_path = Path(__file__).parent / 'mock_data' / 'groq_responses.json'
        
        # If mock data file doesn't exist, create a simple one for testing
        if not mock_data_path.exists():
            os.makedirs(mock_data_path.parent, exist_ok=True)
            mock_data = {
                "how_to_change_oil": {
                    "choices": [
                        {"message": {"content": "To change your oil, first make sure the engine is cool but slightly warm..."}}
                    ]
                },
                "check_tire_pressure": {
                    "choices": [
                        {"message": {"content": "To check your tire pressure, you'll need a tire pressure gauge..."}}
                    ]
                }
            }
            with open(mock_data_path, 'w') as f:
                json.dump(mock_data, f)
        
        # Load mock responses
        with open(mock_data_path, 'r') as f:
            mock_responses = json.load(f)
        
        # Test with a mock response
        with patch.object(self.client, 'chat_completion') as mock_chat:
            # Set up mock to return different responses based on query
            def mock_response_lookup(*args, **kwargs):
                messages = args[0]
                query = messages[-1]['content'].lower()
                
                if 'oil' in query:
                    return mock_responses['how_to_change_oil']
                elif 'tire' in query or 'pressure' in query:
                    return mock_responses['check_tire_pressure']
                else:
                    return {"choices": [{"message": {"content": "I don't have specific information about that."}}]}
            
            mock_chat.side_effect = mock_response_lookup
            
            # Test mock responses
            oil_query = [{"role": "user", "content": "How do I change my oil?"}]
            tire_query = [{"role": "user", "content": "How do I check tire pressure?"}]
            
            oil_response = self.client.chat_completion(oil_query)
            tire_response = self.client.chat_completion(tire_query)
            
            # Verify correct mock responses were returned
            self.assertIn("change your oil", self.client.extract_response_text(oil_response))
            self.assertIn("tire pressure gauge", self.client.extract_response_text(tire_response))


if __name__ == '__main__':
    unittest.main()