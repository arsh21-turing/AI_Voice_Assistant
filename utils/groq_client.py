"""
Groq API client for the Voice-Powered Car Assistant.

This module provides a client for interacting with the Groq API for language model
capabilities, handling authentication, request management, error handling, and
response parsing.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

# Import configuration
try:
    from config import get_config
except ImportError:
    # For standalone usage or testing
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config

class GroqAPIError(Exception):
    """Exception raised for Groq API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class GroqClient:
    """
    Client for interacting with the Groq API.
    
    This class handles authentication, request formatting, error handling,
    and response parsing for interactions with the Groq API.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the Groq client.
        
        Args:
            config_manager: Optional configuration manager instance.
                            If None, will use the global configuration.
        """
        # Get configuration
        self.config = config_manager if config_manager else get_config()
        
        # Set up API configuration
        self.api_key = self.config.get('API_SETTINGS', 'GROQ_API_KEY')
        self.api_base = self.config.get('API_SETTINGS', 'GROQ_API_BASE', 
                                       'https://api.groq.com/openai/v1')
        self.default_model = self.config.get('API_SETTINGS', 'GROQ_MODEL', 
                                            'llama3-8b-8192')
        self.default_max_tokens = self.config.get('API_SETTINGS', 'MAX_TOKENS', 1024)
        self.default_temperature = self.config.get('API_SETTINGS', 'TEMPERATURE', 0.7)
        
        # Set up retry configuration
        self.max_retries = self.config.get('ERROR_HANDLING', 'MAX_RETRY_ATTEMPTS', 3)
        self.retry_delay = self.config.get('ERROR_HANDLING', 'RETRY_DELAY', 1.0)
        self.timeout = self.config.get('APP_SETTINGS', 'TIMEOUT', 30)
        
        # Error messages
        self.network_error_message = self.config.get(
            'ERROR_HANDLING', 
            'NETWORK_ERROR_MESSAGE', 
            "Sorry, I'm having trouble connecting to the network."
        )
        self.api_error_message = self.config.get(
            'ERROR_HANDLING', 
            'API_ERROR_MESSAGE',
            "Sorry, I'm having trouble accessing the necessary services."
        )
        
        # Validate API key
        if not self.api_key:
            logging.error("Groq API key is not set")
            raise ValueError("Groq API key is not set. Please set GROQ_API_KEY in your configuration.")
        
        # Set up headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logging.info(f"Initialized Groq client with model: {self.default_model}")
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: Optional[str] = None, 
                       max_tokens: Optional[int] = None, 
                       temperature: Optional[float] = None) -> Dict:
        """
        Get a chat completion from the Groq API.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (defaults to configuration setting)
            max_tokens: Maximum tokens to generate (defaults to configuration setting)
            temperature: Sampling temperature (defaults to configuration setting)
            
        Returns:
            Complete Groq API response
            
        Raises:
            GroqAPIError: If there is an error with the API request
        """
        endpoint = f"{self.api_base}/chat/completions"
        
        # Prepare payload
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature
        }
        
        logging.debug(f"Making chat completion request with model {payload['model']}")
        return self._make_request(endpoint, payload)
    
    def completion(self, 
                   prompt: str, 
                   model: Optional[str] = None,
                   max_tokens: Optional[int] = None, 
                   temperature: Optional[float] = None) -> Dict:
        """
        Get a text completion from the Groq API.
        
        Args:
            prompt: Text prompt for completion
            model: Model to use (defaults to configuration setting)
            max_tokens: Maximum tokens to generate (defaults to configuration setting)
            temperature: Sampling temperature (defaults to configuration setting)
            
        Returns:
            Complete Groq API response
            
        Raises:
            GroqAPIError: If there is an error with the API request
        """
        # Convert to chat format since Groq API prefers this format
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def extract_response_text(self, response: Dict) -> str:
        """
        Extract the generated text from a response.
        
        Args:
            response: Groq API response object
            
        Returns:
            The generated text content
        """
        try:
            # For chat completions
            if 'choices' in response and len(response['choices']) > 0:
                if 'message' in response['choices'][0]:
                    return response['choices'][0]['message']['content']
                elif 'text' in response['choices'][0]:
                    return response['choices'][0]['text']
            
            # If we couldn't parse the response
            logging.warning(f"Unexpected response format: {response}")
            return ""
        except Exception as e:
            logging.error(f"Error extracting response text: {str(e)}")
            return ""
    
    def _make_request(self, 
                     endpoint: str, 
                     payload: Dict, 
                     retries: Optional[int] = None) -> Dict:
        """
        Make a request to the Groq API with retry logic.
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            retries: Number of retry attempts (defaults to configuration setting)
            
        Returns:
            API response
            
        Raises:
            GroqAPIError: If there is an error with the API request
        """
        max_attempts = retries if retries is not None else self.max_retries
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            try:
                logging.debug(f"API request attempt {attempt}/{max_attempts} to {endpoint}")
                
                # Make the API request
                response = requests.post(
                    endpoint,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    error_msg = f"API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if 'error' in error_data and 'message' in error_data['error']:
                            error_msg = f"API error: {error_data['error']['message']}"
                    except Exception:
                        # If we can't parse the JSON response
                        error_msg = f"API error: {response.text}"
                    
                    if self._handle_error(
                        Exception(error_msg), 
                        endpoint, 
                        payload, 
                        attempt, 
                        max_attempts
                    ):
                        continue  # Retry
                    else:
                        raise GroqAPIError(
                            error_msg, 
                            status_code=response.status_code, 
                            response=response.text
                        )
                
                # Parse the response
                return response.json()
                
            except (ConnectionError, Timeout) as e:
                # Handle network errors
                if self._handle_error(e, endpoint, payload, attempt, max_attempts):
                    continue  # Retry
                else:
                    raise GroqAPIError(
                        f"Network error: {str(e)}", 
                        status_code=None, 
                        response=None
                    )
                
            except RequestException as e:
                # Handle other request exceptions
                if self._handle_error(e, endpoint, payload, attempt, max_attempts):
                    continue  # Retry
                else:
                    raise GroqAPIError(
                        f"Request error: {str(e)}", 
                        status_code=None, 
                        response=None
                    )
                
            except Exception as e:
                # Handle unexpected errors
                logging.error(f"Unexpected error: {str(e)}")
                raise GroqAPIError(f"Unexpected error: {str(e)}")
        
        # If we exhausted all retries
        raise GroqAPIError(f"API request failed after {max_attempts} attempts")
    
    def _handle_error(self, 
                     error: Exception, 
                     endpoint: str, 
                     payload: Dict, 
                     attempt: int, 
                     max_attempts: int) -> bool:
        """
        Handle API errors and determine if retry is needed.
        
        Args:
            error: The encountered error
            endpoint: API endpoint being called
            payload: Request payload
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            
        Returns:
            True if retry is recommended, False otherwise
        """
        error_str = str(error)
        
        # Log the error
        logging.warning(f"API error on attempt {attempt}/{max_attempts}: {error_str}")
        
        # Check if we should retry
        if attempt < max_attempts:
            # Determine if the error is retryable
            retryable_errors = [
                "timeout", 
                "connection",
                "too many requests",
                "rate limit",
                "server error",
                "5",  # 5xx errors
                "503",
                "502",
                "504"
            ]
            
            is_retryable = any(text in error_str.lower() for text in retryable_errors)
            
            if is_retryable:
                # Calculate delay with exponential backoff
                delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logging.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                return True
        
        # If we shouldn't retry
        return False
    
    def get_fallback_response(self, error_type: str = "api") -> str:
        """
        Get an appropriate fallback response based on the error type.
        
        Args:
            error_type: Type of error ("api", "network", or "general")
            
        Returns:
            Fallback response message
        """
        if error_type == "network":
            return self.network_error_message
        elif error_type == "api":
            return self.api_error_message
        else:
            return self.config.get(
                'ERROR_HANDLING', 
                'DEFAULT_ERROR_MESSAGE',
                "Sorry, I encountered an error while processing your request."
            )
    
    def generate_chat_response(self, 
                              user_query: str, 
                              context: Optional[str] = None,
                              system_prompt: Optional[str] = None) -> Tuple[str, Optional[Dict]]:
        """
        Generate a response to a user query with optional context.
        
        Args:
            user_query: The user's query
            context: Additional context information (optional)
            system_prompt: System prompt to use (optional)
            
        Returns:
            Tuple of (response_text, full_response_object)
        """
        # Get default system prompt if not provided
        if not system_prompt:
            system_prompt = self.config.get(
                'PROMPT_TEMPLATES', 
                'SYSTEM_PROMPT',
                "You are an automotive assistant helping drivers with questions about their vehicle."
            )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add context if provided
        if context:
            messages.append({
                "role": "user", 
                "content": f"Here is some relevant context information: {context}"
            })
            messages.append({
                "role": "assistant", 
                "content": "I'll keep that context in mind when answering your question."
            })
        
        # Add user query
        messages.append({"role": "user", "content": user_query})
        
        try:
            # Make the API request
            response = self.chat_completion(messages)
            
            # Extract text response
            response_text = self.extract_response_text(response)
            
            return response_text, response
            
        except GroqAPIError as e:
            # Handle API errors
            logging.error(f"Groq API error: {str(e)}")
            
            error_type = "api"
            if "Network error" in str(e):
                error_type = "network"
            
            return self.get_fallback_response(error_type), None
            
        except Exception as e:
            # Handle unexpected errors
            logging.error(f"Unexpected error in generate_chat_response: {str(e)}")
            return self.get_fallback_response("general"), None


# For testing or direct usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        client = GroqClient()
        response = client.generate_chat_response(
            "What maintenance should I perform on my car every 10,000 miles?"
        )
        print("Response:", response[0])
    except Exception as e:
        print(f"Error: {str(e)}")