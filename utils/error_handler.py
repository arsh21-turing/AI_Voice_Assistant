"""
Centralized error handling system for the Voice-Powered Car Assistant.

This module provides a comprehensive error handling system that manages API failures,
rate limiting, context retrieval issues, and other errors that may occur during
the operation of the application, with configurable retry policies and user-friendly
error messages.
"""

import time
import logging
import traceback
import random
import sys
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from pathlib import Path
import functools
import inspect
import json

# Import from project modules
try:
    from config import get_config
except ImportError:
    # For standalone usage or testing
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config

class ErrorType(Enum):
    """Enum of error types for categorization."""
    GENERAL = "general"
    API = "api"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    CONTEXT = "context"
    RETRIEVAL = "retrieval"
    SPEECH = "speech"
    RECOGNITION = "recognition"
    CONFIGURATION = "configuration"
    DATA = "data"
    SYSTEM = "system"

class ErrorHandler:
    """
    Centralized error handling system for the application.
    
    This class provides methods for handling errors, determining retry strategies,
    generating user-friendly messages, and logging errors in a consistent manner.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the error handler.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        # Get configuration
        self.config = config_manager if config_manager else get_config()
        
        # Initialize logger for error tracking
        self.logger = logging.getLogger("error_handler")
        
        # Load error handling configuration
        self.max_retry_attempts = self.config.get(
            'ERROR_HANDLING', 
            'MAX_RETRY_ATTEMPTS', 
            3
        )
        self.base_retry_delay = self.config.get(
            'ERROR_HANDLING', 
            'RETRY_DELAY', 
            1.0
        )
        self.jitter_factor = self.config.get(
            'ERROR_HANDLING', 
            'JITTER_FACTOR', 
            0.2
        )
        
        # Load error messages from configuration
        self.error_messages = {
            ErrorType.GENERAL.value: self.config.get(
                'ERROR_HANDLING', 
                'DEFAULT_ERROR_MESSAGE', 
                "Sorry, I encountered an error while processing your request."
            ),
            ErrorType.API.value: self.config.get(
                'ERROR_HANDLING', 
                'API_ERROR_MESSAGE', 
                "Sorry, I'm having trouble accessing the necessary services."
            ),
            ErrorType.NETWORK.value: self.config.get(
                'ERROR_HANDLING', 
                'NETWORK_ERROR_MESSAGE', 
                "Sorry, I'm having trouble connecting to the network."
            ),
            ErrorType.RATE_LIMIT.value: self.config.get(
                'ERROR_HANDLING', 
                'RATE_LIMIT_ERROR_MESSAGE', 
                "I'm receiving too many requests right now. Please try again in a moment."
            ),
            ErrorType.TIMEOUT.value: self.config.get(
                'ERROR_HANDLING', 
                'TIMEOUT_ERROR_MESSAGE', 
                "The request is taking too long. Please try again later."
            ),
            ErrorType.AUTHENTICATION.value: self.config.get(
                'ERROR_HANDLING', 
                'AUTHENTICATION_ERROR_MESSAGE', 
                "I'm having trouble authenticating with the service."
            ),
            ErrorType.CONTEXT.value: self.config.get(
                'ERROR_HANDLING', 
                'CONTEXT_ERROR_MESSAGE', 
                "I couldn't properly understand the context of your question."
            ),
            ErrorType.RETRIEVAL.value: self.config.get(
                'ERROR_HANDLING', 
                'RETRIEVAL_ERROR_MESSAGE', 
                "I'm having trouble retrieving the information you requested."
            ),
            ErrorType.SPEECH.value: self.config.get(
                'ERROR_HANDLING', 
                'SPEECH_ERROR_MESSAGE', 
                "I'm having trouble with the speech system."
            ),
            ErrorType.RECOGNITION.value: self.config.get(
                'ERROR_HANDLING', 
                'RECOGNITION_ERROR_MESSAGE', 
                "Sorry, I didn't catch what you said."
            )
        }
        
        # Initialize fallback registry
        self.fallbacks = {}
        
        # Error tracking for analytics
        self.error_counts = {}
        
        self.logger.info("Initialized error handler")
    
    def handle_error(self, 
                    error: Exception, 
                    error_type: str = "general",
                    context: Optional[Dict] = None) -> Tuple[str, bool]:
        """
        Handle an error and determine appropriate response.
        
        Args:
            error: The exception that occurred
            error_type: Type of error (api, network, context, etc.)
            context: Additional context about the error
            
        Returns:
            Tuple containing user-friendly message and whether to retry
        """
        # Get a valid error type
        error_type = self._normalize_error_type(error_type)
        
        # Track error for analytics
        self._track_error(error_type)
        
        # Log the error
        self.log_error(error, error_type, context)
        
        # Determine if this is an error that should trigger a retry
        should_retry = self._is_retryable_error(error, error_type)
        
        # Get appropriate user-facing message
        user_message = self.get_user_message(error_type, context)
        
        return user_message, should_retry
    
    def get_user_message(self, 
                        error_type: str, 
                        error_details: Optional[Dict] = None) -> str:
        """
        Get a user-facing error message.
        
        Args:
            error_type: Type of error
            error_details: Additional error details
            
        Returns:
            User-friendly error message
        """
        # Normalize error type
        error_type = self._normalize_error_type(error_type)
        
        # Get base message for error type
        message = self.error_messages.get(error_type, self.error_messages[ErrorType.GENERAL.value])
        
        # If we have error details, we might want to customize the message
        if error_details:
            # Check if there's a more specific message for the error code
            if 'code' in error_details:
                code_message = self._get_code_specific_message(error_type, error_details['code'])
                if code_message:
                    message = code_message
            
            # Add specific guidance based on error details if available
            guidance = error_details.get('guidance')
            if guidance:
                message = f"{message} {guidance}"
        
        return message
    
    def _get_code_specific_message(self, error_type: str, code: str) -> Optional[str]:
        """Get a message specific to an error code if available."""
        # Check if there's a configuration for this specific error code
        code_key = f"{error_type.upper()}_ERROR_{code}"
        return self.config.get('ERROR_HANDLING', code_key, None)
    
    def log_error(self, 
                 error: Exception, 
                 error_type: str, 
                 context: Optional[Dict] = None) -> None:
        """
        Log an error with appropriate level and details.
        
        Args:
            error: The exception that occurred
            error_type: Type of error
            context: Additional context about the error
        """
        # Normalize error type
        error_type = self._normalize_error_type(error_type)
        
        # Determine log level based on error type
        if error_type in [ErrorType.AUTHENTICATION.value, ErrorType.SYSTEM.value]:
            log_level = logging.ERROR
        elif error_type in [ErrorType.RATE_LIMIT.value, ErrorType.TIMEOUT.value, ErrorType.NETWORK.value]:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
        
        # Format error message with context
        error_str = str(error)
        
        # Basic error message
        log_message = f"Error type: {error_type}, Message: {error_str}"
        
        # Add context if available
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            log_message = f"{log_message}, Context: {{{context_str}}}"
        
        # Add stack trace for serious errors
        if log_level >= logging.ERROR:
            stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            log_message = f"{log_message}\nStack trace:\n{stack_trace}"
        
        # Log the error
        self.logger.log(log_level, log_message)
    
    def should_retry(self, 
                    error_type: str, 
                    attempt: int, 
                    max_attempts: Optional[int] = None, 
                    context: Optional[Dict] = None) -> bool:
        """
        Determine if operation should be retried.
        
        Args:
            error_type: Type of error
            attempt: Current attempt number
            max_attempts: Maximum allowed attempts (defaults to config value)
            context: Additional context about the error
            
        Returns:
            Whether to retry the operation
        """
        # Normalize error type
        error_type = self._normalize_error_type(error_type)
        
        # Use configured max attempts if not specified
        if max_attempts is None:
            max_attempts = self.max_retry_attempts
            
        # Don't retry if we've reached max attempts
        if attempt >= max_attempts:
            return False
            
        # Determine if the error type should be retried
        retryable_types = [
            ErrorType.NETWORK.value,
            ErrorType.RATE_LIMIT.value,
            ErrorType.TIMEOUT.value,
            ErrorType.API.value,
            ErrorType.RETRIEVAL.value
        ]
        
        # For certain error types like authentication, don't retry
        non_retryable_types = [
            ErrorType.AUTHENTICATION.value,
            ErrorType.CONFIGURATION.value
        ]
        
        # Check context for specific retry guidance
        if context and 'retryable' in context:
            return bool(context['retryable'])
            
        # Default retry behavior based on error type
        if error_type in non_retryable_types:
            return False
            
        if error_type in retryable_types:
            return True
            
        # For other types, retry with diminishing probability
        # The higher the attempt, the less likely we'll retry
        retry_probability = 1.0 / (attempt + 1)
        return random.random() < retry_probability
    
    def get_retry_delay(self, error_type: str, attempt: int) -> float:
        """
        Calculate delay before retry with exponential backoff and jitter.
        
        Args:
            error_type: Type of error
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds before next retry
        """
        # Normalize error type
        error_type = self._normalize_error_type(error_type)
        
        # Base delay (from configuration)
        base_delay = self.base_retry_delay
        
        # Apply different base delays for different error types
        if error_type == ErrorType.RATE_LIMIT.value:
            # Rate limit errors need longer delays
            base_delay = base_delay * 3
            
        # Exponential backoff: base_delay * 2^(attempt-1)
        delay = base_delay * (2 ** (attempt - 1))
        
        # Add jitter to avoid thundering herd problem
        jitter = delay * self.jitter_factor * random.random()
        delay = delay + jitter
        
        # Cap the maximum delay at 30 seconds
        return min(delay, 30.0)
    
    def register_fallback(self, error_type: str, fallback_fn: Callable) -> None:
        """
        Register a fallback function for an error type.
        
        Args:
            error_type: Type of error
            fallback_fn: Function to call for fallback handling
        """
        # Normalize error type
        error_type = self._normalize_error_type(error_type)
        
        # Register the fallback
        self.fallbacks[error_type] = fallback_fn
        self.logger.debug(f"Registered fallback for error type: {error_type}")
    
    def execute_with_fallback(self, 
                             func: Callable, 
                             args: Optional[List] = None, 
                             kwargs: Optional[Dict] = None, 
                             error_types: Optional[List[str]] = None, 
                             fallback: Optional[Callable] = None) -> Any:
        """
        Execute a function with automatic error handling and fallback.
        
        Args:
            func: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            error_types: Error types to handle (defaults to all)
            fallback: Fallback function if all retries fail
            
        Returns:
            The result of the function or fallback
        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
            
        # If no specific error types provided, handle all error types
        if error_types is None:
            error_types = [e.value for e in ErrorType]
            
        # Normalize error types
        error_types = [self._normalize_error_type(t) for t in error_types]
        
        max_attempts = self.max_retry_attempts
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            try:
                # Execute the function
                return func(*args, **kwargs)
                
            except Exception as e:
                # Determine error type based on exception
                error_type = self._get_error_type_from_exception(e)
                
                # Skip handling if this error type isn't in our list
                if error_type not in error_types:
                    raise
                
                # Get error handling info
                user_message, should_retry = self.handle_error(
                    e, error_type, {'attempt': attempt, 'function': func.__name__}
                )
                
                # If we shouldn't retry, or this is our last attempt, use fallback
                if not should_retry or attempt >= max_attempts:
                    # Try registered fallback first
                    if error_type in self.fallbacks:
                        return self.fallbacks[error_type](e, *args, **kwargs)
                    
                    # Then try provided fallback
                    if fallback is not None:
                        return fallback(e, *args, **kwargs)
                    
                    # If no fallback available, re-raise the exception
                    raise
                
                # Retry with delay
                retry_delay = self.get_retry_delay(error_type, attempt)
                self.logger.info(f"Retrying {func.__name__} after {retry_delay:.2f}s delay (attempt {attempt}/{max_attempts})")
                time.sleep(retry_delay)
    
    def with_error_handling(self, 
                           error_types: Optional[List[str]] = None,
                           fallback: Optional[Callable] = None):
        """
        Decorator for functions that should use error handling.
        
        Args:
            error_types: Error types to handle (defaults to all)
            fallback: Fallback function if all retries fail
            
        Returns:
            Decorated function with error handling
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_fallback(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    error_types=error_types,
                    fallback=fallback
                )
            return wrapper
        return decorator
    
    def _normalize_error_type(self, error_type: str) -> str:
        """Convert error type to a valid enum value."""
        try:
            return ErrorType(error_type).value
        except ValueError:
            # If not a valid enum value, return general error type
            return ErrorType.GENERAL.value
    
    def _get_error_type_from_exception(self, error: Exception) -> str:
        """Determine the error type based on the exception class."""
        error_class = error.__class__.__name__
        
        # Common API/HTTP errors
        if "Timeout" in error_class or "TimeoutError" in error_class:
            return ErrorType.TIMEOUT.value
            
        if "Connection" in error_class or "Network" in error_class:
            return ErrorType.NETWORK.value
            
        if "RateLimit" in error_class or "TooManyRequests" in error_class:
            return ErrorType.RATE_LIMIT.value
            
        if "Auth" in error_class or "Credentials" in error_class or "Token" in error_class:
            return ErrorType.AUTHENTICATION.value
            
        # Look at error message for clues
        error_message = str(error).lower()
        
        if "rate limit" in error_message or "too many requests" in error_message:
            return ErrorType.RATE_LIMIT.value
            
        if "timeout" in error_message or "timed out" in error_message:
            return ErrorType.TIMEOUT.value
            
        if "network" in error_message or "connection" in error_message:
            return ErrorType.NETWORK.value
            
        if "api" in error_message or "service" in error_message:
            return ErrorType.API.value
            
        if "context" in error_message:
            return ErrorType.CONTEXT.value
            
        if "retriev" in error_message or "find" in error_message or "search" in error_message:
            return ErrorType.RETRIEVAL.value
            
        if "speech" in error_message or "voice" in error_message:
            return ErrorType.SPEECH.value
            
        if "recogni" in error_message or "listen" in error_message or "hear" in error_message:
            return ErrorType.RECOGNITION.value
            
        # Default to general error type
        return ErrorType.GENERAL.value
    
    def _is_retryable_error(self, error: Exception, error_type: str) -> bool:
        """Determine if an error is retryable based on error and type."""
        # Check based on error type first
        if error_type in [ErrorType.RATE_LIMIT.value, ErrorType.TIMEOUT.value, ErrorType.NETWORK.value]:
            return True
            
        if error_type in [ErrorType.AUTHENTICATION.value, ErrorType.CONFIGURATION.value]:
            return False
            
        # Check based on exception class or message
        error_class = error.__class__.__name__
        error_message = str(error).lower()
        
        retryable_substrings = [
            "timeout", "timed out", "retry", "temporary", "connection",
            "network", "rate limit", "too many requests", "server error",
            "503", "502", "504", "unavailable", "try again"
        ]
        
        for substring in retryable_substrings:
            if substring in error_message or substring in error_class.lower():
                return True
                
        return False
    
    def _track_error(self, error_type: str) -> None:
        """Track error occurrences for analytics."""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
    
    def get_error_statistics(self) -> Dict[str, int]:
        """
        Get statistics on error occurrences.
        
        Returns:
            Dictionary of error types and their counts
        """
        return dict(self.error_counts)
    
    def reset_error_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts = {}


# For testing or direct usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Example test
    error_handler = ErrorHandler()
    
    # Test error handling
    def test_function(succeed=False):
        """Test function that fails unless succeed=True."""
        if not succeed:
            raise Exception("Test error message")
        return "Success!"
    
    # Register a fallback
    def test_fallback(error, succeed=False):
        """Fallback function for test_function."""
        return f"Fallback response due to: {str(error)}"
    
    error_handler.register_fallback(ErrorType.GENERAL.value, test_fallback)
    
    # Test with direct execution
    try:
        message, retry = error_handler.handle_error(
            Exception("Network connection failed"),
            ErrorType.NETWORK.value
        )
        print(f"Message: {message}")
        print(f"Should retry: {retry}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test with execute_with_fallback
    result = error_handler.execute_with_fallback(
        test_function,
        kwargs={"succeed": False}
    )
    print(f"Result with fallback: {result}")
    
    # Test with decorator
    @error_handler.with_error_handling()
    def decorated_function(succeed=False):
        if not succeed:
            raise TimeoutError("Operation timed out")
        return "Decorated success!"
    
    try:
        result = decorated_function(succeed=False)
        print(f"Decorated result: {result}")
    except Exception as e:
        print(f"Decorated function error: {str(e)}")
    
    # Print error statistics
    print(f"Error statistics: {error_handler.get_error_statistics()}")