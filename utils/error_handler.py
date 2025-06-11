"""
Centralized error handling system for the Voice-Powered Car Assistant.

This module provides a comprehensive error handling system that manages API failures,
rate limiting, context retrieval issues, and other errors that may occur during
the operation of the application, with configurable retry policies, custom handlers
for each error type, and user-friendly error messages.
"""

import time
import logging
import traceback
import random
import sys
import os
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, Type
from enum import Enum, auto
from pathlib import Path
import functools
import inspect
import importlib
from dataclasses import dataclass, field

# Import from project modules
try:
    from config import get_config
except ImportError:
    # For standalone usage or testing
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config


class ErrorSeverity(Enum):
    """Enum representing the severity of an error."""
    DEBUG = 10      # Minor issues that don't affect functionality
    INFO = 20       # Informational errors that don't require action
    WARNING = 30    # Problems that might require attention
    ERROR = 40      # Significant problems requiring intervention
    CRITICAL = 50   # Severe issues that prevent operation


class ErrorCategory(Enum):
    """High-level categorization of errors."""
    SYSTEM = auto()         # System-level errors (OS, hardware, base services)
    CONNECTIVITY = auto()   # Network, API, and connection issues
    DATA = auto()           # Data processing, storage, and retrieval
    USER_INTERACTION = auto()  # User interface, speech, recognition
    SECURITY = auto()       # Authentication, permissions, encryption
    RESOURCE = auto()       # Resource limitations (memory, CPU, etc.)
    CONFIGURATION = auto()  # Setup and configuration issues


class ErrorType(Enum):
    """Detailed enum of error types for categorization."""
    # General errors
    GENERAL = "general"
    UNKNOWN = "unknown"
    
    # API and connectivity errors (CONNECTIVITY category)
    API = "api"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    
    # Data processing errors (DATA category)
    CONTEXT = "context"
    RETRIEVAL = "retrieval"
    DATA_FORMAT = "data_format"
    DATA_MISSING = "data_missing"
    DATA_CORRUPTION = "data_corruption"
    PDF_PROCESSING = "pdf_processing"
    EMBEDDING = "embedding"
    INDEX = "index"
    
    # User interaction errors (USER_INTERACTION category)
    SPEECH = "speech"
    RECOGNITION = "recognition"
    OUTPUT = "output"
    TEXT_TO_SPEECH = "text_to_speech"
    USER_INPUT = "user_input"
    
    # System errors (SYSTEM category)
    SYSTEM = "system"
    MEMORY = "memory"
    FILE_SYSTEM = "file_system"
    PROCESS = "process"
    HARDWARE = "hardware"
    AUDIO_DEVICE = "audio_device"
    
    # Configuration errors (CONFIGURATION category)
    CONFIGURATION = "configuration"
    ENVIRONMENT = "environment"
    DEPENDENCY = "dependency"
    INITIALIZATION = "initialization"
    
    # Security errors (SECURITY category)
    PERMISSION = "permission"
    ENCRYPTION = "encryption"
    TOKEN = "token"
    
    # Resource errors (RESOURCE category)
    RESOURCE_EXHAUSTED = "resource_exhausted"
    QUOTA_EXCEEDED = "quota_exceeded"
    LIMIT_REACHED = "limit_reached"

    @classmethod
    def get_category(cls, error_type) -> ErrorCategory:
        """Get the high-level category for an error type."""
        # Convert string to enum if needed
        if isinstance(error_type, str):
            try:
                error_type = cls(error_type)
            except ValueError:
                return ErrorCategory.SYSTEM
        
        # Connectivity category
        if error_type in (cls.API, cls.NETWORK, cls.RATE_LIMIT, cls.TIMEOUT):
            return ErrorCategory.CONNECTIVITY
            
        # Data category
        if error_type in (cls.CONTEXT, cls.RETRIEVAL, cls.DATA_FORMAT, cls.DATA_MISSING, 
                          cls.DATA_CORRUPTION, cls.PDF_PROCESSING, cls.EMBEDDING, cls.INDEX):
            return ErrorCategory.DATA
            
        # User interaction category
        if error_type in (cls.SPEECH, cls.RECOGNITION, cls.OUTPUT, 
                          cls.TEXT_TO_SPEECH, cls.USER_INPUT):
            return ErrorCategory.USER_INTERACTION
            
        # System category
        if error_type in (cls.SYSTEM, cls.MEMORY, cls.FILE_SYSTEM, 
                          cls.PROCESS, cls.HARDWARE, cls.AUDIO_DEVICE):
            return ErrorCategory.SYSTEM
            
        # Configuration category
        if error_type in (cls.CONFIGURATION, cls.ENVIRONMENT, 
                          cls.DEPENDENCY, cls.INITIALIZATION):
            return ErrorCategory.CONFIGURATION
            
        # Security category
        if error_type in (cls.AUTHENTICATION, cls.PERMISSION, 
                          cls.ENCRYPTION, cls.TOKEN):
            return ErrorCategory.SECURITY
            
        # Resource category
        if error_type in (cls.RESOURCE_EXHAUSTED, cls.QUOTA_EXCEEDED, 
                          cls.LIMIT_REACHED):
            return ErrorCategory.RESOURCE
            
        # Default for general errors
        return ErrorCategory.SYSTEM


@dataclass
class ErrorContext:
    """Container for contextual information about an error."""
    error_type: ErrorType
    exception: Optional[Exception] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    component: str = "unknown"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    @property
    def has_exception(self) -> bool:
        """Check if this error context has an exception."""
        return self.exception is not None
    
    @property
    def category(self) -> ErrorCategory:
        """Get the high-level category for this error."""
        return ErrorType.get_category(self.error_type)
    
    @property
    def should_retry(self) -> bool:
        """Check if we should retry based on retry count."""
        return self.retry_count < self.max_retries
    
    @property
    def error_id(self) -> str:
        """Generate a unique ID for this error instance."""
        if not hasattr(self, '_error_id'):
            # Create a unique ID based on timestamp and hash of exception
            exception_hash = hash(str(self.exception)) if self.exception else random.randint(0, 10000)
            self._error_id = f"{int(self.timestamp * 1000)}-{exception_hash}"
        return self._error_id
    
    def add_detail(self, key: str, value: Any) -> 'ErrorContext':
        """Add a detail to the error context."""
        self.details[key] = value
        return self
    
    def increment_retry(self) -> 'ErrorContext':
        """Increment the retry counter."""
        self.retry_count += 1
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error context to a dictionary."""
        return {
            "error_id": self.error_id,
            "error_type": self.error_type.value if isinstance(self.error_type, ErrorType) else str(self.error_type),
            "category": self.category.name,
            "message": self.message or (str(self.exception) if self.exception else "Unknown error"),
            "details": self.details,
            "severity": self.severity.name,
            "timestamp": self.timestamp,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "component": self.component,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id
        }


class ErrorHandlerInterface:
    """Interface that all specific error handlers must implement."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """
        Determine if this handler can handle the given error.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            True if this handler can handle the error, False otherwise
        """
        raise NotImplementedError
    
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """
        Handle the error and return a user message and retry flag.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            Tuple of (user_message, should_retry)
        """
        raise NotImplementedError
    
    def get_retry_delay(self, error_context: ErrorContext) -> float:
        """
        Calculate delay before retry.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            Delay in seconds before next retry
        """
        raise NotImplementedError


class BaseErrorHandler(ErrorHandlerInterface):
    """Base implementation for error handlers with common functionality."""
    
    def __init__(self, config=None):
        """
        Initialize the handler.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Default implementation always returns False. Subclasses should override."""
        return False
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Default implementation returns a generic message and no retry."""
        return "An error occurred. Please try again later.", False
        
    def get_retry_delay(self, error_context: ErrorContext) -> float:
        """Default implementation uses exponential backoff with jitter."""
        base_delay = 1.0
        max_delay = 30.0
        
        # Exponential backoff
        delay = base_delay * (2 ** error_context.retry_count)
        
        # Add jitter (±20%)
        jitter_factor = 0.2
        jitter = delay * jitter_factor * (random.random() * 2 - 1)
        delay = max(0.1, delay + jitter)
        
        # Cap at max delay
        return min(delay, max_delay)


class NetworkErrorHandler(BaseErrorHandler):
    """Handler for network, timeout, and connectivity errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a network-related error."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type in (
                ErrorType.NETWORK, ErrorType.TIMEOUT, 
                ErrorType.API, ErrorType.RATE_LIMIT
            )
        return str(error_context.error_type).lower() in (
            "network", "timeout", "api", "rate_limit"
        )
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle network errors with appropriate messages and retry logic."""
        # Determine the specific type of network error
        if isinstance(error_context.error_type, ErrorType):
            error_type = error_context.error_type
        else:
            try:
                error_type = ErrorType(str(error_context.error_type).lower())
            except ValueError:
                error_type = ErrorType.NETWORK
        
        # Determine if we should retry
        should_retry = error_context.should_retry
        
        # Different messages based on error type
        if error_type == ErrorType.RATE_LIMIT:
            message = ("I'm receiving too many requests right now. "
                      "Please wait a moment before trying again.")
        elif error_type == ErrorType.TIMEOUT:
            message = ("The request is taking too long to complete. "
                      "This might be due to a slow connection.")
        elif error_type == ErrorType.API:
            message = ("I'm having trouble connecting to the necessary services. "
                      "This might be a temporary issue.")
        else:  # General network error
            message = ("I'm experiencing connection issues. "
                      "Please check your network connection.")
        
        # Add retry information if we're going to retry
        if should_retry:
            retry_delay = self.get_retry_delay(error_context)
            retry_message = f" I'll automatically try again in {int(retry_delay)} seconds."
            message += retry_message
            
        return message, should_retry
        
    def get_retry_delay(self, error_context: ErrorContext) -> float:
        """Calculate retry delay with different base values for rate limiting."""
        # Rate limit errors need longer backoff
        if error_context.error_type == ErrorType.RATE_LIMIT:
            base_delay = 3.0
        else:
            base_delay = 1.0
        
        max_delay = 30.0
        
        # Exponential backoff
        delay = base_delay * (2 ** error_context.retry_count)
        
        # Add jitter (±20%)
        jitter_factor = 0.2
        jitter = delay * jitter_factor * (random.random() * 2 - 1)
        delay = max(0.1, delay + jitter)
        
        # Cap at max delay
        return min(delay, max_delay)


class SpeechRecognitionErrorHandler(BaseErrorHandler):
    """Handler for speech recognition and audio input errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a speech or audio-related error."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type in (
                ErrorType.RECOGNITION, ErrorType.SPEECH,
                ErrorType.AUDIO_DEVICE, ErrorType.USER_INPUT
            )
        return str(error_context.error_type).lower() in (
            "recognition", "speech", "audio_device", "user_input"
        )
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle speech and audio errors with appropriate messages."""
        # Determine the specific type of speech error
        if isinstance(error_context.error_type, ErrorType):
            error_type = error_context.error_type
        else:
            try:
                error_type = ErrorType(str(error_context.error_type).lower())
            except ValueError:
                error_type = ErrorType.RECOGNITION
        
        # Different messages based on error type
        if error_type == ErrorType.RECOGNITION:
            message = "I didn't catch that. Could you please speak more clearly?"
            should_retry = True
            
        elif error_type == ErrorType.AUDIO_DEVICE:
            # Check if this is a permissions issue
            if error_context.exception and "permission" in str(error_context.exception).lower():
                message = "I don't have permission to access your microphone. Please check your settings."
                should_retry = False
            else:
                message = "I'm having trouble with the audio device. Please check that your microphone is working."
                should_retry = error_context.should_retry
                
        elif error_type == ErrorType.SPEECH:
            message = "I'm having trouble with the speech system. This might be temporary."
            should_retry = error_context.should_retry
            
        else:  # User input or other audio issues
            message = "I couldn't understand your input. Please try again."
            should_retry = True
            
        return message, should_retry


class DataErrorHandler(BaseErrorHandler):
    """Handler for data retrieval, processing and storage errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a data-related error."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type in (
                ErrorType.CONTEXT, ErrorType.RETRIEVAL, 
                ErrorType.DATA_FORMAT, ErrorType.DATA_MISSING,
                ErrorType.DATA_CORRUPTION, ErrorType.PDF_PROCESSING,
                ErrorType.EMBEDDING, ErrorType.INDEX
            )
        return str(error_context.error_type).lower() in (
            "context", "retrieval", "data_format", "data_missing",
            "data_corruption", "pdf_processing", "embedding", "index"
        )
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle data errors with appropriate messages."""
        # Determine the specific type of data error
        if isinstance(error_context.error_type, ErrorType):
            error_type = error_context.error_type
        else:
            try:
                error_type = ErrorType(str(error_context.error_type).lower())
            except ValueError:
                error_type = ErrorType.RETRIEVAL
        
        # Different messages based on error type
        if error_type == ErrorType.CONTEXT:
            message = "I couldn't properly understand the context of your question."
            should_retry = False
            
        elif error_type == ErrorType.RETRIEVAL:
            message = "I'm having trouble retrieving the information you requested."
            should_retry = error_context.should_retry
            
        elif error_type in (ErrorType.DATA_FORMAT, ErrorType.DATA_CORRUPTION):
            message = "The data appears to be in an unexpected format or may be corrupted."
            should_retry = False
            
        elif error_type == ErrorType.DATA_MISSING:
            message = "I couldn't find the information you're looking for."
            should_retry = False
            
        elif error_type == ErrorType.PDF_PROCESSING:
            message = "I'm having trouble processing the PDF document. It may be damaged or in an unsupported format."
            should_retry = error_context.should_retry
            
        elif error_type == ErrorType.EMBEDDING:
            message = "I'm having trouble analyzing the document content."
            should_retry = error_context.should_retry
            
        elif error_type == ErrorType.INDEX:
            message = "I'm having trouble accessing the document index."
            should_retry = error_context.should_retry
            
        else:
            message = "I encountered a data processing error."
            should_retry = error_context.should_retry
            
        return message, should_retry


class SystemErrorHandler(BaseErrorHandler):
    """Handler for system and resource errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a system-related error."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type in (
                ErrorType.SYSTEM, ErrorType.MEMORY, ErrorType.PROCESS,
                ErrorType.FILE_SYSTEM, ErrorType.HARDWARE,
                ErrorType.RESOURCE_EXHAUSTED, ErrorType.QUOTA_EXCEEDED,
                ErrorType.LIMIT_REACHED
            )
        return str(error_context.error_type).lower() in (
            "system", "memory", "process", "file_system", "hardware",
            "resource_exhausted", "quota_exceeded", "limit_reached"
        )
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle system errors with appropriate messages."""
        # Determine severity
        is_critical = error_context.severity in (
            ErrorSeverity.CRITICAL, ErrorSeverity.ERROR
        )
        
        # Determine if we should retry
        should_retry = not is_critical and error_context.should_retry
        
        # Create a message based on error type
        if isinstance(error_context.error_type, ErrorType):
            error_type = error_context.error_type
        else:
            try:
                error_type = ErrorType(str(error_context.error_type).lower())
            except ValueError:
                error_type = ErrorType.SYSTEM
        
        if error_type == ErrorType.MEMORY:
            message = "I'm running low on memory. Please try a simpler request."
        elif error_type in (ErrorType.RESOURCE_EXHAUSTED, ErrorType.QUOTA_EXCEEDED, ErrorType.LIMIT_REACHED):
            message = "I've reached a resource limit. Please try again later."
        elif error_type == ErrorType.FILE_SYSTEM:
            message = "I'm having trouble accessing files on the system."
        elif error_type == ErrorType.HARDWARE:
            message = "There appears to be a hardware issue. Please check your system."
        else:  # General system error
            message = "I'm experiencing a system issue. This might be temporary."
        
        # Add additional information for critical errors
        if is_critical:
            message += " This is a critical error that requires attention."
            
        return message, should_retry


class ConfigurationErrorHandler(BaseErrorHandler):
    """Handler for configuration and initialization errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a configuration-related error."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type in (
                ErrorType.CONFIGURATION, ErrorType.ENVIRONMENT,
                ErrorType.DEPENDENCY, ErrorType.INITIALIZATION
            )
        return str(error_context.error_type).lower() in (
            "configuration", "environment", "dependency", "initialization"
        )
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle configuration errors with appropriate messages."""
        # Configuration errors generally shouldn't be retried
        should_retry = False
        
        # Create a message based on error type
        if isinstance(error_context.error_type, ErrorType):
            error_type = error_context.error_type
        else:
            try:
                error_type = ErrorType(str(error_context.error_type).lower())
            except ValueError:
                error_type = ErrorType.CONFIGURATION
        
        if error_type == ErrorType.ENVIRONMENT:
            message = "There's an issue with the environment configuration."
        elif error_type == ErrorType.DEPENDENCY:
            message = "A required dependency is missing or incompatible."
        elif error_type == ErrorType.INITIALIZATION:
            message = "The system failed to initialize properly."
        else:  # General configuration error
            message = "There's a configuration issue that needs to be addressed."
            
        return message, should_retry


class SecurityErrorHandler(BaseErrorHandler):
    """Handler for security, authentication and permission errors."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a security-related error."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type in (
                ErrorType.AUTHENTICATION, ErrorType.PERMISSION,
                ErrorType.ENCRYPTION, ErrorType.TOKEN
            )
        return str(error_context.error_type).lower() in (
            "authentication", "permission", "encryption", "token"
        )
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle security errors with appropriate messages."""
        # Security errors generally shouldn't be retried
        should_retry = False
        
        # Some token errors can be retried
        if (error_context.error_type == ErrorType.TOKEN and 
            "expired" in str(error_context.exception or "").lower()):
            should_retry = error_context.should_retry
        
        # Create a message based on error type
        if isinstance(error_context.error_type, ErrorType):
            error_type = error_context.error_type
        else:
            try:
                error_type = ErrorType(str(error_context.error_type).lower())
            except ValueError:
                error_type = ErrorType.AUTHENTICATION
        
        if error_type == ErrorType.AUTHENTICATION:
            message = "There's an authentication issue. The system may need new credentials."
        elif error_type == ErrorType.PERMISSION:
            message = "You don't have permission to perform this action."
        elif error_type == ErrorType.ENCRYPTION:
            message = "There's an issue with data encryption or security."
        elif error_type == ErrorType.TOKEN and should_retry:
            message = "The security token has expired. I'll try to refresh it."
        else:  # General security error or token error
            message = "There's a security issue that prevents this operation."
            
        return message, should_retry


class FallbackErrorHandler(BaseErrorHandler):
    """Fallback handler for any error types not handled by other handlers."""
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """This handler can handle any error as a last resort."""
        return True
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """General error handling with generic messages."""
        # Determine if we should retry based on the error context
        should_retry = error_context.should_retry
        
        # Basic message
        message = "I encountered an unexpected issue while processing your request."
        
        # Add retry information if appropriate
        if should_retry:
            message += " I'll try again automatically."
            
        return message, should_retry


class CustomErrorHandler(BaseErrorHandler):
    """Configurable handler for custom error types."""
    
    def __init__(self, config=None, error_types=None, message_template=None, retryable=None):
        """
        Initialize the custom handler.
        
        Args:
            config: Optional configuration dictionary
            error_types: List of error types this handler can handle
            message_template: Template string for error messages
            retryable: Whether errors of this type should be retried
        """
        super().__init__(config)
        self.error_types = error_types or []
        self.message_template = message_template or "An error occurred: {message}"
        self.retryable = retryable if retryable is not None else True
        
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can handle the given error type."""
        if isinstance(error_context.error_type, ErrorType):
            return error_context.error_type.value in self.error_types
        return str(error_context.error_type) in self.error_types
        
    def handle(self, error_context: ErrorContext) -> Tuple[str, bool]:
        """Handle the error with the configured message template."""
        # Format the message template with context variables
        context_dict = error_context.to_dict()
        
        try:
            message = self.message_template.format(**context_dict)
        except KeyError:
            # Fallback if template has placeholders we can't fill
            message = f"An error occurred: {error_context.message or 'Unknown error'}"
        
        # Determine if we should retry
        should_retry = self.retryable and error_context.should_retry
        
        return message, should_retry


class ErrorHandler:
    """
    Centralized error handling system for the application.
    
    This class maintains a registry of specialized handlers for different
    error types, providing methods for handling errors, determining retry
    strategies, generating user-friendly messages, and logging errors
    in a consistent manner.
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
        
        # Initialize handler registry
        self.handlers: List[ErrorHandlerInterface] = []
        
        # Load standard handlers
        self._register_standard_handlers()
        
        # Load custom handlers from configuration
        self._register_custom_handlers()
        
        # Always register the fallback handler last
        self.register_handler(FallbackErrorHandler())
        
        # Error tracking for analytics
        self.error_counts = {}
        self.error_history = []
        
        # Maximum errors to keep in history
        self.max_error_history = 100
        
        # Registry of fallback functions
        self.fallbacks = {}
        
        self.logger.info("Initialized error handler with %d handlers", len(self.handlers))
    
    def _register_standard_handlers(self):
        """Register the standard set of error handlers."""
        standard_handlers = [
            NetworkErrorHandler(),
            SpeechRecognitionErrorHandler(),
            DataErrorHandler(),
            SystemErrorHandler(),
            ConfigurationErrorHandler(),
            SecurityErrorHandler()
        ]
        
        for handler in standard_handlers:
            self.register_handler(handler)
    
    def _register_custom_handlers(self):
        """Register custom handlers from configuration."""
        custom_handlers = self.config.get('ERROR_HANDLING', 'CUSTOM_HANDLERS', [])
        
        for handler_config in custom_handlers:
            try:
                # Get handler configuration
                error_types = handler_config.get('error_types', [])
                message_template = handler_config.get('message_template', None)
                retryable = handler_config.get('retryable', None)
                
                # Create and register the handler
                handler = CustomErrorHandler(
                    config=self.config,
                    error_types=error_types,
                    message_template=message_template,
                    retryable=retryable
                )
                self.register_handler(handler)
                
            except Exception as e:
                self.logger.error("Failed to register custom handler: %s", str(e))
    
    def register_handler(self, handler: ErrorHandlerInterface) -> None:
        """
        Register a new error handler.
        
        Args:
            handler: The error handler to register
        """
        self.handlers.append(handler)
    
    def handle_error(self, 
                    error: Exception, 
                    error_type: Union[str, ErrorType] = "general",
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
        # Normalize error type
        if isinstance(error_type, str):
            try:
                error_type = ErrorType(error_type)
            except ValueError:
                error_type = ErrorType.GENERAL
        
        # Create error context
        error_context = self._create_error_context(error, error_type, context)
        
        # Track error for analytics
        self._track_error(error_context)
        
        # Log the error
        self._log_error(error_context)
        
        # Find a handler for this error
        for handler in self.handlers:
            if handler.can_handle(error_context):
                return handler.handle(error_context)
        
        # Should never get here since we have a fallback handler,
        # but just in case...
        return "An unexpected error occurred.", False
    
    def _create_error_context(
        self, error: Exception, error_type: ErrorType, context: Optional[Dict]
    ) -> ErrorContext:
        """Create an ErrorContext object from the error information."""
        # Initialize with basic information
        error_context = ErrorContext(
            error_type=error_type,
            exception=error,
            message=str(error)
        )
        
        # Add context details if provided
        if context:
            error_context.details.update(context)
            
            # Extract known fields from context
            if 'component' in context:
                error_context.component = context['component']
            if 'retry_count' in context:
                error_context.retry_count = context['retry_count']
            if 'max_retries' in context:
                error_context.max_retries = context['max_retries']
            if 'user_id' in context:
                error_context.user_id = context['user_id']
            if 'session_id' in context:
                error_context.session_id = context['session_id']
            if 'request_id' in context:
                error_context.request_id = context['request_id']
                
            # Set severity if provided
            if 'severity' in context:
                try:
                    error_context.severity = ErrorSeverity[context['severity'].upper()]
                except (KeyError, AttributeError):
                    pass
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log the error with appropriate level and details."""
        # Determine log level based on severity
        log_level = logging.INFO
        if error_context.severity == ErrorSeverity.DEBUG:
            log_level = logging.DEBUG
        elif error_context.severity == ErrorSeverity.INFO:
            log_level = logging.INFO
        elif error_context.severity == ErrorSeverity.WARNING:
            log_level = logging.WARNING
        elif error_context.severity == ErrorSeverity.ERROR:
            log_level = logging.ERROR
        elif error_context.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        
        # Format error message
        log_message = (f"Error[{error_context.error_id}] "
                      f"Type: {error_context.error_type.value}, "
                      f"Message: {error_context.message}")
        
        # Add component if available
        if error_context.component != "unknown":
            log_message = f"{log_message}, Component: {error_context.component}"
        
        # Add context details if available
        if error_context.details:
            # Format details as key=value pairs
            details_str = ", ".join(f"{k}={v}" for k, v in error_context.details.items()
                                 if k not in ('traceback', 'stack_trace'))
            log_message = f"{log_message}, Details: {{{details_str}}}"
        
        # Add stack trace for ERROR and CRITICAL
        if log_level >= logging.ERROR and error_context.exception:
            stack_trace = "".join(
                traceback.format_exception(
                    type(error_context.exception),
                    error_context.exception,
                    error_context.exception.__traceback__
                )
            )
            log_message = f"{log_message}\nStack trace:\n{stack_trace}"
        
        # Log the error
        self.logger.log(log_level, log_message)
    
    def _track_error(self, error_context: ErrorContext) -> None:
        """Track error occurrences for analytics."""
        error_type = error_context.error_type.value
        
        # Update counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Add to history
        self.error_history.append(error_context.to_dict())
        
        # Trim history if too long
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
    
    def get_user_message(self, error_type: Union[str, ErrorType], context: Optional[Dict] = None) -> str:
        """
        Get a user-facing error message.
        
        Args:
            error_type: Type of error
            context: Additional context for the message
            
        Returns:
            User-friendly error message
        """
        # Create a minimal error context
        if isinstance(error_type, str):
            try:
                error_type = ErrorType(error_type)
            except ValueError:
                error_type = ErrorType.GENERAL
        
        error_context = ErrorContext(
            error_type=error_type,
            details=context or {}
        )
        
        # Find a handler and get just the message
        for handler in self.handlers:
            if handler.can_handle(error_context):
                message, _ = handler.handle(error_context)
                return message
        
        # Fallback message
        return "I encountered an unexpected issue while processing your request."
    
    def get_retry_delay(self, error_context: ErrorContext) -> float:
        """
        Calculate appropriate retry delay based on error context.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            Delay in seconds before next retry
        """
        # Find a handler for this error type
        for handler in self.handlers:
            if handler.can_handle(error_context):
                return handler.get_retry_delay(error_context)
        
        # Default delay with exponential backoff
        base_delay = 1.0
        max_delay = 30.0
        
        # Exponential backoff
        delay = base_delay * (2 ** error_context.retry_count)
        
        # Add jitter (±20%)
        jitter_factor = 0.2
        jitter = delay * jitter_factor * (random.random() * 2 - 1)
        delay = max(0.1, delay + jitter)
        
        # Cap at max delay
        return min(delay, max_delay)
    
    def register_fallback(self, error_type: Union[str, ErrorType], fallback_fn: Callable) -> None:
        """
        Register a fallback function for an error type.
        
        Args:
            error_type: Type of error
            fallback_fn: Function to call for fallback handling
        """
        # Normalize error type
        if isinstance(error_type, str):
            try:
                error_type = ErrorType(error_type)
            except ValueError:
                error_type = ErrorType.GENERAL
                
        # Register the fallback
        self.fallbacks[error_type] = fallback_fn
        self.logger.debug(f"Registered fallback for error type: {error_type.value}")
    
    def execute_with_fallback(self, 
                             func: Callable, 
                             args: Optional[List] = None, 
                             kwargs: Optional[Dict] = None, 
                             error_types: Optional[List[Union[str, ErrorType]]] = None, 
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
            
        # Normalize error types
        normalized_types = []
        if error_types:
            for et in error_types:
                if isinstance(et, str):
                    try:
                        normalized_types.append(ErrorType(et))
                    except ValueError:
                        normalized_types.append(ErrorType.GENERAL)
                else:
                    normalized_types.append(et)
        else:
            # If no specific error types provided, handle all error types
            normalized_types = list(ErrorType)
        
        # Get max retries from config
        max_retries = int(self.config.get('ERROR_HANDLING', 'MAX_RETRY_ATTEMPTS', 3))
        
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                # Execute the function
                return func(*args, **kwargs)
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                # Determine error type based on exception
                error_type = self._determine_error_type(e)
                
                # Skip handling if this error type isn't in our list
                if error_type not in normalized_types:
                    raise
                
                # Create error context
                context = {
                    'retry_count': attempt - 1,  # Current attempt before increment
                    'max_retries': max_retries,
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                error_context = self._create_error_context(e, error_type, context)
                
                # Get error handling info
                user_message, should_retry = self.handle_error(e, error_type, context)
                
                # If we shouldn't retry, or this is our last attempt, use fallback
                if not should_retry or attempt >= max_retries:
                    # Try registered fallback first
                    if error_type in self.fallbacks:
                        return self.fallbacks[error_type](e, *args, **kwargs)
                    
                    # Then try provided fallback
                    if fallback is not None:
                        return fallback(e, *args, **kwargs)
                    
                    # If no fallback available, re-raise the exception
                    raise
                
                # Retry with delay
                retry_delay = self.get_retry_delay(error_context)
                self.logger.info(
                    f"Retrying {func.__name__} after {retry_delay:.2f}s delay "
                    f"(attempt {attempt}/{max_retries})"
                )
                time.sleep(retry_delay)
        
        # Should never get here, but just in case
        if last_error:
            raise last_error
        raise Exception(f"Failed to execute {func.__name__} after {max_retries} attempts")
    
    def with_error_handling(self, 
                           error_types: Optional[List[Union[str, ErrorType]]] = None,
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
    
    def _determine_error_type(self, error: Exception) -> ErrorType:
        """Determine the error type based on the exception class and message."""
        error_class = error.__class__.__name__
        error_message = str(error).lower()
        
        # Check exception class first
        
        # Network and API errors
        if "Timeout" in error_class or "TimeoutError" in error_class:
            return ErrorType.TIMEOUT
            
        if "Connection" in error_class or "Network" in error_class:
            return ErrorType.NETWORK
            
        if "RateLimit" in error_class or "TooManyRequests" in error_class:
            return ErrorType.RATE_LIMIT
            
        # Authentication and security errors
        if "Auth" in error_class or "Credentials" in error_class or "Token" in error_class:
            return ErrorType.AUTHENTICATION
            
        if "Permission" in error_class or "Access" in error_class:
            return ErrorType.PERMISSION
            
        # Data errors
        if "NotFound" in error_class or "Missing" in error_class:
            return ErrorType.DATA_MISSING
            
        if "Format" in error_class or "Parse" in error_class or "Json" in error_class:
            return ErrorType.DATA_FORMAT
            
        if "Corrupt" in error_class:
            return ErrorType.DATA_CORRUPTION
            
        # Resource errors
        if "Memory" in error_class or "OutOf" in error_class:
            return ErrorType.MEMORY
            
        if "Quota" in error_class or "Limit" in error_class:
            return ErrorType.RESOURCE_EXHAUSTED
            
        # Now check error message content
        
        # Network and API errors
        if "rate limit" in error_message or "too many requests" in error_message:
            return ErrorType.RATE_LIMIT
            
        if "timeout" in error_message or "timed out" in error_message:
            return ErrorType.TIMEOUT
            
        if "network" in error_message or "connection" in error_message:
            return ErrorType.NETWORK
            
        if "api" in error_message:
            return ErrorType.API
        
        # Data errors
        if "context" in error_message:
            return ErrorType.CONTEXT
            
        if "retriev" in error_message or "find" in error_message or "search" in error_message:
            return ErrorType.RETRIEVAL
            
        if "pdf" in error_message or "document" in error_message:
            return ErrorType.PDF_PROCESSING
            
        if "embedding" in error_message or "vector" in error_message:
            return ErrorType.EMBEDDING
            
        if "index" in error_message:
            return ErrorType.INDEX
        
        # Speech and audio errors
        if "speech" in error_message or "voice" in error_message:
            return ErrorType.SPEECH
            
        if "recogni" in error_message or "listen" in error_message or "hear" in error_message:
            return ErrorType.RECOGNITION
            
        if "audio" in error_message or "microphone" in error_message or "sound" in error_message:
            return ErrorType.AUDIO_DEVICE
            
        if "text to speech" in error_message or "tts" in error_message:
            return ErrorType.TEXT_TO_SPEECH
        
        # Resource errors
        if "memory" in error_message or "out of" in error_message:
            return ErrorType.MEMORY
            
        if "limit" in error_message or "quota" in error_message or "exceed" in error_message:
            return ErrorType.RESOURCE_EXHAUSTED
            
        # System errors
        if "file" in error_message or "directory" in error_message or "path" in error_message:
            return ErrorType.FILE_SYSTEM
            
        if "hardware" in error_message or "device" in error_message:
            return ErrorType.HARDWARE
            
        if "system" in error_message:
            return ErrorType.SYSTEM
            
        # Configuration errors
        if "config" in error_message or "setting" in error_message:
            return ErrorType.CONFIGURATION
            
        if "environment" in error_message or "env" in error_message:
            return ErrorType.ENVIRONMENT
            
        if "dependency" in error_message or "module" in error_message or "import" in error_message:
            return ErrorType.DEPENDENCY
            
        if "init" in error_message or "start" in error_message:
            return ErrorType.INITIALIZATION
            
        # Default to general error type
        return ErrorType.GENERAL
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get statistics on error occurrences.
        
        Returns:
            Dictionary of error statistics
        """
        # Basic counts by error type
        counts_by_type = dict(self.error_counts)
        
        # Counts by category
        counts_by_category = {}
        for error_type, count in self.error_counts.items():
            try:
                category = ErrorType.get_category(error_type).name
                if category not in counts_by_category:
                    counts_by_category[category] = 0
                counts_by_category[category] += count
            except Exception:
                # Skip if category can't be determined
                pass
        
        # Recent errors
        recent_errors = self.error_history[-10:] if self.error_history else []
        
        return {
            "counts_by_type": counts_by_type,
            "counts_by_category": counts_by_category,
            "error_count": sum(self.error_counts.values()),
            "recent_errors": recent_errors,
            "timestamp": time.time()
        }
    
    def reset_error_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts = {}
        self.error_history = []
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of the error handling system.
        
        Returns:
            Dictionary with health status information
        """
        error_stats = self.get_error_statistics()
        
        # Calculate severity based on error counts
        severity = "healthy"
        total_errors = error_stats["error_count"]
        
        if total_errors > 100:
            severity = "critical"
        elif total_errors > 50:
            severity = "warning"
        elif total_errors > 10:
            severity = "degraded"
        
        # Get top error categories
        categories = sorted(
            error_stats["counts_by_category"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_categories = dict(categories[:3]) if categories else {}
        
        return {
            "status": severity,
            "error_count": total_errors,
            "top_error_categories": top_categories,
            "handler_count": len(self.handlers),
            "timestamp": time.time()
        }


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
    
    error_handler.register_fallback(ErrorType.GENERAL, test_fallback)
    
    # Test with direct error handling
    try:
        message, retry = error_handler.handle_error(
            Exception("Network connection failed"),
            ErrorType.NETWORK
        )
        print(f"Message: {message}")
        print(f"Should retry: {retry}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test with different error types
    for error_type in [
        ErrorType.TIMEOUT, 
        ErrorType.RECOGNITION,
        ErrorType.PDF_PROCESSING,
        ErrorType.MEMORY,
        ErrorType.AUTHENTICATION
    ]:
        message, retry = error_handler.handle_error(
            Exception(f"Test {error_type.value} error"),
            error_type
        )
        print(f"{error_type.value}: {message} (retry={retry})")
    
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
    
    # Print error statistics and health
    print(f"Error statistics: {error_handler.get_error_statistics()}")
    print(f"Health status: {error_handler.get_health_status()}")