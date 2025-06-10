import logging
import os
import json
import time
import datetime
import traceback
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class VoiceAssistantLogger:
    """Comprehensive logging system for the voice assistant application."""
    
    def __init__(self, log_dir: str = "logs", app_name: str = "voice_assistant", 
                 log_level: int = logging.INFO, enable_console: bool = True,
                 max_log_files: int = 10, max_file_size_mb: int = 10):
        """Initialize the logging system.
        
        Args:
            log_dir: Directory to store log files
            app_name: Name of the application for log file prefixes
            log_level: Default logging level
            enable_console: Whether to output logs to console
            max_log_files: Maximum number of log files to keep (rotation)
            max_file_size_mb: Maximum size of each log file in MB
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.start_time = time.time()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create different loggers for different log types
        self.loggers = {}
        log_types = {
            "interaction": "User interactions with the system",
            "error": "Errors and exceptions",
            "performance": "Performance metrics and timings",
            "system": "System events and operations"
        }
        
        # Configure each logger
        for log_type, description in log_types.items():
            logger = logging.getLogger(f"{app_name}.{log_type}")
            logger.setLevel(log_level)
            logger.propagate = False  # Don't propagate to root logger
            
            # Remove existing handlers if any
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # File handler with rotation
            log_file = self.log_dir / f"{app_name}_{log_type}_{self.session_id}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(file_handler)
            
            # Console handler if enabled
            if enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S'
                ))
                # Only show errors and higher in console by default
                console_handler.setLevel(logging.ERROR)
                logger.addHandler(console_handler)
            
            # Store the logger
            self.loggers[log_type] = logger
        
        # Manage log rotation
        self._rotate_logs(max_log_files)
        
        # Log system start
        self.log_system("startup", f"===== {app_name} logging started =====", {
            "session_id": self.session_id,
            "log_level": logging.getLevelName(log_level),
            "python_version": platform.python_version()
        })
    
    def _rotate_logs(self, max_files: int):
        """Rotate log files to keep only the most recent ones.
        
        Args:
            max_files: Maximum number of log files to keep per type
        """
        for log_type in self.loggers.keys():
            pattern = f"{self.app_name}_{log_type}_*.log"
            log_files = sorted(
                self.log_dir.glob(pattern),
                key=os.path.getmtime
            )
            
            # If we have more than max_files, delete the oldest ones
            if len(log_files) > max_files:
                for old_file in log_files[:-max_files]:
                    try:
                        old_file.unlink()
                    except Exception as e:
                        print(f"Error removing old log file {old_file}: {str(e)}")
    
    def _format_log_data(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Format log data as a structured string.
        
        Args:
            message: Log message
            metadata: Additional metadata to include
            
        Returns:
            Formatted log string
        """
        log_data = {"message": message, "timestamp": time.time()}
        
        if metadata:
            log_data.update(metadata)
            
        try:
            return json.dumps(log_data)
        except Exception:
            # If JSON serialization fails, fall back to string representation
            return f"{message} | Metadata: {str(metadata)}"
    
    def log_interaction(self, interaction_type: str, query: str, response: str, 
                        metadata: Optional[Dict[str, Any]] = None, level: int = logging.INFO):
        """Log user interaction with the system.
        
        Args:
            interaction_type: Type of interaction (query, wake_word, etc.)
            query: User's query or input
            response: System's response
            metadata: Additional metadata about the interaction
            level: Logging level for this event
        """
        logger = self.loggers["interaction"]
        
        # Create basic metadata if none provided
        if metadata is None:
            metadata = {}
            
        # Add standard information
        metadata.update({
            "type": interaction_type,
            "query": query,
            "response_length": len(response),
            "session_id": self.session_id
        })
        
        # Add response summary but don't log full long responses to keep logs manageable
        if len(response) > 200:
            metadata["response_summary"] = response[:197] + "..."
        else:
            metadata["response"] = response
        
        log_message = f"INTERACTION [{interaction_type}]: {query}"
        logger.log(level, self._format_log_data(log_message, metadata))
    
    def log_error(self, error: Union[str, Exception], context: str, 
                  metadata: Optional[Dict[str, Any]] = None, level: int = logging.ERROR):
        """Log an error or exception.
        
        Args:
            error: Error message or exception object
            context: Where the error occurred
            metadata: Additional metadata about the error
            level: Logging level for this error
        """
        logger = self.loggers["error"]
        
        # Create basic metadata if none provided
        if metadata is None:
            metadata = {}
            
        # Add standard information
        metadata.update({
            "context": context,
            "session_id": self.session_id
        })
        
        # Extract error details
        if isinstance(error, Exception):
            error_type = type(error).__name__
            error_message = str(error)
            metadata["error_type"] = error_type
            metadata["traceback"] = traceback.format_exc()
            log_message = f"ERROR [{context}]: {error_type} - {error_message}"
        else:
            log_message = f"ERROR [{context}]: {error}"
        
        logger.log(level, self._format_log_data(log_message, metadata))
    
    def log_performance(self, operation: str, duration: float, 
                        metadata: Optional[Dict[str, Any]] = None, level: int = logging.DEBUG):
        """Log performance metric for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration of the operation in seconds
            metadata: Additional metadata about the operation
            level: Logging level for this metric
        """
        logger = self.loggers["performance"]
        
        # Create basic metadata if none provided
        if metadata is None:
            metadata = {}
            
        # Add standard information
        metadata.update({
            "operation": operation,
            "duration": duration,
            "duration_ms": int(duration * 1000),
            "session_id": self.session_id
        })
        
        log_message = f"PERFORMANCE [{operation}]: {duration:.4f}s"
        logger.log(level, self._format_log_data(log_message, metadata))
    
    def log_system(self, event_type: str, message: str, 
                   metadata: Optional[Dict[str, Any]] = None, level: int = logging.INFO):
        """Log system event.
        
        Args:
            event_type: Type of system event
            message: Event description
            metadata: Additional metadata about the event
            level: Logging level for this event
        """
        logger = self.loggers["system"]
        
        # Create basic metadata if none provided
        if metadata is None:
            metadata = {}
            
        # Add standard information
        metadata.update({
            "type": event_type,
            "session_id": self.session_id,
            "uptime": time.time() - self.start_time
        })
        
        log_message = f"SYSTEM [{event_type}]: {message}"
        logger.log(level, self._format_log_data(log_message, metadata))
    
    def log_cache_stats(self, cache_name: str, stats: Dict[str, Any], level: int = logging.DEBUG):
        """Log cache statistics.
        
        Args:
            cache_name: Name of the cache
            stats: Cache statistics
            level: Logging level for this event
        """
        self.log_performance(f"cache_{cache_name}", 0, {
            "cache_name": cache_name,
            "stats": stats
        }, level)

    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to time an operation and log its performance.
        
        Args:
            operation: Name of the operation
            metadata: Additional metadata about the operation
        
        Returns:
            Context manager that times the operation
        """
        return OperationTimer(self, operation, metadata)


class OperationTimer:
    """Context manager for timing operations and logging their performance."""
    
    def __init__(self, logger, operation, metadata=None):
        self.logger = logger
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        # Add exception info to metadata if an exception occurred
        if exc_type is not None:
            self.metadata["exception"] = {
                "type": exc_type.__name__,
                "message": str(exc_val)
            }
            
        self.logger.log_performance(self.operation, duration, self.metadata)
        
        # Don't suppress exceptions
        return False 