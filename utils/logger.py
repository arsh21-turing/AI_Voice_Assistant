import logging
import os
import json
import time
import datetime
import traceback
import platform
import shutil
import gzip
import tarfile
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class VoiceAssistantLogger:
    """Comprehensive logging system for the voice assistant application."""
    
    def __init__(self, log_dir: str = "logs", app_name: str = "voice_assistant", 
                 log_level: int = logging.INFO, enable_console: bool = True,
                 max_log_files: int = 10, max_file_size_mb: int = 10,
                 archive_dir: str = None, rotation_interval_days: int = 7):
        """Initialize the logging system.
        
        Args:
            log_dir: Directory to store log files
            app_name: Name of the application for log file prefixes
            log_level: Default logging level
            enable_console: Whether to output logs to console
            max_log_files: Maximum number of log files to keep (rotation)
            max_file_size_mb: Maximum size of each log file in MB
            archive_dir: Directory to store archived log files (defaults to log_dir/archives)
            rotation_interval_days: Interval in days for log rotation based on age
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.start_time = time.time()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_log_files = max_log_files
        self.rotation_interval_days = rotation_interval_days
        
        # Set up archive directory
        if archive_dir is None:
            self.archive_dir = self.log_dir / "archives"
        else:
            self.archive_dir = Path(archive_dir)
            
        # Create log and archive directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Create different loggers for different log types
        self.loggers = {}
        self.log_files = {}
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
            
            # Store the log file path
            self.log_files[log_type] = log_file
            
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
        
        # Perform initial log rotation to clean up old files
        self.rotate_logs()
        
        # Log system start
        self.log_system("startup", f"===== {app_name} logging started =====", {
            "session_id": self.session_id,
            "log_level": logging.getLevelName(log_level),
            "python_version": platform.python_version(),
            "log_rotation": {
                "max_size_mb": max_file_size_mb,
                "max_files": max_log_files,
                "rotation_interval_days": rotation_interval_days
            }
        })
    
    def _rotate_logs(self, max_files: int):
        """Basic log rotation to keep only the most recent ones (legacy method).
        
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
    
    def rotate_logs(self):
        """Intelligent log rotation that compresses old logs into dated archives.
        
        This method:
        1. Checks for log files exceeding size threshold
        2. Compresses log files older than the rotation interval
        3. Manages archived log files by date
        4. Ensures only the specified maximum number of files are kept
        """
        try:
            self.log_system("log_rotation", "Starting log rotation", {
                "max_size_bytes": self.max_file_size_bytes,
                "max_files": self.max_log_files,
                "rotation_interval_days": self.rotation_interval_days
            })
            
            # Get current time for age-based checks
            now = time.time()
            rotation_threshold = now - (self.rotation_interval_days * 86400)
            
            # Check all log files in the log directory
            rotated_count = 0
            size_rotated = 0
            age_rotated = 0
            
            for log_file in self.log_dir.glob(f"{self.app_name}_*.log"):
                # Skip current session log files
                if self.session_id in log_file.name:
                    continue
                    
                # Check if file exceeds size threshold
                file_size = log_file.stat().st_size
                file_mtime = log_file.stat().st_mtime
                
                # Determine if file should be rotated (by size or age)
                rotate_by_size = file_size > self.max_file_size_bytes
                rotate_by_age = file_mtime < rotation_threshold
                
                if rotate_by_size or rotate_by_age:
                    # Extract date from filename or use modification time
                    date_match = re.search(r'_(\d{8})_', log_file.name)
                    if date_match:
                        date_str = date_match.group(1)
                    else:
                        # Use file modification time if date not in filename
                        date_str = datetime.datetime.fromtimestamp(file_mtime).strftime("%Y%m%d")
                    
                    # Compress the log file
                    if self._compress_log_file(log_file, date_str):
                        rotated_count += 1
                        
                        if rotate_by_size:
                            size_rotated += 1
                        if rotate_by_age:
                            age_rotated += 1
            
            # Perform cleanup of archived logs (keep only max_log_files archives per type)
            self._cleanup_archives()
            
            self.log_system("log_rotation", "Log rotation completed", {
                "rotated_count": rotated_count,
                "rotated_by_size": size_rotated,
                "rotated_by_age": age_rotated
            })
            
        except Exception as e:
            self.log_error(e, "log_rotation", {
                "error": str(e)
            })
    
    def _compress_log_file(self, log_file: Path, date_str: str) -> bool:
        """Compress a log file into the archive directory.
        
        Args:
            log_file: Path to the log file
            date_str: Date string for archive naming
            
        Returns:
            True if compression successful, False otherwise
        """
        try:
            # Extract log type from filename
            for log_type in self.loggers.keys():
                if f"_{log_type}_" in log_file.name:
                    break
            else:
                log_type = "unknown"
                
            # Create archive filename
            archive_name = f"{self.app_name}_{log_type}_{date_str}.log.gz"
            archive_path = self.archive_dir / archive_name
            
            # Compress with gzip
            with open(log_file, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Verify compression was successful
            if archive_path.exists() and archive_path.stat().st_size > 0:
                # Remove the original file
                log_file.unlink()
                return True
            else:
                # Something went wrong, keep the original
                self.log_error(
                    "Compression failed", 
                    "log_rotation", 
                    {"file": str(log_file)}
                )
                return False
                
        except Exception as e:
            self.log_error(
                e,
                "log_rotation",
                {"file": str(log_file), "operation": "compress"}
            )
            return False
    
    def _cleanup_archives(self):
        """Clean up archived log files, keeping only the most recent ones."""
        try:
            # Group archives by log type
            archive_groups = {}
            for archive_file in self.archive_dir.glob(f"{self.app_name}_*.log.gz"):
                # Extract log type from filename
                for log_type in self.loggers.keys():
                    if f"_{log_type}_" in archive_file.name:
                        if log_type not in archive_groups:
                            archive_groups[log_type] = []
                        archive_groups[log_type].append(archive_file)
                        break
            
            # For each group, keep only max_log_files archives (the newest ones)
            deleted_count = 0
            for log_type, archives in archive_groups.items():
                # Sort by modification time (newest last)
                sorted_archives = sorted(archives, key=os.path.getmtime)
                
                # If we have more than max_files, delete the oldest ones
                if len(sorted_archives) > self.max_log_files:
                    for old_archive in sorted_archives[:-self.max_log_files]:
                        try:
                            old_archive.unlink()
                            deleted_count += 1
                        except Exception as e:
                            self.log_error(
                                e,
                                "log_rotation",
                                {"file": str(old_archive), "operation": "delete_archive"}
                            )
            
            self.log_system("archive_cleanup", f"Deleted {deleted_count} old archive(s)")
            
            # Look for very old archives (3x rotation_interval) and create consolidated monthly archives
            self._consolidate_old_archives()
                
        except Exception as e:
            self.log_error(
                e,
                "log_rotation",
                {"operation": "cleanup_archives"}
            )
    
    def _consolidate_old_archives(self):
        """Consolidate very old archives into monthly archives."""
        try:
            # Get threshold for old archives (3x rotation interval)
            now = time.time()
            consolidation_threshold = now - (self.rotation_interval_days * 3 * 86400)
            
            # Find old archives grouped by month
            monthly_archives = {}
            for archive_file in self.archive_dir.glob(f"{self.app_name}_*.log.gz"):
                # Skip if file is too recent
                if archive_file.stat().st_mtime > consolidation_threshold:
                    continue
                
                # Extract date from filename
                date_match = re.search(r'_(\d{6})\d{2}', archive_file.name)
                if date_match:
                    # Get year and month (YYYYMM)
                    year_month = date_match.group(1)
                    if year_month not in monthly_archives:
                        monthly_archives[year_month] = []
                    monthly_archives[year_month].append(archive_file)
            
            # For each month with multiple archives, consolidate into a single tarball
            consolidated_count = 0
            for year_month, archives in monthly_archives.items():
                # Only consolidate if we have multiple archives for the month
                if len(archives) > 1:
                    # Create monthly archive filename
                    monthly_archive = self.archive_dir / f"{self.app_name}_logs_{year_month}.tar.gz"
                    
                    # Skip if it already exists
                    if monthly_archive.exists():
                        continue
                    
                    # Create the tar archive
                    with tarfile.open(monthly_archive, "w:gz") as tar:
                        for archive in archives:
                            tar.add(archive, arcname=archive.name)
                    
                    # Verify tarball was created successfully
                    if monthly_archive.exists() and monthly_archive.stat().st_size > 0:
                        # Remove individual archives
                        for archive in archives:
                            try:
                                archive.unlink()
                                consolidated_count += 1
                            except Exception as e:
                                self.log_error(
                                    e,
                                    "log_consolidation",
                                    {"file": str(archive)}
                                )
            
            if consolidated_count > 0:
                self.log_system("log_consolidation", f"Consolidated {consolidated_count} archives into monthly archives")
                
        except Exception as e:
            self.log_error(
                e,
                "log_rotation",
                {"operation": "consolidate_archives"}
            )
    
    def check_and_rotate_current_logs(self):
        """Check current session's log files and rotate if they exceed size threshold."""
        for log_type, log_file in self.log_files.items():
            try:
                if log_file.exists() and log_file.stat().st_size > self.max_file_size_bytes:
                    # Log that we're rotating the current log file
                    self.log_system("log_rotation", f"Rotating current {log_type} log due to size", {
                        "file": str(log_file),
                        "size_bytes": log_file.stat().st_size,
                        "threshold_bytes": self.max_file_size_bytes
                    })
                    
                    # Close current handlers
                    logger = self.loggers[log_type]
                    handlers = list(logger.handlers)
                    for handler in handlers:
                        if isinstance(handler, logging.FileHandler):
                            handler.close()
                            logger.removeHandler(handler)
                    
                    # Create archive of current log
                    date_str = self.session_id.split('_')[0]
                    self._compress_log_file(log_file, date_str)
                    
                    # Create a new log file
                    new_log_file = self.log_dir / f"{self.app_name}_{log_type}_{self.session_id}_cont.log"
                    file_handler = logging.FileHandler(new_log_file, encoding='utf-8')
                    file_handler.setFormatter(logging.Formatter(
                        '%(asctime)s [%(levelname)s] %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S'
                    ))
                    logger.addHandler(file_handler)
                    
                    # Update the log file reference
                    self.log_files[log_type] = new_log_file
                    
                    # Log to the new file that we rotated
                    self.log_system("log_rotation", f"Created new {log_type} log file after rotation", {
                        "previous_file": str(log_file),
                        "new_file": str(new_log_file)
                    })
                    
            except Exception as e:
                # Log to a different logger to avoid recursion
                system_logger = self.loggers.get("system")
                if system_logger:
                    system_logger.error(f"Error rotating current log file: {str(e)}")
    
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
        
        # Check if we need to rotate the log file after this entry
        self.check_and_rotate_current_logs()
    
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
        
        # Check if we need to rotate the log file after this entry
        self.check_and_rotate_current_logs()
    
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
        
        # Check if we need to rotate the log file after this entry
        self.check_and_rotate_current_logs()
    
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
        
        # Don't check rotation on system logs to avoid potential recursion
        # when called from rotation methods
    
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
        
        # Check if we need to rotate logs periodically based on total log volume
        if stats.get("total_entries", 0) % 100 == 0:
            self.rotate_logs()

    def time_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to time an operation and log its performance.
        
        Args:
            operation: Name of the operation
            metadata: Additional metadata about the operation
        
        Returns:
            Context manager that times the operation
        """
        return OperationTimer(self, operation, metadata)
    
    def cleanup(self):
        """Perform final cleanup operations before application exit."""
        # Flush all loggers
        for log_type, logger in self.loggers.items():
            for handler in logger.handlers:
                handler.flush()
                
        # Perform one final log rotation
        self.rotate_logs()
        
        # Log shutdown
        self.log_system("shutdown", f"===== {self.app_name} logging stopped =====", {
            "session_duration": time.time() - self.start_time
        })


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