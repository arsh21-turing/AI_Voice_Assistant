import enum
import time
import threading
import requests
import queue
import pyaudio
import wave
import os
from typing import Dict, Any, List, Tuple, Optional, Union

class SystemStatus(enum.Enum):
    """Enum representing the status of a system component."""
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"
    UNKNOWN = "UNKNOWN"

class SystemComponent(enum.Enum):
    """Enum representing different system components."""
    MICROPHONE = "Microphone"
    API = "API Connection"
    AUDIO_OUTPUT = "Audio Output"
    RAG = "RAG System" 
    EMBEDDING = "Embedding Model"
    INDEX = "FAISS Index"
    OVERALL = "Overall System"

class SystemMonitor:
    """Class for monitoring the health and status of various system components."""
    
    def __init__(self, voice_assistant: 'VoiceAssistant'):
        """Initialize the system monitor.
        
        Args:
            voice_assistant: Reference to the main VoiceAssistant instance
        """
        self.assistant = voice_assistant
        
        # Initialize component statuses
        self.component_status = {
            SystemComponent.MICROPHONE: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}},
            SystemComponent.API: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}},
            SystemComponent.AUDIO_OUTPUT: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}},
            SystemComponent.RAG: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}},
            SystemComponent.EMBEDDING: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}},
            SystemComponent.INDEX: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}},
            SystemComponent.OVERALL: {"status": SystemStatus.UNKNOWN, "last_check": 0, "message": "", "details": {}}
        }
        
        # Settings
        self.check_interval = 60  # Check components every 60 seconds
        self.status_decay_time = 300  # Status is considered stale after 5 minutes
        
        # Create a notification queue for messages to the user
        self.notification_queue = queue.Queue()
        
        # Flag to control the monitoring thread
        self.monitor_active = True
        
        # Initialize the monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            name="SystemMonitorThread",
            daemon=True
        )
        
        # Start the monitoring thread
        self.monitor_thread.start()
        
        # Initial check of all systems
        self.check_all_systems()
        
        # Log initialization
        print("System monitor initialized and running")
    
    def _monitoring_loop(self):
        """Background thread that periodically checks system components."""
        while self.monitor_active:
            try:
                # Sleep first to avoid checking right after initialization
                time.sleep(self.check_interval)
                
                # Check if it's time to refresh status
                if time.time() - self.component_status[SystemComponent.OVERALL]["last_check"] >= self.check_interval:
                    self.check_all_systems()
            except Exception as e:
                # Log error but don't stop the monitoring loop
                print(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Sleep briefly to avoid tight error loops
    
    def check_all_systems(self):
        """Check status of all system components."""
        print("\n--- Checking system status ---")
        
        # Check each component individually
        self.check_microphone()
        self.check_api_status()
        self.check_audio_output()
        self.check_rag_system()
        
        # Update overall system status
        self._update_overall_status()
        
        print(f"Overall system status: {self.component_status[SystemComponent.OVERALL]['status'].value}")
    
    def check_microphone(self):
        """Check if microphone is available and functioning."""
        print("Checking microphone...")
        
        try:
            # Attempt to initialize and use the microphone
            p = pyaudio.PyAudio()
            
            # Check if any input devices are available
            input_devices = [
                i for i in range(p.get_device_count())
                if p.get_device_info_by_index(i)["maxInputChannels"] > 0
            ]
            
            p.terminate()
            
            if not input_devices:
                self._update_component_status(
                    SystemComponent.MICROPHONE,
                    SystemStatus.ERROR,
                    "No microphone found",
                    {"available_devices": 0}
                )
                return False
            
            # If the assistant already has a microphone initialized, check if it's working
            if hasattr(self.assistant, 'microphone'):
                # Check if recognizer is also initialized
                if not hasattr(self.assistant, 'recognizer'):
                    self._update_component_status(
                        SystemComponent.MICROPHONE,
                        SystemStatus.WARNING,
                        "Speech recognizer not initialized",
                        {"available_devices": len(input_devices)}
                    )
                    return False
                
                # Simple connectivity check (doesn't try to recognize anything)
                try:
                    # Just a quick listen to see if the microphone is responding
                    with self.assistant.microphone as source:
                        # Very brief adjustment to avoid long wait
                        self.assistant.recognizer.adjust_for_ambient_noise(source, duration=0.1)
                    
                    self._update_component_status(
                        SystemComponent.MICROPHONE,
                        SystemStatus.OK,
                        "Microphone is functioning",
                        {"available_devices": len(input_devices)}
                    )
                    return True
                    
                except Exception as e:
                    self._update_component_status(
                        SystemComponent.MICROPHONE,
                        SystemStatus.ERROR,
                        f"Error accessing microphone: {e.__class__.__name__}",
                        {"error": str(e), "available_devices": len(input_devices)}
                    )
                    return False
            else:
                # No microphone initialized yet, but devices are available
                self._update_component_status(
                    SystemComponent.MICROPHONE,
                    SystemStatus.WARNING,
                    "Microphone devices available but not initialized",
                    {"available_devices": len(input_devices)}
                )
                return False
            
        except Exception as e:
            self._update_component_status(
                SystemComponent.MICROPHONE,
                SystemStatus.ERROR,
                f"Error checking microphone: {e.__class__.__name__}",
                {"error": str(e)}
            )
            return False
    
    def check_api_status(self):
        """Check if API services are accessible."""
        print("Checking API connection...")
        
        try:
            # Check if client is initialized
            if not hasattr(self.assistant, 'client') or self.assistant.client is None:
                self._update_component_status(
                    SystemComponent.API,
                    SystemStatus.ERROR,
                    "API client not initialized",
                    {}
                )
                return False
            
            # Try a simple API request (with timeout to avoid long waits)
            test_response = self.assistant.client.chat.completions.create(
                model=self.assistant.groq_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5  # 5 second timeout
            )
            
            # If we reach here, the API is responsive
            self._update_component_status(
                SystemComponent.API,
                SystemStatus.OK,
                "API connection established",
                {"api": "Groq", "model": self.assistant.groq_model}
            )
            return True
            
        except requests.exceptions.Timeout:
            self._update_component_status(
                SystemComponent.API,
                SystemStatus.ERROR,
                "API connection timed out",
                {"api": "Groq", "error_type": "timeout"}
            )
            return False
            
        except requests.exceptions.ConnectionError:
            self._update_component_status(
                SystemComponent.API,
                SystemStatus.ERROR,
                "Cannot connect to API server",
                {"api": "Groq", "error_type": "connection"}
            )
            return False
            
        except Exception as e:
            self._update_component_status(
                SystemComponent.API,
                SystemStatus.ERROR,
                f"API error: {e.__class__.__name__}",
                {"api": "Groq", "error": str(e)}
            )
            return False
    
    def check_audio_output(self):
        """Verify audio output is working."""
        print("Checking audio output...")
        
        try:
            # Check if TTS engine is initialized
            if not hasattr(self.assistant, 'engine'):
                self._update_component_status(
                    SystemComponent.AUDIO_OUTPUT,
                    SystemStatus.ERROR,
                    "Speech synthesis engine not initialized",
                    {}
                )
                return False
            
            # Check if voices are available
            if not hasattr(self.assistant, 'voices') or not self.assistant.voices:
                self._update_component_status(
                    SystemComponent.AUDIO_OUTPUT,
                    SystemStatus.WARNING,
                    "No voices available for speech synthesis",
                    {}
                )
                return False
            
            # Check if we can access the audio device (without actually playing)
            # This is a lightweight check as we don't want to speak something
            try:
                # Just verify the engine properties are accessible
                rate = self.assistant.engine.getProperty('rate')
                volume = self.assistant.engine.getProperty('volume')
                
                self._update_component_status(
                    SystemComponent.AUDIO_OUTPUT,
                    SystemStatus.OK,
                    "Audio output system ready",
                    {"voices_available": len(self.assistant.voices)}
                )
                return True
                
            except Exception as e:
                self._update_component_status(
                    SystemComponent.AUDIO_OUTPUT,
                    SystemStatus.ERROR,
                    f"Error accessing audio system: {e.__class__.__name__}",
                    {"error": str(e)}
                )
                return False
                
        except Exception as e:
            self._update_component_status(
                SystemComponent.AUDIO_OUTPUT,
                SystemStatus.ERROR,
                f"Error checking audio output: {e.__class__.__name__}",
                {"error": str(e)}
            )
            return False
    
    def check_rag_system(self):
        """Check if RAG components are working properly."""
        print("Checking RAG system...")
        
        # Check embedding model
        embedding_ok = self._check_embedding_model()
        
        # Check FAISS index 
        index_ok = self._check_faiss_index()
        
        # Overall RAG status depends on both components
        if embedding_ok and index_ok:
            self._update_component_status(
                SystemComponent.RAG,
                SystemStatus.OK,
                "RAG system fully operational",
                {"embedding_ok": True, "index_ok": True}
            )
            return True
        elif not embedding_ok and not index_ok:
            self._update_component_status(
                SystemComponent.RAG,
                SystemStatus.ERROR,
                "RAG system completely unavailable",
                {"embedding_ok": False, "index_ok": False}
            )
            return False
        else:
            self._update_component_status(
                SystemComponent.RAG,
                SystemStatus.WARNING,
                "RAG system partially operational",
                {"embedding_ok": embedding_ok, "index_ok": index_ok}
            )
            return False
    
    def _check_embedding_model(self):
        """Check if the embedding model is loaded and functioning."""
        try:
            # Check if embedding model exists
            if not hasattr(self.assistant, 'embedding_model') or self.assistant.embedding_model is None:
                self._update_component_status(
                    SystemComponent.EMBEDDING,
                    SystemStatus.ERROR,
                    "Embedding model not initialized",
                    {}
                )
                return False
            
            # Check if we can generate an embedding (quick test)
            test_embedding = self.assistant.embedding_model.encode("test", convert_to_tensor=False)
            
            if test_embedding is not None and len(test_embedding) > 0:
                self._update_component_status(
                    SystemComponent.EMBEDDING,
                    SystemStatus.OK,
                    "Embedding model operational",
                    {"embedding_dimension": len(test_embedding)}
                )
                return True
            else:
                self._update_component_status(
                    SystemComponent.EMBEDDING,
                    SystemStatus.ERROR,
                    "Embedding model not returning valid embeddings",
                    {}
                )
                return False
                
        except Exception as e:
            self._update_component_status(
                SystemComponent.EMBEDDING,
                SystemStatus.ERROR,
                f"Embedding model error: {e.__class__.__name__}",
                {"error": str(e)}
            )
            return False
    
    def _check_faiss_index(self):
        """Check if the FAISS index is loaded and usable."""
        try:
            # Check if index exists
            if not hasattr(self.assistant, 'index') or self.assistant.index is None:
                self._update_component_status(
                    SystemComponent.INDEX,
                    SystemStatus.ERROR,
                    "FAISS index not initialized",
                    {}
                )
                return False
            
            # Check if index contains vectors
            if self.assistant.index.ntotal == 0:
                self._update_component_status(
                    SystemComponent.INDEX,
                    SystemStatus.WARNING,
                    "FAISS index is empty (no documents indexed)",
                    {"vectors": 0}
                )
                return False
            
            # Check if we can access metadata
            if not hasattr(self.assistant, 'metadata') or not self.assistant.metadata:
                self._update_component_status(
                    SystemComponent.INDEX,
                    SystemStatus.WARNING,
                    "FAISS index exists but metadata is missing",
                    {"vectors": self.assistant.index.ntotal}
                )
                return False
            
            # Verify metadata length matches index
            if len(self.assistant.metadata.get("texts", [])) != self.assistant.index.ntotal:
                self._update_component_status(
                    SystemComponent.INDEX,
                    SystemStatus.WARNING,
                    "FAISS index and metadata size mismatch",
                    {
                        "index_size": self.assistant.index.ntotal,
                        "metadata_size": len(self.assistant.metadata.get("texts", []))
                    }
                )
                return False
            
            # All checks passed
            self._update_component_status(
                SystemComponent.INDEX,
                SystemStatus.OK,
                "FAISS index operational",
                {
                    "vectors": self.assistant.index.ntotal,
                    "documents": len(self.assistant.metadata.get("document_hashes", {}))
                }
            )
            return True
            
        except Exception as e:
            self._update_component_status(
                SystemComponent.INDEX,
                SystemStatus.ERROR,
                f"FAISS index error: {e.__class__.__name__}",
                {"error": str(e)}
            )
            return False
    
    def _update_component_status(
        self, 
        component: SystemComponent, 
        status: SystemStatus, 
        message: str, 
        details: Dict[str, Any]
    ):
        """Update the status of a system component.
        
        Args:
            component: The component to update
            status: The new status value
            message: User-friendly status message
            details: Additional status details
        """
        # Get the previous status
        prev_status = self.component_status[component]["status"]
        
        # Update the component status
        self.component_status[component] = {
            "status": status,
            "last_check": time.time(),
            "message": message,
            "details": details
        }
        
        # Log the status change
        print(f"{component.value}: {status.value} - {message}")
        
        # If status changed from OK to not OK, or from not OK to OK,
        # add a notification to the queue
        if prev_status != status and (prev_status == SystemStatus.OK or status == SystemStatus.OK):
            notification = self._generate_status_change_message(component, prev_status, status)
            if notification:
                self.notification_queue.put({
                    "component": component,
                    "prev_status": prev_status,
                    "status": status,
                    "message": notification
                })
    
    def _update_overall_status(self):
        """Update the overall system status based on component statuses."""
        # Count components by status
        status_count = {status: 0 for status in SystemStatus}
        for component, info in self.component_status.items():
            if component != SystemComponent.OVERALL:
                status_count[info["status"]] += 1
        
        # Determine overall status
        if status_count[SystemStatus.ERROR] > 0:
            overall_status = SystemStatus.ERROR
        elif status_count[SystemStatus.WARNING] > 0:
            overall_status = SystemStatus.WARNING
        elif status_count[SystemStatus.OK] > 0:
            overall_status = SystemStatus.OK
        else:
            overall_status = SystemStatus.UNKNOWN
        
        # Generate a summary message
        message = self._generate_overall_status_message(status_count)
        
        # Update overall status
        self._update_component_status(
            SystemComponent.OVERALL,
            overall_status,
            message,
            {"status_count": {s.value: c for s, c in status_count.items()}}
        )
    
    def _generate_overall_status_message(self, status_count: Dict[SystemStatus, int]) -> str:
        """Generate a summary message describing overall system status.
        
        Args:
            status_count: Count of components by status
            
        Returns:
            A summary message
        """
        if status_count[SystemStatus.ERROR] > 0:
            return f"System experiencing issues with {status_count[SystemStatus.ERROR]} components"
        elif status_count[SystemStatus.WARNING] > 0:
            return f"System has {status_count[SystemStatus.WARNING]} components with warnings"
        elif status_count[SystemStatus.OK] > 0:
            return "All system components operating normally"
        else:
            return "System status unknown"
    
    def _generate_status_change_message(
        self, 
        component: SystemComponent, 
        prev_status: SystemStatus, 
        new_status: SystemStatus
    ) -> str:
        """Generate a user-friendly message describing a component status change.
        
        Args:
            component: The component that changed
            prev_status: The previous status
            new_status: The new status
            
        Returns:
            A status change message suitable for the user
        """
        component_name = component.value
        message = self.component_status[component]["message"]
        
        # If status improved (from ERROR/WARNING to OK)
        if new_status == SystemStatus.OK:
            if prev_status == SystemStatus.ERROR:
                return f"Good news! The {component_name} issue has been resolved. {message}"
            elif prev_status == SystemStatus.WARNING:
                return f"The {component_name} is now functioning normally. {message}"
            else:
                return f"The {component_name} is now operational. {message}"
        
        # If status degraded (from OK to WARNING/ERROR)
        elif prev_status == SystemStatus.OK:
            if new_status == SystemStatus.ERROR:
                return (f"I'm experiencing an issue with {component_name}. {message} "
                        f"{self._get_fallback_message(component)}")
            elif new_status == SystemStatus.WARNING:
                return f"Just to let you know, there's a minor issue with {component_name}. {message}"
        
        # If status went from WARNING to ERROR
        elif prev_status == SystemStatus.WARNING and new_status == SystemStatus.ERROR:
            return (f"The issue with {component_name} has gotten worse. {message} "
                    f"{self._get_fallback_message(component)}")
        
        # If status went from ERROR to WARNING
        elif prev_status == SystemStatus.ERROR and new_status == SystemStatus.WARNING:
            return f"The {component_name} is partially working now. {message}"
            
        return None
    
    def _get_fallback_message(self, component: SystemComponent) -> str:
        """Get a message explaining what fallback options are available.
        
        Args:
            component: The component that failed
            
        Returns:
            A message about fallback options
        """
        if component == SystemComponent.MICROPHONE:
            return "I'll try to continue listening, but you might need to speak more clearly or try an external microphone."
        
        elif component == SystemComponent.API:
            return "I'll continue with basic functions, but some advanced features may be limited."
        
        elif component == SystemComponent.AUDIO_OUTPUT:
            return "I'll switch to text-only mode for responses."
        
        elif component == SystemComponent.RAG or component == SystemComponent.EMBEDDING or component == SystemComponent.INDEX:
            return "I'll try to answer your questions without document search capabilities."
        
        return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of all system components.
        
        Returns:
            Dict containing status information for all components
        """
        return {
            component.value: {
                "status": info["status"].value,
                "message": info["message"],
                "last_check": info["last_check"],
                "details": info["details"]
            }
            for component, info in self.component_status.items()
        }
    
    def handle_component_failure(
        self, 
        component: SystemComponent, 
        error: Exception,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """Handle a component failure, update status, and generate user message.
        
        Args:
            component: The failed component
            error: The exception that occurred
            context: Additional context about the failure
            
        Returns:
            Tuple of (should_retry, user_message)
        """
        # Update component status based on the error
        error_type = self._determine_error_type(error)
        
        # Default values
        should_retry = False
        status = SystemStatus.ERROR
        
        # Determine if error is retryable and appropriate status 
        if error_type in ["timeout", "temporary_network"]:
            should_retry = True
            status = SystemStatus.WARNING
            message = f"Temporary {component.value} issue: {error_type}"
        elif error_type == "permissions":
            status = SystemStatus.ERROR
            message = f"Permission error accessing {component.value}"
            should_retry = False
        elif error_type == "not_found":
            status = SystemStatus.ERROR
            message = f"{component.value} resource not found"
            should_retry = False
        elif error_type == "rate_limit":
            status = SystemStatus.WARNING
            message = f"{component.value} rate limit exceeded"
            should_retry = True
        else:
            status = SystemStatus.ERROR  
            message = f"{component.value} error: {error.__class__.__name__}"
            should_retry = False
        
        # Include context in details
        details = {"error": str(error), "error_type": error_type}
        if context:
            details.update(context)
            
        # Update the component status
        self._update_component_status(component, status, message, details)
        
        # Also update overall status
        self._update_overall_status()
        
        # Get appropriate user message
        user_message = self._generate_user_error_message(component, error_type, should_retry)
        
        return should_retry, user_message
    
    def _determine_error_type(self, error: Exception) -> str:
        """Determine the type of error from the exception.
        
        Args:
            error: The exception to analyze
            
        Returns:
            Error type as a string
        """
        error_str = str(error).lower()
        error_class = error.__class__.__name__.lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        elif "connection" in error_str and ("reset" in error_str or "refused" in error_str):
            return "temporary_network"
        elif "permission" in error_str or "access denied" in error_str:
            return "permissions"
        elif "not found" in error_str or "404" in error_str:
            return "not_found"
        elif "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
            return "rate_limit"
        elif "busy" in error_str or "resource" in error_str and "unavailable" in error_str:
            return "resource_conflict"
        elif "memory" in error_str or "allocation" in error_str:
            return "out_of_memory"
        elif "initialized" in error_str and "not" in error_str:
            return "not_initialized"
        else:
            return "unknown"
    
    def _generate_user_error_message(self, component: SystemComponent, error_type: str, should_retry: bool) -> str:
        """Generate a user-friendly error message.
        
        Args:
            component: The component that failed
            error_type: The type of error 
            should_retry: Whether the operation should be retried
            
        Returns:
            A user-friendly error message
        """
        # Base messages for different error types
        messages = {
            "timeout": "I'm experiencing a slow connection",
            "temporary_network": "I'm having some connectivity issues",
            "permissions": "I don't have permission to access a required resource",
            "not_found": "I couldn't find a required resource",
            "rate_limit": "I've reached a rate limit",
            "resource_conflict": "A required resource is currently busy",
            "out_of_memory": "I've run out of memory",
            "not_initialized": "A required component isn't properly set up",
            "unknown": "I've encountered an unexpected error"
        }
        
        # Get the base message
        message = messages.get(error_type, messages["unknown"])
        
        # Add component-specific details
        if component == SystemComponent.MICROPHONE:
            component_msg = "with your microphone"
            fallback = "You can try speaking more clearly or using an external microphone."
        elif component == SystemComponent.API:
            component_msg = "with my language processing service"
            fallback = "I'll continue with basic functionality, but advanced features may be limited."
        elif component == SystemComponent.AUDIO_OUTPUT:
            component_msg = "with my speech system" 
            fallback = "I'll provide text responses instead of speaking."
        elif component == SystemComponent.RAG or component == SystemComponent.EMBEDDING or component == SystemComponent.INDEX:
            component_msg = "with my knowledge system"
            fallback = "I'll try to answer based on general knowledge without searching documents."
        else:
            component_msg = ""
            fallback = ""
        
        # Construct the full message
        full_message = f"{message} {component_msg}."
        
        # Add retry or fallback information
        if should_retry:
            full_message += f" I'll try again in a moment. {fallback}"
        else:
            full_message += f" {fallback}"
        
        return full_message
    
    def has_notifications(self) -> bool:
        """Check if there are any notifications for the user.
        
        Returns:
            True if there are notifications in the queue
        """
        return not self.notification_queue.empty()
    
    def get_notification(self) -> Optional[Dict[str, Any]]:
        """Get the next notification from the queue.
        
        Returns:
            A notification dict or None if queue is empty
        """
        try:
            return self.notification_queue.get_nowait()
        except queue.Empty:
            return None
    
    def handle_error(
        self, 
        error: Exception, 
        context: str, 
        component: SystemComponent = None, 
        details: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """Handle an error by updating status and generating a user message.
        
        Args:
            error: The exception that occurred
            context: Description of what was happening
            component: The affected component (or None to determine automatically)
            details: Additional error details
            
        Returns:
            Tuple of (should_retry, user_message)
        """
        # Determine which component is affected if not specified
        if component is None:
            component = self._determine_affected_component(error, context)
        
        # Prepare details
        if details is None:
            details = {}
        details["context"] = context
        
        # Handle the component failure
        return self.handle_component_failure(component, error, details)
    
    def _determine_affected_component(self, error: Exception, context: str) -> SystemComponent:
        """Determine which component is affected by an error.
        
        Args:
            error: The exception that occurred
            context: Description of what was happening
            
        Returns:
            The affected SystemComponent
        """
        error_str = str(error).lower()
        context_lower = context.lower()
        
        # Check for microphone/audio input related errors
        if ("microphone" in error_str or "audio" in error_str or 
            "listen" in context_lower or "speech recognition" in context_lower):
            return SystemComponent.MICROPHONE
            
        # Check for API related errors
        elif ("api" in error_str or "http" in error_str or "request" in error_str or 
              "connection" in error_str or "network" in error_str or 
              "query" in context_lower or "generate" in context_lower):
            return SystemComponent.API
            
        # Check for speech output related errors
        elif ("speak" in context_lower or "voice" in error_str or 
              "tts" in error_str or "text to speech" in context_lower):
            return SystemComponent.AUDIO_OUTPUT
            
        # Check for RAG related errors
        elif ("index" in error_str or "embedding" in error_str or 
              "retriev" in context_lower or "rag" in context_lower or 
              "document" in error_str):
            return SystemComponent.RAG
            
        # Default to overall system if we can't determine
        else:
            return SystemComponent.OVERALL
    
    def shutdown(self):
        """Clean up resources used by the system monitor."""
        self.monitor_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        print("System monitor shutdown") 