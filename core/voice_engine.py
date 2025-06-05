"""
core/voice_engine.py
Main voice processing pipeline for the Car Assistant.
Coordinates speech recognition, query processing, and speech synthesis.
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

# Import custom modules
# These would be the actual imports in a real implementation
from speech.recognition import VoiceRecognizer
from speech.synthesis import VoiceSynthesizer
from core.assistant_manager import AssistantManager
from rag.retriever import ManualRetriever
from rag.processor import QueryProcessor
from utils.error_handler import ErrorHandler
from utils.terminology import TerminologyManager

# Configure logging
logger = logging.getLogger(__name__)

class QueryCategory(Enum):
    """Categories for automotive queries."""
    MAINTENANCE = "maintenance"
    TROUBLESHOOTING = "troubleshooting"
    SAFETY = "safety"
    SPECIFICATION = "specification"
    PROCEDURE = "procedure"
    GENERAL = "general"

class VoiceEngine:
    """
    Main processing pipeline for the Voice-Powered Car Assistant.
    Coordinates speech recognition, query processing, RAG retrieval, and speech synthesis.
    """

    def __init__(
        self,
        recognizer: VoiceRecognizer,
        synthesizer: VoiceSynthesizer,
        retriever: ManualRetriever,
        assistant: AssistantManager,
        error_handler: Optional[ErrorHandler] = None,
        terminology_manager: Optional[TerminologyManager] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the VoiceEngine with component dependencies.

        Args:
            recognizer: Instance of VoiceRecognizer for speech input
            synthesizer: Instance of VoiceSynthesizer for speech output
            retriever: Instance of ManualRetriever for document retrieval
            assistant: Instance of AssistantManager for conversation state
            error_handler: Optional instance of ErrorHandler for error management
            terminology_manager: Optional instance of TerminologyManager for term normalization
            log_level: Logging level (default: logging.INFO)
        """
        # Set up logger
        self._configure_logger(log_level)
        
        logger.info("Initializing VoiceEngine")
        
        # Store component references
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self.retriever = retriever
        self.assistant = assistant
        self.error_handler = error_handler or ErrorHandler()
        self.terminology_manager = terminology_manager or TerminologyManager()
        
        # Initialize state
        self.last_query = None
        self.last_response = None
        self.last_context = None
        self.is_processing = False
        self.current_procedure = None
        self.procedure_step = 0
        
        # Wake words and exit commands
        self.wake_words = ["hey car", "hey assistant", "car assistant", "hello assistant"]
        self.exit_commands = ["exit", "quit", "stop", "goodbye", "bye"]
        
        # Initialize query processor
        self.query_processor = QueryProcessor(terminology_manager=self.terminology_manager)
        
        logger.info("VoiceEngine initialized successfully")

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
            
        logger.debug("Logger configured for VoiceEngine")

    def _create_mock_processor(self) -> Any:
        """
        Create a mock query processor (temporary until real implementation).
        
        Returns:
            Object with a process_query method
        """
        class MockProcessor:
            def process_query(self, query: str, context: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
                """Simple mock processor that returns pre-defined responses based on keywords."""
                query = query.lower()
                
                # Sample responses based on query content
                if "oil" in query or "change" in query:
                    response = "Your car's oil should be changed every 5,000 to 7,500 miles. Would you like me to check when your last oil change was?"
                    context = {"topic": "maintenance", "subtopic": "oil_change"}
                
                elif "tire" in query or "pressure" in query:
                    response = "The recommended tire pressure for your vehicle is 32 PSI for front tires and 30 PSI for rear tires. Would you like instructions on how to check your tire pressure?"
                    context = {"topic": "maintenance", "subtopic": "tire_pressure"}
                
                elif "music" in query or "play" in query:
                    response = "I'd be happy to play music for you. What would you like to listen to?"
                    context = {"topic": "entertainment", "subtopic": "music"}
                
                elif "temperature" in query or "climate" in query:
                    response = "The current temperature in the car is 72 degrees. Would you like me to adjust it?"
                    context = {"topic": "climate_control", "action": "status"}
                
                elif "help" in query:
                    response = "I can help you with vehicle maintenance information, climate control, navigation, and entertainment. What would you like assistance with?"
                    context = {"topic": "help", "action": "overview"}
                
                else:
                    response = "I'm sorry, I'm not sure how to help with that yet. You can ask me about vehicle maintenance, climate control, or entertainment features."
                    context = {"topic": "unknown", "query": query}
                
                return response, context
                
        return MockProcessor()

    def process_query(self, query: str) -> str:
        """
        Process a voice query through the pipeline.
        
        Args:
            query: The text query to process
            
        Returns:
            str: Response text to be spoken
        """
        if not query:
            logger.warning("Empty query received")
            return "I'm sorry, I didn't catch that. Could you please repeat?"
        
        logger.info(f"Processing query: '{query}'")
        self.is_processing = True
        self.last_query = query
        
        try:
            # Check for exit commands
            if any(exit_cmd in query.lower() for exit_cmd in self.exit_commands):
                logger.info("Exit command detected")
                self.is_processing = False
                return "Goodbye! Have a safe trip."
            
            # Normalize technical terms
            try:
                normalized_query = self.terminology_manager.normalize_query(query)
            except Exception as e:
                logger.warning(f"Failed to normalize query: {str(e)}")
                normalized_query = query  # Use original query if normalization fails
            
            # Get conversation context from assistant
            context = self.assistant.get_context()
            logger.debug(f"Current context: {context}")
            
            # Process query to determine category and entities
            try:
                processed_query, query_type, entities = self.query_processor.preprocess_query(normalized_query)
            except Exception as e:
                logger.warning(f"Failed to preprocess query: {str(e)}")
                processed_query = normalized_query
                query_type = "general"
                entities = {}
            
            # Handle multi-step procedures
            if self.current_procedure and self.procedure_step > 0:
                response = self._handle_procedure_step(query, context)
                return response
            
            # Route query based on category
            if query_type == QueryCategory.MAINTENANCE.value:
                response = self._handle_maintenance_query(processed_query, entities, context)
            elif query_type == QueryCategory.TROUBLESHOOTING.value:
                response = self._handle_troubleshooting_query(processed_query, entities, context)
            elif query_type == QueryCategory.SAFETY.value:
                response = self._handle_safety_query(processed_query, entities, context)
            elif query_type == QueryCategory.PROCEDURE.value:
                response = self._handle_procedure_query(processed_query, entities, context)
            else:
                response = self._handle_general_query(processed_query, context)
            
            # Update assistant context
            self.assistant.update_context({
                'last_query': query,
                'query_type': query_type,
                'entities': entities,
                'response': response
            })
            
            # Store the response and context
            self.last_response = response
            self.last_context = context
            
            logger.info(f"Response generated: '{response[:50]}{'...' if len(response) > 50 else ''}'")
            
            return response
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "query_processing")
            logger.error(f"Error processing query: {error_msg}", exc_info=True)
            return f"I'm sorry, I encountered an error while processing your request: {error_msg}"
        
        finally:
            self.is_processing = False

    def _handle_maintenance_query(self, query: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Handle maintenance-related queries."""
        try:
            # Get relevant maintenance information
            docs = self.retriever.retrieve_documents(query, category="maintenance")
            
            if not docs:
                return "I couldn't find specific maintenance information for that. Could you please rephrase your question?"
            
            # Process the documents to generate response
            response = self._generate_maintenance_response(docs, entities)
            return response
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "maintenance_query")
            return f"I'm sorry, I had trouble finding maintenance information: {error_msg}"

    def _handle_troubleshooting_query(self, query: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Handle troubleshooting queries."""
        try:
            # Get relevant troubleshooting information
            docs = self.retriever.retrieve_documents(query, category="troubleshooting")
            
            if not docs:
                return "I couldn't find specific troubleshooting information for that issue. Could you please provide more details?"
            
            # Process the documents to generate response
            response = self._generate_troubleshooting_response(docs, entities)
            return response
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "troubleshooting_query")
            return f"I'm sorry, I had trouble finding troubleshooting information: {error_msg}"

    def _handle_safety_query(self, query: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Handle safety-related queries."""
        try:
            # Get relevant safety information
            docs = self.retriever.retrieve_documents(query, category="safety")
            
            if not docs:
                return "I couldn't find specific safety information for that. Please consult your vehicle's manual or a professional for safety concerns."
            
            # Process the documents to generate response
            response = self._generate_safety_response(docs, entities)
            return response
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "safety_query")
            return f"I'm sorry, I had trouble finding safety information: {error_msg}"

    def _handle_procedure_query(self, query: str, entities: Dict[str, List[str]], context: Dict[str, Any]) -> str:
        """Handle procedure-related queries."""
        try:
            # Get relevant procedure information
            docs = self.retriever.retrieve_documents(query, category="procedure")
            
            if not docs:
                return "I couldn't find specific procedure information for that. Could you please rephrase your question?"
            
            # Initialize procedure tracking
            self.current_procedure = self._extract_procedure(docs)
            self.procedure_step = 1
            
            # Return first step
            return self._get_procedure_step(1)
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "procedure_query")
            return f"I'm sorry, I had trouble finding procedure information: {error_msg}"

    def _handle_procedure_step(self, query: str, context: Dict[str, Any]) -> str:
        """Handle multi-step procedure interactions."""
        try:
            # Check for procedure navigation commands
            if "next" in query.lower():
                self.procedure_step += 1
            elif "previous" in query.lower():
                self.procedure_step = max(1, self.procedure_step - 1)
            elif "repeat" in query.lower():
                pass  # Keep current step
            elif "stop" in query.lower():
                self.current_procedure = None
                self.procedure_step = 0
                return "Procedure stopped. How else can I help you?"
            
            # Get current step
            step_response = self._get_procedure_step(self.procedure_step)
            
            # Check if we've reached the end
            if not step_response:
                self.current_procedure = None
                self.procedure_step = 0
                return "You've completed all steps of the procedure. Is there anything else you need help with?"
            
            return step_response
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "procedure_step")
            return f"I'm sorry, I had trouble with the procedure step: {error_msg}"

    def _handle_general_query(self, query: str, context: Dict[str, Any]) -> str:
        """Handle general queries."""
        try:
            # Get relevant general information
            docs = self.retriever.retrieve_documents(query)
            
            if not docs:
                return "I'm not sure about that. Could you please rephrase your question or ask about something else?"
            
            # Process the documents to generate response
            response = self._generate_general_response(docs)
            return response
            
        except Exception as e:
            error_msg = self.error_handler.handle_error(e, "general_query")
            return f"I'm sorry, I had trouble finding information: {error_msg}"

    def _extract_procedure(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract procedure steps from documents."""
        procedure = {
            'title': '',
            'steps': [],
            'warnings': [],
            'tools': [],
            'parts': []
        }
        
        # Extract procedure information from documents
        for doc in docs:
            if 'title' in doc and not procedure['title']:
                procedure['title'] = doc['title']
            if 'steps' in doc:
                procedure['steps'].extend(doc['steps'])
            if 'warnings' in doc:
                procedure['warnings'].extend(doc['warnings'])
            if 'tools' in doc:
                procedure['tools'].extend(doc['tools'])
            if 'parts' in doc:
                procedure['parts'].extend(doc['parts'])
        
        return procedure

    def _get_procedure_step(self, step_number: int) -> str:
        """Get a specific step from the current procedure."""
        if not self.current_procedure or not self.current_procedure['steps']:
            return None
        
        if step_number < 1 or step_number > len(self.current_procedure['steps']):
            return None
        
        step = self.current_procedure['steps'][step_number - 1]
        
        # Format step response
        response = f"Step {step_number}: {step['description']}"
        
        # Add warnings if any
        if step.get('warnings'):
            response += "\nWarnings: " + ", ".join(step['warnings'])
        
        # Add tools if any
        if step.get('tools'):
            response += "\nRequired tools: " + ", ".join(step['tools'])
        
        return response

    def _generate_maintenance_response(self, docs: List[Dict[str, Any]], entities: Dict[str, List[str]]) -> str:
        """Generate response for maintenance queries."""
        # Implementation depends on your response generation strategy
        pass

    def _generate_troubleshooting_response(self, docs: List[Dict[str, Any]], entities: Dict[str, List[str]]) -> str:
        """Generate response for troubleshooting queries."""
        # Implementation depends on your response generation strategy
        pass

    def _generate_safety_response(self, docs: List[Dict[str, Any]], entities: Dict[str, List[str]]) -> str:
        """Generate response for safety queries."""
        # Implementation depends on your response generation strategy
        pass

    def _generate_general_response(self, docs: List[Dict[str, Any]]) -> str:
        """Generate response for general queries."""
        # Implementation depends on your response generation strategy
        pass

    def continuous_listen(self, wake_word_required: bool = True) -> None:
        """
        Start a continuous listening loop.
        
        Args:
            wake_word_required: Whether a wake word is required to process queries
        """
        logger.info(f"Starting continuous listening loop (wake word {'required' if wake_word_required else 'not required'})")
        self.synthesizer.speak("Voice assistant is ready. How can I help you with your vehicle today?")
        
        try:
            while True:
                # Listen for query
                query = self.recognizer.listen()
                
                if not query:
                    continue
                
                # Check for wake word if required
                if wake_word_required:
                    if not self._contains_wake_word(query):
                        logger.debug(f"Wake word not detected in: '{query}'")
                        continue
                    else:
                        # Remove wake word from query
                        query = self._remove_wake_word(query)
                        logger.debug(f"Wake word removed, processing: '{query}'")
                
                # Process query and speak response
                response = self.process_query(query)
                self.synthesizer.speak(response)
                
                # Check if it was an exit command
                if any(exit_cmd in query.lower() for exit_cmd in self.exit_commands):
                    logger.info("Exit command processed, ending continuous listening")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, stopping continuous listening")
            self.synthesizer.speak("Voice assistant shutting down.")
            
        except Exception as e:
            logger.error(f"Error in continuous listening loop: {e}", exc_info=True)
            self.synthesizer.speak("I've encountered a problem and need to shut down. Please restart me.")

    def _contains_wake_word(self, query: str) -> bool:
        """
        Check if the query contains a wake word.
        
        Args:
            query: The query to check
            
        Returns:
            bool: True if a wake word is found
        """
        return any(wake_word in query.lower() for wake_word in self.wake_words)

    def _remove_wake_word(self, query: str) -> str:
        """
        Remove wake word from query.
        
        Args:
            query: The query containing a wake word
            
        Returns:
            str: Query with wake word removed
        """
        query_lower = query.lower()
        for wake_word in self.wake_words:
            if wake_word in query_lower:
                return query.lower().replace(wake_word, "", 1).strip()
        return query

    def get_last_interaction(self) -> Dict[str, Any]:
        """
        Get information about the last interaction.
        
        Returns:
            Dict: Last query, response, and context
        """
        return {
            "query": self.last_query,
            "response": self.last_response,
            "context": self.last_context
        }

    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update engine settings.
        
        Args:
            settings: Dictionary of settings to update
        """
        logger.info("Updating VoiceEngine settings")
        
        # Update wake words if provided
        if "wake_words" in settings:
            self.wake_words = settings["wake_words"]
            logger.debug(f"Updated wake words: {self.wake_words}")
            
        # Update exit commands if provided
        if "exit_commands" in settings:
            self.exit_commands = settings["exit_commands"]
            logger.debug(f"Updated exit commands: {self.exit_commands}")
            
        # Update log level if provided
        if "log_level" in settings:
            self._configure_logger(settings["log_level"])
            logger.debug(f"Updated log level: {settings['log_level']}")
        
        # Pass relevant settings to components
        component_settings = {}
        
        # Settings for recognizer
        recognizer_keys = ["energy_threshold", "pause_threshold", "timeout", 
                          "phrase_time_limit", "dynamic_energy_threshold"]
        recognizer_settings = {k: v for k, v in settings.items() if k in recognizer_keys}
        if recognizer_settings:
            logger.debug(f"Updating recognizer with settings: {recognizer_settings}")
            self.recognizer.update_settings(recognizer_settings)
        
        # Settings for synthesizer
        synthesizer_keys = ["rate", "volume", "voice_id"]
        synthesizer_settings = {k: v for k, v in settings.items() if k in synthesizer_keys}
        if synthesizer_settings:
            logger.debug(f"Updating synthesizer with settings: {synthesizer_settings}")
            self.synthesizer.update_settings(synthesizer_settings)


if __name__ == "__main__":
    # Set up basic logging configuration for the test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Import required classes (would normally be at top of file)
    from speech.recognition import VoiceRecognizer
    from speech.synthesis import VoiceSynthesizer
    from rag.retriever import ManualRetriever
    from core.assistant_manager import AssistantManager
    
    # Create mock instances for testing
    # In a real implementation, these would be properly instantiated
    recognizer = VoiceRecognizer()
    synthesizer = VoiceSynthesizer()
    retriever = ManualRetriever()
    assistant = AssistantManager()
    
    # Create the engine
    engine = VoiceEngine(recognizer, synthesizer, retriever, assistant)
    
    # Test with a sample query
    test_query = "What's the recommended tire pressure for my car?"
    print(f"\nTesting with query: '{test_query}'")
    
    response = engine.process_query(test_query)
    print(f"Response: {response}")
    
    # Speak the response
    synthesizer.speak(response)