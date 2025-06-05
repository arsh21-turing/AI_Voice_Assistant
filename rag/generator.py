"""
Response generation module with Groq API integration for the RAG pipeline.

This module provides the ResponseGenerator class, which generates natural language
responses based on retrieved context chunks, using the Groq API for LLM capabilities
and specialized templates for different query types.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import random
from pathlib import Path

# Import from project modules
try:
    from config import get_config
    from rag.templates import PromptTemplateManager, TemplateError
    from utils.groq_client import GroqClient, GroqAPIError
except ImportError:
    # For standalone usage or testing
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config
    from rag.templates import PromptTemplateManager, TemplateError
    from utils.groq_client import GroqClient, GroqAPIError

class ResponseGenerationError(Exception):
    """Exception raised for response generation errors."""
    pass

class ResponseGenerator:
    """
    Enhanced response generator with Groq API integration and context-aware responses.
    
    This class generates natural language responses based on retrieved context chunks,
    using the Groq API for LLM capabilities and optimizing responses for voice output.
    """
    
    def __init__(self, 
                config_manager=None, 
                template_manager=None,
                groq_client=None):
        """
        Initialize the response generator.
        
        Args:
            config_manager: Optional configuration manager instance
            template_manager: Optional template manager instance
            groq_client: Optional Groq API client instance
        """
        # Get configuration
        self.config = config_manager if config_manager else get_config()
        
        # Initialize template manager if not provided
        self.template_manager = template_manager
        if not self.template_manager:
            self.template_manager = PromptTemplateManager(self.config)
        
        # Initialize Groq client if not provided
        self.groq_client = groq_client
        if not self.groq_client:
            self.groq_client = GroqClient(self.config)
        
        # Get configuration settings
        self.max_response_length = self.config.get(
            'APP_SETTINGS', 
            'RESPONSE_MAX_LENGTH', 
            250
        )
        self.fallback_messages = {
            'no_context': self.config.get(
                'ERROR_HANDLING',
                'NO_CONTEXT_MESSAGE',
                "I don't have enough information to answer that question about your vehicle. "
                "Could you try asking in a different way or about a different topic?"
            ),
            'insufficient_context': self.config.get(
                'ERROR_HANDLING',
                'INSUFFICIENT_CONTEXT_MESSAGE',
                "I found some information about that, but it doesn't fully answer your question. "
                "Would you like me to tell you what I know, or would you prefer to ask something else?"
            ),
            'general_error': self.config.get(
                'ERROR_HANDLING',
                'DEFAULT_ERROR_MESSAGE',
                "Sorry, I encountered an error while processing your request. "
                "Could you please try again?"
            )
        }
        
        logging.info("Initialized enhanced response generator with Groq API integration")
    
    def generate(self, 
                context_chunks: List[Dict], 
                query: str,
                query_type: str = "general") -> str:
        """
        Generate a response from context chunks using Groq API.
        
        Args:
            context_chunks: Retrieved context chunks with metadata
            query: The user's original query
            query_type: Type of query for specialized response formatting
            
        Returns:
            A natural language response answering the query
            
        Raises:
            ResponseGenerationError: If response generation fails
        """
        start_time = time.time()
        logging.debug(f"Generating response for query type: {query_type}")
        
        # Check if we have any context chunks
        if not context_chunks or len(context_chunks) == 0:
            logging.warning(f"No context chunks provided for query: {query}")
            return self.fallback_response(query, reason="no_context")
        
        try:
            # Format context for prompt
            formatted_context = self.context_to_prompt(context_chunks)
            
            # Determine response type from query type
            response_type = self._map_query_to_response_type(query_type)
            
            # Generate response with Groq API
            response = self.generate_with_groq(
                query=query,
                context=formatted_context,
                response_type=response_type
            )
            
            # Ensure manual citations are included
            response = self.ensure_manual_citations(response, context_chunks)
            
            # Format for speech output
            response = self.format_for_speech(response)
            
            # Apply final post-processing
            response = self.post_process_response(response)
            
            generation_time = time.time() - start_time
            logging.info(f"Generated response in {generation_time:.2f} seconds")
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return self.fallback_response(query, error=e)
    
    def context_to_prompt(self, context_chunks: List[Dict]) -> str:
        """
        Convert context chunks into a structured prompt with section references.
        
        Args:
            context_chunks: Retrieved context chunks with metadata
            
        Returns:
            Formatted context for inclusion in prompts
        """
        # Extract sections and references
        section_map = self.extract_manual_sections(context_chunks)
        
        # Create a structured context string
        context_parts = []
        
        # Add each chunk with its section reference
        for i, chunk in enumerate(context_chunks):
            # Get chunk information
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            
            # Skip empty chunks
            if not chunk_text.strip():
                continue
                
            # Get section information
            section = metadata.get('section', '')
            page = metadata.get('page', '')
            source = metadata.get('source', '')
            
            # Create section header
            section_header = ""
            if section:
                section_header = f"Section {section}"
                if page:
                    section_header += f" (Page {page})"
                if source:
                    section_header += f" from {source}"
                    
            # Add to context parts
            if section_header:
                context_parts.append(f"{section_header}:\n{chunk_text}")
            else:
                context_parts.append(chunk_text)
        
        # Combine all context parts
        formatted_context = "\n\n".join(context_parts)
        
        # Try to use the template to format context
        try:
            # Get the template for context formatting
            context_prompt = self.template_manager.get_template('CONTEXT_FORMAT_PROMPT')
            
            # Use Groq API to format the context
            chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk.get('text', '')}" 
                                     for i, chunk in enumerate(context_chunks)])
            
            format_prompt = context_prompt.replace("{chunks}", chunks_text)
            
            # Generate formatted context
            messages = [
                {"role": "system", "content": "You are a helpful assistant that formats technical information clearly."},
                {"role": "user", "content": format_prompt}
            ]
            
            response = self.groq_client.chat_completion(messages)
            formatted_text = self.groq_client.extract_response_text(response)
            
            if formatted_text and len(formatted_text) > 50:  # Ensure we got a meaningful response
                formatted_context = formatted_text
                
        except Exception as e:
            logging.warning(f"Error formatting context with template: {str(e)}. Using default formatting.")
            # We'll use our simple formatting from above if the template approach fails
        
        return formatted_context
    
    def extract_manual_sections(self, chunks: List[Dict]) -> Dict[str, str]:
        """
        Extract and format manual section references from chunks.
        
        Args:
            chunks: Context chunks with metadata
            
        Returns:
            Mapping of section references to content
        """
        section_map = {}
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            section = metadata.get('section', '')
            
            if section and 'text' in chunk:
                # Store section text (potentially overwriting with duplicate sections)
                section_map[section] = chunk['text']
        
        return section_map
    
    def generate_with_groq(self, 
                          query: str,
                          context: str,
                          response_type: str = "general") -> str:
        """
        Generate response using Groq API with appropriate templates.
        
        Args:
            query: The user's query
            context: Formatted context information
            response_type: Type of response to generate
            
        Returns:
            Generated response from Groq API
        """
        # Get appropriate response prompt template
        try:
            prompt_template = self.template_manager.get_response_prompt(response_type)
        except TemplateError:
            logging.warning(f"Template for response type '{response_type}' not found. Using general template.")
            prompt_template = self.template_manager.get_response_prompt()
        
        # Format the prompt with query and context
        formatted_prompt = prompt_template.replace("{query}", query).replace("{context}", context)
        
        # Get system prompt
        system_prompt = self.template_manager.get_system_prompt()
        
        # Add voice optimization guidelines
        voice_guidelines = self.template_manager.get_voice_optimization_guidelines()
        system_prompt = f"{system_prompt}\n\n{voice_guidelines}"
        
        # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Make API request
        try:
            response = self.groq_client.chat_completion(messages)
            response_text = self.groq_client.extract_response_text(response)
            
            # Check if we got a valid response
            if not response_text or len(response_text) < 20:
                raise ResponseGenerationError("Received empty or too short response from API")
                
            return response_text
            
        except GroqAPIError as e:
            logging.error(f"Groq API error: {str(e)}")
            raise ResponseGenerationError(f"API error: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error in response generation: {str(e)}")
            raise ResponseGenerationError(f"Generation error: {str(e)}")
    
    def fallback_response(self, 
                         query: str, 
                         error: Optional[Exception] = None,
                         reason: Optional[str] = None) -> str:
        """
        Generate fallback response when normal generation fails.
        
        Args:
            query: The original user query
            error: Error that triggered fallback
            reason: Reason for fallback
            
        Returns:
            Graceful fallback response
        """
        # Determine fallback type
        fallback_type = reason if reason else "general_error"
        
        # Log the fallback event
        if error:
            logging.warning(f"Using fallback response due to error: {str(error)}")
        else:
            logging.info(f"Using fallback response. Reason: {fallback_type}")
        
        # Get fallback message from config
        fallback_message = self.fallback_messages.get(
            fallback_type, 
            self.fallback_messages['general_error']
        )
        
        # Try to generate a more specific fallback for certain queries
        if fallback_type == "no_context" or fallback_type == "insufficient_context":
            try:
                # See if we can provide general guidance without specific manual info
                general_guidance = self._generate_general_guidance(query)
                if general_guidance:
                    fallback_message = (
                        f"{fallback_message}\n\n"
                        f"I can provide some general guidance, though: {general_guidance}"
                    )
            except Exception as e:
                logging.warning(f"Failed to generate general guidance: {str(e)}")
        
        return fallback_message
    
    def _generate_general_guidance(self, query: str) -> Optional[str]:
        """
        Generate general guidance when specific manual information is unavailable.
        
        Args:
            query: The user's query
            
        Returns:
            General guidance or None if unable to generate
        """
        # Prompt to generate general guidance
        prompt = (
            f"The user asked: '{query}'\n\n"
            f"I don't have access to their specific vehicle manual. "
            f"Provide a brief, general response (2-3 sentences max) that acknowledges "
            f"the lack of specific information but offers general automotive guidance "
            f"if possible. If the query is too specific to a particular vehicle, "
            f"simply suggest consulting the owner's manual."
        )
        
        try:
            messages = [
                {"role": "system", "content": "You provide concise, helpful general automotive guidance."},
                {"role": "user", "content": prompt}
            ]
            
            # Generate general guidance with lower max tokens to keep it brief
            response = self.groq_client.chat_completion(
                messages=messages,
                max_tokens=150
            )
            guidance = self.groq_client.extract_response_text(response)
            
            if guidance and len(guidance) > 20:
                return guidance
                
        except Exception as e:
            logging.warning(f"Failed to generate general guidance: {str(e)}")
            
        return None
    
    def format_for_speech(self, response: str) -> str:
        """
        Optimize response for speech synthesis.
        
        Args:
            response: Generated response text
            
        Returns:
            Response formatted for optimal speech synthesis
        """
        # Replace special characters that might affect speech synthesis
        speech_formatted = response
        
        # Remove markdown formatting
        speech_formatted = re.sub(r'\*\*(.*?)\*\*', r'\1', speech_formatted)  # Bold
        speech_formatted = re.sub(r'\*(.*?)\*', r'\1', speech_formatted)      # Italic
        
        # Replace bullet points with clear spoken alternatives
        bullet_pattern = re.compile(r'^[\s]*[-•*][\s]*(.*?)$', re.MULTILINE)
        speech_formatted = bullet_pattern.sub(r'- \1', speech_formatted)
        
        # Format numbers for better speech synthesis
        # Make sure decimal numbers are read correctly
        speech_formatted = re.sub(r'(\d)\.(\d)', r'\1 point \2', speech_formatted)
        
        # Format references to sections
        speech_formatted = re.sub(r'Section (\d+)\.(\d+)', r'Section \1 point \2', speech_formatted)
        
        # Spell out common acronyms
        acronyms = {
            'ABS': 'A B S (Anti-lock Braking System)',
            'TPMS': 'T P M S (Tire Pressure Monitoring System)',
            'ECU': 'E C U (Engine Control Unit)',
            'OBD': 'O B D (On-Board Diagnostics)',
            'RPM': 'RPM (revolutions per minute)',
            'MPG': 'MPG (miles per gallon)',
            'A/C': 'air conditioning',
            'MPH': 'miles per hour',
        }
        
        for acronym, expansion in acronyms.items():
            # Replace only standalone acronyms (with word boundaries)
            speech_formatted = re.sub(r'\b' + acronym + r'\b', expansion, speech_formatted)
        
        return speech_formatted
    
    def ensure_manual_citations(self, response: str, context_chunks: List[Dict]) -> str:
        """
        Ensure manual sections are properly cited in the response.
        
        Args:
            response: Generated response text
            context_chunks: Source context chunks
            
        Returns:
            Response with proper citations
        """
        # Extract all section references from chunks
        sections = set()
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            section = metadata.get('section', '')
            if section:
                sections.add(section)
        
        # If we have sections but none are mentioned in the response, add a citation
        if sections and not any(f"Section {s}" in response for s in sections):
            # Some responses will naturally incorporate section references,
            # but if they don't, we'll add a reference to the most relevant sections
            section_list = sorted(list(sections))
            if len(section_list) == 1:
                citation = f" This information comes from Section {section_list[0]} of your manual."
            elif len(section_list) > 1:
                if len(section_list) > 3:
                    # If there are many sections, just mention a few
                    section_list = section_list[:3]
                    citation = f" This information comes from Sections {', '.join(section_list)} and other parts of your manual."
                else:
                    citation = f" This information comes from Sections {', '.join(section_list)} of your manual."
            
            # Add the citation to the end of the response
            if not response.endswith("."):
                response += "."
            response += citation
            
        return response
    
    def post_process_response(self, response: str) -> str:
        """
        Apply final post-processing to the response.
        
        Args:
            response: Generated response text
            
        Returns:
            Final processed response
        """
        # Trim response if too long
        if self.max_response_length > 0 and len(response) > self.max_response_length:
            # Try to trim at a sentence boundary
            sentences = re.split(r'(?<=[.!?])\s+', response)
            processed = ""
            for sentence in sentences:
                if len(processed) + len(sentence) + 1 <= self.max_response_length:
                    if processed:
                        processed += " "
                    processed += sentence
                else:
                    break
            
            # If we couldn't find a good sentence boundary, just trim with ellipsis
            if not processed or len(processed) < 0.5 * self.max_response_length:
                processed = response[:self.max_response_length-3] + "..."
            
            response = processed
        
        # Ensure response doesn't have trailing whitespace
        response = response.strip()
        
        return response
    
    def _map_query_to_response_type(self, query_type: str) -> str:
        """
        Map query type to appropriate response type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Corresponding response type
        """
        # Map of query types to response types
        mapping = {
            "maintenance": "maintenance",
            "troubleshooting": "diagnostic",
            "diagnostic": "diagnostic", 
            "feature": "feature",
            "procedural": "maintenance",
            "safety": "safety",
            "specification": "specification"
        }
        
        # Return mapped type or default to "general"
        return mapping.get(query_type.lower(), "general")


# For testing or direct usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the generator
    generator = ResponseGenerator()
    
    # Sample context and query
    context_chunks = [
        {
            'text': 'The brake system uses hydraulic pressure to activate the brake pads.',
            'metadata': {
                'section': 'Brake System',
                'page': '45',
                'source': 'Owner\'s Manual'
            }
        }
    ]
    
    query = "How does the brake system work?"
    
    try:
        # Generate response
        response = generator.generate(context_chunks, query)
        print("\nGenerated Response:")
        print(response)
        
    except Exception as e:
        print(f"Error: {str(e)}")