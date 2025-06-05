"""
Prompt template management for the automotive assistant RAG pipeline.

This module provides a centralized manager for all prompt templates used in the
RAG pipeline, including system prompts, query processing prompts, context
formatting prompts, and response generation prompts specialized for automotive
knowledge and voice interactions.
"""

import logging
import string
from typing import Dict, Any, Optional, List, Union

# Import configuration
try:
    from config import get_config
except ImportError:
    # For standalone usage or testing
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config

class TemplateError(Exception):
    """Exception raised for template-related errors."""
    pass

class PromptTemplateManager:
    """
    Manager for automotive-specific prompt templates used in the RAG pipeline.
    
    This class provides centralized access to all prompt templates used throughout
    the application, specializing in automotive knowledge representation and
    voice-optimized responses.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the prompt template manager.
        
        Args:
            config_manager: Optional configuration manager instance.
                            If None, will use the global configuration.
        """
        # Get configuration
        self.config = config_manager if config_manager else get_config()
        
        # Initialize templates from configuration
        self.templates = self._initialize_templates()
        
        # Register any additional templates not in config
        self._register_default_templates()
        
        logging.info("Initialized prompt template manager")
    
    def _initialize_templates(self) -> Dict[str, str]:
        """
        Initialize templates dictionary from configuration.
        
        Returns:
            Dictionary of template names to template texts
        """
        templates = {}
        
        # Get templates from configuration
        config_templates = self.config.get_config_section('PROMPT_TEMPLATES')
        if config_templates:
            templates.update(config_templates)
            
        return templates
    
    def _register_default_templates(self):
        """Register default templates that may not be in the configuration."""
        # System prompt
        if 'SYSTEM_PROMPT' not in self.templates:
            self.register_template(
                'SYSTEM_PROMPT',
                """You are an automotive assistant with expertise in vehicle maintenance, 
                diagnostics, and general car knowledge. Your role is to provide accurate,
                helpful information to drivers about their vehicles.

                Guidelines:
                - Provide concise, accurate information suitable for voice responses
                - Use clear, simple language avoiding technical jargon unless necessary
                - Format responses to be easily understood when spoken aloud
                - When giving instructions, use step-by-step format with clear sequencing
                - If you don't have enough information, clearly state what you know and what you don't
                - Focus on safety and best practices for vehicle maintenance
                - Cite the specific section of the manual when relevant

                Remember that your responses will be converted to speech and
                should be optimized for listening rather than reading."""
            )
        
        # Response generation prompt
        if 'RESPONSE_PROMPT' not in self.templates:
            self.register_template(
                'RESPONSE_PROMPT',
                """Based on the context information and the user's question, provide a helpful
                response that directly answers the question. If the information in the context 
                is insufficient to fully answer the question, clearly state what you can and 
                cannot answer.

                Context information:
                {context}

                User question:
                {query}

                Guidelines for your response:
                - Be concise and direct, focusing on key information
                - Use natural, conversational language optimized for speech
                - Structure complex information in easy-to-follow steps
                - Avoid technical jargon unless it appears in the context
                - Do not introduce information not found in the context
                - If giving instructions, present them in a clear, sequential format
                - Mention the specific section of the manual if referenced in the context

                Response:"""
            )
        
        # Context formatting prompt
        if 'CONTEXT_FORMAT_PROMPT' not in self.templates:
            self.register_template(
                'CONTEXT_FORMAT_PROMPT',
                """Format these context chunks into a coherent, well-structured document that 
                preserves all key information from the original sources. Focus on maintaining 
                all technical details accurately.

                Context chunks:
                {chunks}

                Guidelines:
                - Maintain all technical specifications exactly as given
                - Preserve any numerical values and measurements precisely
                - Keep all part numbers, model references, and technical terms intact
                - Organize information in a logical flow
                - Remove redundant information that appears in multiple chunks
                - Preserve source attribution (e.g., "According to the owner's manual...")
                - For procedural information, ensure steps remain in the correct order
                - Distinguish between different vehicle models or years if mentioned
                
                Formatted context:"""
            )

        # Query understanding prompt
        if 'QUERY_UNDERSTANDING_PROMPT' not in self.templates:
            self.register_template(
                'QUERY_UNDERSTANDING_PROMPT',
                """Analyze the following user query about their vehicle to identify the key 
                information being requested, the type of query, and any specific entities 
                mentioned.

                User query:
                {query}

                Task:
                1. Determine the primary type of query (choose one):
                   - Maintenance question (how to maintain a component)
                   - Troubleshooting question (diagnosing a problem)
                   - Feature inquiry (how a feature works)
                   - Specification question (details about the vehicle)
                   - Procedural question (step-by-step instructions)
                   - Safety question (safety-related concerns)
                   - General information (other general knowledge)

                2. Extract key entities mentioned:
                   - Vehicle components (e.g., "engine", "brakes", "transmission")
                   - Symptoms (e.g., "noise", "warning light", "leak")
                   - Actions (e.g., "replacing", "adjusting", "checking")
                   - Specifications (e.g., "oil type", "tire pressure", "capacity")
                
                3. Identify any implied vehicle model or context information
                
                4. Restate the query in a standardized format suitable for information retrieval

                Your analysis:"""
            )

        # Voice optimization guidelines
        if 'VOICE_OPTIMIZATION_GUIDELINES' not in self.templates:
            self.register_template(
                'VOICE_OPTIMIZATION_GUIDELINES',
                """When formatting responses for voice output:
                
                Content structure:
                - Start with the most important information first
                - Use shorter sentences and simple grammatical structures
                - Avoid long lists of items that would be hard to remember when heard
                - Use clear transitions between different parts of the answer
                - For numerical information, round to simplest form when precision isn't critical
                
                Speech-friendly formatting:
                - Spell out acronyms the first time they are used
                - Avoid special characters that might not translate well to speech
                - Write numbers in a way that sounds natural when read aloud
                - Use words like "first," "second," "next" for sequences rather than numbers or bullets
                - Include slight pauses (with commas) where a listener would need to process information
                - For part numbers or codes, include both the code and a description

                Format URLs, part numbers, or complex identifiers in a way that would be clear
                if read aloud."""
            )

        # Different response types
        response_types = {
            'DIAGNOSTIC_RESPONSE_PROMPT': 
            """Based on the provided context and the user's question about a vehicle issue, 
            provide a diagnostic response that helps identify the potential problem and suggests
            appropriate next steps. Focus on safety, clarity, and practicality.
            
            Context information:
            {context}
            
            User question:
            {query}
            
            Your diagnostic response should:
            - Identify the most likely causes based on the symptoms described
            - Suggest simple checks the user can perform safely themselves
            - Note when professional service is recommended
            - Prioritize safety-critical issues
            - When relevant, explain warning signs that indicate a serious problem
            - Avoid technical jargon unless it appears in the context
            
            Response:""",
            
            'MAINTENANCE_RESPONSE_PROMPT':
            """Based on the provided context and the user's maintenance question, provide a
            clear, step-by-step response about vehicle maintenance procedures. Ensure all
            information is accurate according to the vehicle documentation.
            
            Context information:
            {context}
            
            User question:
            {query}
            
            Your maintenance response should:
            - Provide clear sequential steps for the maintenance procedure
            - Note any required tools, parts, or fluids with specific recommendations
            - Include important safety warnings and precautions
            - Mention the recommended maintenance interval if relevant
            - Highlight if some steps require professional assistance
            - Reference appropriate sections of the manual
            
            Response:""",
            
            'FEATURE_RESPONSE_PROMPT':
            """Based on the provided context and the user's question about a vehicle feature,
            provide a clear explanation of how the feature works and how to use it properly.
            
            Context information:
            {context}
            
            User question:
            {query}
            
            Your feature explanation should:
            - Clearly explain the purpose and benefits of the feature
            - Provide step-by-step instructions for using it
            - Note any limitations or conditions where it may not work
            - Mention related features or settings if relevant
            - Include any safety considerations
            - Reflect the exact terminology used in the vehicle's manual
            
            Response:"""
        }
        
        # Register each response type
        for name, template in response_types.items():
            if name not in self.templates:
                self.register_template(name, template)
    
    def get_template(self, template_name: str, default: Optional[str] = None) -> str:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template to retrieve
            default: Default template text if not found
            
        Returns:
            The prompt template text
            
        Raises:
            TemplateError: If template is not found and no default is provided
        """
        if template_name in self.templates:
            return self.templates[template_name]
        
        if default is not None:
            return default
        
        raise TemplateError(f"Template '{template_name}' not found and no default provided")
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """
        Format a template by substituting variables.
        
        Args:
            template_name: Name of the template to format
            **kwargs: Variables to substitute in the template
            
        Returns:
            The formatted prompt with variables substituted
            
        Raises:
            TemplateError: If template formatting fails
        """
        try:
            template = self.get_template(template_name)
            
            # Use string formatting for templates (safe with provided values)
            formatter = string.Formatter()
            return formatter.format(template, **kwargs)
        except KeyError as e:
            # This happens when a required template variable is missing
            raise TemplateError(f"Missing required template variable: {str(e)}")
        except Exception as e:
            # Other formatting errors
            raise TemplateError(f"Error formatting template '{template_name}': {str(e)}")
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the assistant.
        
        Returns:
            The system prompt defining the assistant's behavior
        """
        return self.get_template('SYSTEM_PROMPT')
    
    def get_query_prompt(self, query_type: str = "general") -> str:
        """
        Get prompt template for query processing based on query type.
        
        Args:
            query_type: Type of query (general, diagnostic, maintenance, etc.)
            
        Returns:
            The appropriate query processing prompt
        """
        template_name = f"{query_type.upper()}_QUERY_PROMPT"
        
        # Try to get a specialized template for this query type
        try:
            return self.get_template(template_name)
        except TemplateError:
            # Fall back to the general query understanding prompt
            return self.get_template('QUERY_UNDERSTANDING_PROMPT')
    
    def get_context_prompt(self) -> str:
        """
        Get prompt for context formatting.
        
        Returns:
            Template for structuring context information
        """
        return self.get_template('CONTEXT_FORMAT_PROMPT')
    
    def get_response_prompt(self, response_type: str = "general") -> str:
        """
        Get prompt template for response generation based on response type.
        
        Args:
            response_type: Type of response (general, instruction, diagnostic, etc.)
            
        Returns:
            The appropriate response generation prompt
        """
        template_name = f"{response_type.upper()}_RESPONSE_PROMPT"
        
        # Try to get a specialized template for this response type
        try:
            return self.get_template(template_name)
        except TemplateError:
            # Fall back to the general response prompt
            return self.get_template('RESPONSE_PROMPT')
    
    def get_voice_optimization_guidelines(self) -> str:
        """
        Get guidelines for voice-optimized responses.
        
        Returns:
            Guidelines for formatting responses for speech synthesis
        """
        return self.get_template('VOICE_OPTIMIZATION_GUIDELINES')
    
    def register_template(self, template_name: str, template_text: str) -> None:
        """
        Register a new template or update an existing one.
        
        Args:
            template_name: Name for the template
            template_text: The template text
        """
        self.templates[template_name] = template_text
        logging.debug(f"Registered template: {template_name}")
    
    def list_templates(self) -> List[str]:
        """
        List all available template names.
        
        Returns:
            List of template names
        """
        return sorted(list(self.templates.keys()))

    def get_category_templates(self, category: str) -> Dict[str, str]:
        """
        Get all templates belonging to a specific category.
        
        Args:
            category: Category prefix (e.g., 'DIAGNOSTIC', 'MAINTENANCE')
            
        Returns:
            Dictionary of template names to template texts for the category
        """
        category_prefix = f"{category.upper()}_"
        return {
            name: template 
            for name, template in self.templates.items() 
            if name.startswith(category_prefix)
        }

    def export_templates_to_config(self) -> None:
        """
        Export all templates to the configuration.
        
        This method updates the configuration with all templates in the manager,
        which can be useful for saving custom templates.
        """
        for name, template in self.templates.items():
            self.config.update_config('PROMPT_TEMPLATES', name, template)
        
        logging.info(f"Exported {len(self.templates)} templates to configuration")


# For testing or direct usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    template_manager = PromptTemplateManager()
    print(f"Available templates: {template_manager.list_templates()}")
    
    # Example usage
    try:
        formatted = template_manager.format_template(
            'RESPONSE_PROMPT',
            context="Section 7.3: The recommended oil change interval is every 5,000 miles or 6 months, whichever comes first. Use 5W-30 oil for temperatures above 0°F.",
            query="When should I change my oil?"
        )
        print("\nFormatted template:")
        print(formatted)
    except TemplateError as e:
        print(f"Template error: {str(e)}")