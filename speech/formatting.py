"""
Voice response formatting for natural-sounding speech output.

This module provides the VoiceResponseFormatter class which optimizes text responses
from the LLM for natural-sounding speech output, handling pauses, emphasis,
acronyms, numbers, and technical terminology.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import json
import string
import random

class VoiceResponseFormatter:
    """Optimizes text responses for natural-sounding speech output."""
    
    def __init__(self, formatting_config=None):
        """Initialize the formatter with speech parameters from configuration.
        
        Args:
            formatting_config: Dictionary of formatting configuration options
        """
        self.config = formatting_config or {}
        
        # Default configuration values
        self.use_ssml = self.config.get('use_ssml', False)
        self.pause_words = self.config.get('pause_words', 
            ["however", "additionally", "furthermore", "nevertheless"])
        self.emphasis_keywords = self.config.get('emphasis_keywords', 
            ["warning", "caution", "important", "note"])
        self.number_formats = self.config.get('number_formats', {})
        self.acronym_mappings = self.config.get('acronym_mappings', {})
        self.technical_terms = self.config.get('technical_terms', {})
        
        # Initialize pause durations
        self.pause_durations = {
            'light': {'comma': 250, 'period': 500, 'paragraph': 750, 'bullet': 350, 'important': 400},
            'medium': {'comma': 350, 'period': 650, 'paragraph': 1000, 'bullet': 450, 'important': 500},
            'strong': {'comma': 450, 'period': 800, 'paragraph': 1250, 'bullet': 550, 'important': 600}
        }
        self.durations = self.pause_durations.get('medium')  # Default to medium pause strength
        
        # Load any custom terminology
        self.load_terminology()
        logging.info("Initialized voice response formatter")

    def load_terminology(self):
        """Load terminology mappings for acronyms and technical terms."""
        self.terminology = {
            'ABS': 'A B S', 'TPMS': 'T P M S', 'ECU': 'E C U', 'OBD': 'O B D',
            'OBD-II': 'O B D two', 'RPM': 'R P M', 'MPG': 'miles per gallon',
            'MPH': 'miles per hour', 'CVT': 'C V T', 'A/C': 'air conditioning',
            'torque': 'tork', 'chassis': 'chassee', '5W-30': 'five W thirty',
            '10W-40': 'ten W forty', 'psi': 'pounds per square inch',
            'lbs': 'pounds', '0W-20': 'zero W twenty', 'liter': 'leeter',
        }
        
        # Update with any custom mappings from config
        self.terminology.update(self.acronym_mappings)
        self.terminology.update(self.technical_terms)

    def format_response(self, text: str) -> str:
        """Format a response for optimal speech synthesis.

        Args:
            text: Raw text response from the LLM

        Returns:
            Formatted text optimized for speech synthesis
        """
        if not text:
            return ""

        text = self.format_numbers(text)
        text = self.format_units(text)
        text = self.format_acronyms(text)
        text = self.format_technical_terms(text)
        text = self.format_section_references(text)
        text = self.add_pauses(text)

        if self.use_ssml:
            text = self.wrap_ssml(text)

        return text

    def format_numbers(self, text):
        """Format numbers for natural speech."""
        # Apply custom number formats from config
        for pattern, replacement in self.number_formats.items():
            text = re.sub(pattern, replacement, text)
            
        # Format decimal numbers
        text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)
        
        # Format number ranges (e.g., "5,000-7,500")
        def format_range(match):
            start = match.group(1).replace(',', '')
            end = match.group(2).replace(',', '')
            return f"{self._format_large_number(start)} to {self._format_large_number(end)}"
        text = re.sub(r'(\d+(?:,\d+)*)-(\d+(?:,\d+)*)(?!\w)', format_range, text)
        
        # Format large numbers with commas
        text = re.sub(r'(\d+(?:,\d+)*)', lambda m: self._format_large_number(m.group(1)), text)
        
        return text
        
    def _format_large_number(self, num_str: str) -> str:
        """Format a large number for speech.
        
        Args:
            num_str: Number string that may contain commas
            
        Returns:
            Formatted number string for speech
        """
        # Remove commas and convert to integer
        num = int(num_str.replace(',', ''))
        
        # Format based on size
        if num >= 1000000:
            millions = num // 1000000
            remainder = num % 1000000
            if remainder == 0:
                return f"{millions} million"
            return f"{millions} million {self._format_large_number(str(remainder))}"
        elif num >= 1000:
            thousands = num // 1000
            remainder = num % 1000
            if remainder == 0:
                return f"{thousands} thousand"
            return f"{thousands} thousand {self._format_large_number(str(remainder))}"
        else:
            return str(num)

    def format_units(self, text):
        """Format units for natural speech."""
        units = {'psi': 'pounds per square inch', 'mph': 'miles per hour', 'rpm': 'R P M'}
        for k, v in units.items():
            text = re.sub(rf'(\d+)\s*{k}\b', rf'\1 {v}', text)
        return text

    def format_acronyms(self, text):
        """Format acronyms for natural speech."""
        for k, v in self.terminology.items():
            if re.match(r'^[A-Z\-/]+$', k):
                text = re.sub(rf'\b{k}\b', v, text)
        return text

    def format_technical_terms(self, text):
        """Format technical terms for natural speech."""
        for k, v in self.terminology.items():
            if not re.match(r'^[A-Z\-/]+$', k):
                text = re.sub(rf'\b{k}\b', v, text, flags=re.IGNORECASE)
        return text

    def format_section_references(self, text):
        """Format section references for natural speech."""
        text = re.sub(r'Section (\d+)\.(\d+)', r'Section \1 point \2', text)
        return text

    def add_pauses(self, text):
        """Add natural pauses to the text."""
        if not self.use_ssml:
            return text
            
        # Add pauses after punctuation
        text = re.sub(r'(?<=\.)\s+', f'<break time="{self.durations["period"]}ms"/> ', text)
        text = re.sub(r'(?<=,)\s+', f'<break time="{self.durations["comma"]}ms"/> ', text)
        
        # Add pauses before important words
        for word in self.pause_words:
            text = re.sub(rf'\b{word}\b', f'<break time="{self.durations["important"]}ms"/> {word}', text)
            
        # Add emphasis to important keywords
        for word in self.emphasis_keywords:
            text = re.sub(rf'\b{word}\b', f'<emphasis level="strong">{word}</emphasis>', text)
            
        return text

    def wrap_ssml(self, text):
        """Wrap the text in SSML tags."""
        return f'<speak>{text}</speak>'

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    formatter = VoiceResponseFormatter()
    test_input = """
    To change your oil, follow these steps:

    1. Make sure your engine is cool, but slightly warm. Park on level ground.
    2. Locate the oil drain plug underneath your vehicle.
    3. Place a container (at least 5qt capacity) under the drain plug.
    4. Use a 14mm wrench to loosen the drain plug.
    5. Once loose, unscrew the plug by hand - careful, the oil will be warm!
    6. Allow all oil to drain (about 5-10 minutes).
    7. Replace the drain plug, tightening to 25 ft-lbs of torque.

    According to Section 7.3 of your manual, the recommended oil is 5W-30 for temperatures above 32°F, or 0W-20 for temperatures below 32°F. Your vehicle needs 4.5qt of oil.

    WARNING: Never start the engine without oil or damage to the engine can occur.

    The OBD-II system may trigger a code P0300 if oil quality is poor.

    Additional information:
    - The engine produces 270hp at 5,500rpm
    - The vehicle has a 3.5L V6 engine
    - The temperature should be between 195°F and 220°F
    - The oil change interval is 5,000-7,500 miles
    - The fuel tank capacity is 15.5 gallons
    - The vehicle weighs 3,500 pounds
    - The engine displacement is 3,456 cubic centimeters
    """
    formatted = formatter.format_response(test_input)
    print("Original text:")
    print(test_input)
    print("\nFormatted for speech:")
    print(formatted)
