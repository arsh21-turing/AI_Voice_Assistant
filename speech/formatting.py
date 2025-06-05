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

# Import from project modules
try:
    from config import get_config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from config import get_config

class VoiceResponseFormatter:
    """
    Formatter for optimizing text responses for speech synthesis.
    """

    def __init__(self, config_manager=None):
        self.config = config_manager if config_manager else get_config()
        self.use_ssml = self.config.get('SPEECH_SETTINGS', 'USE_SSML', False)
        self.auto_pause_before_important = self.config.get('SPEECH_SETTINGS', 'AUTO_PAUSE_BEFORE_IMPORTANT', True)
        self.pause_strength = self.config.get('SPEECH_SETTINGS', 'PAUSE_STRENGTH', 'medium')
        self.emphasis_level = self.config.get('SPEECH_SETTINGS', 'EMPHASIS_LEVEL', 'moderate')
        self.acronym_mode = self.config.get('SPEECH_SETTINGS', 'ACRONYM_MODE', 'spelled_first')
        self.technical_term_handling = self.config.get('SPEECH_SETTINGS', 'TECHNICAL_TERM_HANDLING', 'careful')

        self.pause_durations = {
            'light': {'comma': 250, 'period': 500, 'paragraph': 750, 'bullet': 350, 'important': 400},
            'medium': {'comma': 350, 'period': 650, 'paragraph': 1000, 'bullet': 450, 'important': 500},
            'strong': {'comma': 450, 'period': 800, 'paragraph': 1250, 'bullet': 550, 'important': 600}
        }
        self.durations = self.pause_durations.get(self.pause_strength, self.pause_durations['medium'])
        self.load_terminology()
        logging.info("Initialized voice response formatter")

    def load_terminology(self):
        self.terminology = {
            'ABS': 'A B S', 'TPMS': 'T P M S', 'ECU': 'E C U', 'OBD': 'O B D',
            'OBD-II': 'O B D two', 'RPM': 'R P M', 'MPG': 'miles per gallon',
            'MPH': 'miles per hour', 'CVT': 'C V T', 'A/C': 'air conditioning',
            'torque': 'tork', 'chassis': 'chassee', '5W-30': 'five W thirty',
            '10W-40': 'ten W forty', 'psi': 'pounds per square inch',
            'lbs': 'pounds', '0W-20': 'zero W twenty', 'liter': 'leeter',
        }
        try:
            terminology_path = self.config.get('SPEECH_SETTINGS', 'TERMINOLOGY_PATH', './data/speech/automotive_pronunciations.json')
            if Path(terminology_path).exists():
                with open(terminology_path, 'r') as f:
                    custom_terminology = json.load(f)
                    self.terminology.update(custom_terminology)
                logging.info(f"Loaded custom terminology from {terminology_path}")
        except Exception as e:
            logging.warning(f"Could not load custom terminology: {str(e)}")

    def format_response(self, text: str) -> str:
        """
        Format a response for optimal speech synthesis.

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
        text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)
        text = re.sub(r'(\d+)-(\d+)(?!\w)', r'\1 to \2', text)
        return text

    def format_units(self, text):
        units = {'psi': 'pounds per square inch', 'mph': 'miles per hour', 'rpm': 'R P M'}
        for k, v in units.items():
            text = re.sub(rf'(\d+)\s*{k}\b', rf'\1 {v}', text)
        return text

    def format_acronyms(self, text):
        for k, v in self.terminology.items():
            if re.match(r'^[A-Z\-/]+$', k):
                text = re.sub(rf'\b{k}\b', v, text)
        return text

    def format_technical_terms(self, text):
        for k, v in self.terminology.items():
            if not re.match(r'^[A-Z\-/]+$', k):
                text = re.sub(rf'\b{k}\b', v, text, flags=re.IGNORECASE)
        return text

    def format_section_references(self, text):
        text = re.sub(r'Section (\d+)\.(\d+)', r'Section \1 point \2', text)
        return text

    def add_pauses(self, text):
        if not self.use_ssml:
            return text
        text = re.sub(r'(?<=\.)\s+', f'<break time="{self.durations["period"]}ms"/> ', text)
        text = re.sub(r'(?<=,)\s+', f'<break time="{self.durations["comma"]}ms"/> ', text)
        return text

    def wrap_ssml(self, text):
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
    """
    formatted = formatter.format_response(test_input)
    print("Original text:")
    print(test_input)
    print("\nFormatted for speech:")
    print(formatted)
