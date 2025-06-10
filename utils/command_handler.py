import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from utils.voice_profile import VoiceProfileManager

class CommandHandler:
    """Handler for voice commands to control the assistant's behavior."""
    
    def __init__(self, profiles_dir: str = "profiles"):
        """Initialize the command handler with supported commands."""
        # Store the last response for repeat command
        self.last_response = ""
        self.command_history = []
        
        # Initialize profile manager
        self.profile_manager = VoiceProfileManager(profiles_dir)
        
        # Define command patterns and their handlers
        self.commands = {
            # Basic commands
            r"speak (slower|faster)": self._handle_speech_rate,
            r"(speed|slow|fast) (up|down|slower|faster)": self._handle_speech_rate,
            r"(increase|decrease) (speed|rate)": self._handle_speech_rate,
            r"(volume|louder|quieter) (up|down)": self._handle_volume,
            r"(increase|decrease) volume": self._handle_volume,
            r"(louder|quieter)": self._handle_volume,
            r"(repeat|say) (that|again)": self._handle_repeat,
            r"what (did you say|was that)": self._handle_repeat,
            r"(pause|stop)": self._handle_pause,
            r"(resume|continue)": self._handle_resume,
            r"reset (settings|voice)": self._handle_reset,
            r"default (settings|voice)": self._handle_reset,
            r"(voice|speech) commands": self._handle_help,
            r"what (commands|can I say)": self._handle_help,
            r"use (male|female|different) voice": self._handle_voice_selection,
            
            # Profile commands
            r"save (this|current) (profile|voice) as (.+)": self._handle_save_profile,
            r"create (profile|voice profile) (.+)": self._handle_save_profile,
            r"switch to (profile|voice) (.+)": self._handle_switch_profile,
            r"use (profile|voice profile) (.+)": self._handle_switch_profile,
            r"list (profiles|voice profiles)": self._handle_list_profiles,
            r"what profiles (do I have|are available)": self._handle_list_profiles,
            r"delete profile (.+)": self._handle_delete_profile,
            r"remove profile (.+)": self._handle_delete_profile,
            r"current profile": self._handle_current_profile,
            r"which profile": self._handle_current_profile,
        }
    
    def process_command(self, text: str, settings: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Process a potential voice command.
        
        Args:
            text: The user's input text
            settings: Current voice settings dictionary
            
        Returns:
            Tuple of (is_command, response_text, updated_settings)
        """
        # Convert to lowercase for better matching
        text = text.lower().strip()
        
        # Try to match a command
        for pattern, handler in self.commands.items():
            match = re.search(pattern, text)
            if match:
                # Add to command history
                self.command_history.append(text)
                
                # Process the command with match groups if needed
                response_text, updated_settings = handler(text, settings.copy(), match)
                return True, response_text, updated_settings
        
        # Not a command
        return False, None, settings
    
    def set_last_response(self, response: str) -> None:
        """Store the last response for the repeat command.
        
        Args:
            response: The last response from the assistant
        """
        self.last_response = response
    
    def _handle_speech_rate(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle commands to adjust speech rate.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, updated_settings)
        """
        current_rate = settings.get("rate", 150)
        
        # Determine if we need to speed up or slow down
        if any(term in text for term in ["slower", "slow down", "decrease"]):
            # Slow down (decrease rate)
            new_rate = max(80, current_rate - 25)
            response = "I'll speak slower now."
        else:
            # Speed up (increase rate)
            new_rate = min(250, current_rate + 25)
            response = "I'll speak faster now."
        
        # Update settings
        settings["rate"] = new_rate
        
        # Update current profile if one is active
        if self.profile_manager.current_profile_name:
            self.profile_manager.update_profile(
                self.profile_manager.current_profile_name,
                {"rate": new_rate}
            )
        
        return response, settings
    
    def _handle_volume(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle commands to adjust volume.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, updated_settings)
        """
        current_volume = settings.get("volume", 1.0)
        
        # Determine if we need to increase or decrease volume
        if any(term in text for term in ["down", "quieter", "decrease"]):
            # Decrease volume
            new_volume = max(0.3, current_volume - 0.2)
            response = "I've lowered my volume."
        else:
            # Increase volume
            new_volume = min(1.0, current_volume + 0.2)
            response = "I've increased my volume."
        
        # Update settings
        settings["volume"] = new_volume
        
        # Update current profile if one is active
        if self.profile_manager.current_profile_name:
            self.profile_manager.update_profile(
                self.profile_manager.current_profile_name,
                {"volume": new_volume}
            )
        
        return response, settings
    
    def _handle_repeat(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to repeat the last response.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        if self.last_response:
            return "I'll repeat my last response: " + self.last_response, settings
        else:
            return "I don't have any previous response to repeat.", settings
    
    def _handle_pause(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to pause.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        return "Pausing.", settings
    
    def _handle_resume(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to resume.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        return "Resuming.", settings
    
    def _handle_reset(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to reset voice settings.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, updated_settings)
        """
        # Reset to default settings
        default_settings = {
            "rate": 150,
            "volume": 1.0,
            "voice_id": settings.get("voice_id")  # Keep current voice_id
        }
        
        # Update current profile if one is active
        if self.profile_manager.current_profile_name:
            self.profile_manager.update_profile(
                self.profile_manager.current_profile_name,
                default_settings
            )
        
        return "I've reset my voice settings to default.", default_settings
    
    def _handle_help(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle help command for voice commands.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        help_text = (
            "Here are the voice commands you can use: "
            "speak slower or faster, volume up or down, repeat that, "
            "use male or female voice, reset voice. "
            "For profiles, you can say: save profile as [name], "
            "use profile [name], list profiles, delete profile [name]. "
            "You can also ask 'what profile am I using' to check the current profile."
        )
        
        return help_text, settings
    
    def _handle_voice_selection(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to change voice.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, updated_settings)
        """
        # This will be handled by the VoiceAssistant class since it requires
        # access to available voices, but we'll provide the response
        if "male" in text:
            response = "Switching to a male voice."
            settings["requested_voice"] = "male"
        elif "female" in text:
            response = "Switching to a female voice."
            settings["requested_voice"] = "female"
        else:
            response = "Switching to a different voice."
            settings["requested_voice"] = "different"
        
        return response, settings
    
    def _handle_save_profile(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to save current settings as a profile.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        try:
            # Extract profile name from the command
            if "save this" in text or "save current" in text:
                # Pattern: save this/current profile as NAME
                profile_name = match.group(3).strip()
            else:
                # Pattern: create profile NAME
                profile_name = match.group(2).strip()
            
            # Ensure name isn't empty
            if not profile_name or profile_name in ['as', 'called', 'named']:
                return "I need a valid name for the profile. Please try again.", settings
            
            # Try to add the profile
            success = self.profile_manager.add_profile(profile_name, settings)
            
            if success:
                # Set as current profile
                self.profile_manager.set_current_profile(profile_name)
                return f"I've saved your current voice settings as profile '{profile_name}' and switched to it.", settings
            else:
                return f"A profile named '{profile_name}' already exists. Please use a different name or delete the existing profile first.", settings
            
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
            return "I had trouble saving that profile. Please try again.", settings
    
    def _handle_switch_profile(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to switch to a different profile.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, updated_settings)
        """
        try:
            # Extract profile name from the command
            profile_name = match.group(2).strip()
            
            # Try to switch profiles
            if self.profile_manager.set_current_profile(profile_name):
                # Get profile settings
                new_settings = self.profile_manager.get_profile_settings(profile_name)
                return f"I've switched to the '{profile_name}' voice profile.", new_settings
            else:
                # Try to find a close match
                available_profiles = self.profile_manager.get_profile_names()
                closest_match = None
                for p in available_profiles:
                    if profile_name.lower() in p.lower():
                        closest_match = p
                        break
                
                if closest_match:
                    suggestion = f" Did you mean '{closest_match}'?"
                else:
                    suggestion = ""
                
                return f"I couldn't find a profile named '{profile_name}'.{suggestion} Available profiles are: {', '.join(available_profiles)}", settings
            
        except Exception as e:
            print(f"Error switching profiles: {str(e)}")
            return "I had trouble switching profiles. Please try again.", settings
    
    def _handle_list_profiles(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to list available profiles.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        try:
            profiles = self.profile_manager.get_profile_names()
            current = self.profile_manager.current_profile_name
            
            if not profiles:
                return "You don't have any voice profiles saved yet. You can create one by saying 'save this profile as' followed by a name.", settings
            
            response = f"You have {len(profiles)} voice profiles: "
            for i, name in enumerate(profiles):
                if name == current:
                    response += f"'{name}' (current)"
                else:
                    response += f"'{name}'"
                    
                if i < len(profiles) - 1:
                    response += ", "
            
            response += ". You can switch profiles by saying 'use profile' followed by the name."
            return response, settings
            
        except Exception as e:
            print(f"Error listing profiles: {str(e)}")
            return "I had trouble retrieving your profiles. Please try again.", settings
    
    def _handle_delete_profile(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to delete a profile.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        try:
            # Extract profile name from the command
            profile_name = match.group(1).strip()
            
            # Try to delete the profile
            if self.profile_manager.delete_profile(profile_name):
                return f"I've deleted the '{profile_name}' voice profile.", settings
            else:
                # Check if it's the current profile
                if profile_name == self.profile_manager.current_profile_name:
                    return f"I can't delete the '{profile_name}' profile because it's currently in use. Please switch to another profile first.", settings
                else:
                    return f"I couldn't find a profile named '{profile_name}' to delete.", settings
            
        except Exception as e:
            print(f"Error deleting profile: {str(e)}")
            return "I had trouble deleting that profile. Please try again.", settings
    
    def _handle_current_profile(self, text: str, settings: Dict[str, Any], match) -> Tuple[str, Dict[str, Any]]:
        """Handle command to report the current profile.
        
        Args:
            text: The command text
            settings: Current voice settings
            match: Regex match object
            
        Returns:
            Tuple of (response_text, unchanged_settings)
        """
        try:
            current = self.profile_manager.current_profile_name
            if current:
                return f"You're currently using the '{current}' voice profile.", settings
            else:
                return "You're not using any saved voice profile.", settings
            
        except Exception as e:
            print(f"Error getting current profile: {str(e)}")
            return "I had trouble determining the current profile. Please try again.", settings 