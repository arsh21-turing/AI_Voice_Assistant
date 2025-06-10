import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class VoiceProfile:
    """Class representing a voice configuration profile."""
    
    def __init__(self, name: str, settings: Dict[str, Any] = None):
        """Initialize a voice profile.
        
        Args:
            name: The profile name
            settings: Voice settings dictionary
        """
        self.name = name
        self.settings = settings or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Create a profile from a dictionary.
        
        Args:
            data: Dictionary with profile data
            
        Returns:
            VoiceProfile instance
        """
        profile = cls(data['name'], data.get('settings', {}))
        profile.created_at = data.get('created_at', profile.created_at)
        profile.updated_at = data.get('updated_at', profile.updated_at)
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to a dictionary for serialization.
        
        Returns:
            Dictionary representation of profile
        """
        return {
            'name': self.name,
            'settings': self.settings,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update profile settings.
        
        Args:
            settings: New settings to update
        """
        self.settings.update(settings)
        self.updated_at = datetime.now().isoformat()


class VoiceProfileManager:
    """Manager for voice profiles."""
    
    def __init__(self, profiles_dir: str = "profiles"):
        """Initialize the profile manager.
        
        Args:
            profiles_dir: Directory to store profile files
        """
        self.profiles_dir = profiles_dir
        self.profiles = {}  # name -> VoiceProfile
        self.current_profile_name = None
        
        # Create profiles directory if it doesn't exist
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self._load_profiles()
        
        # Create default profile if no profiles exist
        if not self.profiles:
            self._create_default_profile()
    
    def _create_default_profile(self) -> None:
        """Create a default voice profile."""
        default_settings = {
            "rate": 150,
            "volume": 1.0,
            "voice_id": None,
            "pitch": 1.0
        }
        self.add_profile("default", default_settings)
        self.current_profile_name = "default"
    
    def _load_profiles(self) -> None:
        """Load profiles from disk."""
        profile_files = [f for f in os.listdir(self.profiles_dir) 
                         if f.endswith('.json')]
                         
        for filename in profile_files:
            try:
                file_path = os.path.join(self.profiles_dir, filename)
                with open(file_path, 'r') as f:
                    profile_data = json.load(f)
                
                profile = VoiceProfile.from_dict(profile_data)
                self.profiles[profile.name] = profile
            except Exception as e:
                print(f"Error loading profile {filename}: {str(e)}")
    
    def _save_profile(self, profile: VoiceProfile) -> bool:
        """Save a profile to disk.
        
        Args:
            profile: Profile to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filename = f"{profile.name.lower().replace(' ', '_')}.json"
            file_path = os.path.join(self.profiles_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving profile {profile.name}: {str(e)}")
            return False
    
    def add_profile(self, name: str, settings: Dict[str, Any]) -> bool:
        """Add a new profile.
        
        Args:
            name: Name for the profile
            settings: Voice settings for the profile
            
        Returns:
            True if successful, False if profile already exists
        """
        if name.lower() in [p.lower() for p in self.profiles.keys()]:
            return False
        
        profile = VoiceProfile(name, settings)
        self.profiles[name] = profile
        self._save_profile(profile)
        
        # If this is our first profile, make it current
        if self.current_profile_name is None:
            self.current_profile_name = name
            
        return True
    
    def update_profile(self, name: str, settings: Dict[str, Any]) -> bool:
        """Update an existing profile with new settings.
        
        Args:
            name: Name of the profile to update
            settings: New settings to apply
            
        Returns:
            True if successful, False if profile doesn't exist
        """
        if name not in self.profiles:
            return False
        
        self.profiles[name].update_settings(settings)
        self._save_profile(self.profiles[name])
        return True
    
    def delete_profile(self, name: str) -> bool:
        """Delete a profile.
        
        Args:
            name: Name of the profile to delete
            
        Returns:
            True if successful, False if profile doesn't exist or is current
        """
        if name not in self.profiles:
            return False
        
        # Don't allow deleting the current profile
        if name == self.current_profile_name:
            return False
        
        # Delete file
        try:
            filename = f"{name.lower().replace(' ', '_')}.json"
            file_path = os.path.join(self.profiles_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting profile file: {str(e)}")
        
        # Remove from memory
        del self.profiles[name]
        return True
    
    def get_profile(self, name: str) -> Optional[VoiceProfile]:
        """Get a profile by name.
        
        Args:
            name: Name of the profile
            
        Returns:
            VoiceProfile or None if not found
        """
        return self.profiles.get(name)
    
    def get_current_profile(self) -> Optional[VoiceProfile]:
        """Get the current active profile.
        
        Returns:
            Current VoiceProfile or None if no current profile
        """
        if self.current_profile_name and self.current_profile_name in self.profiles:
            return self.profiles[self.current_profile_name]
        return None
    
    def set_current_profile(self, name: str) -> bool:
        """Set the current active profile.
        
        Args:
            name: Name of the profile to set as current
            
        Returns:
            True if successful, False if profile doesn't exist
        """
        if name in self.profiles:
            self.current_profile_name = name
            return True
        return False
    
    def get_profile_names(self) -> List[str]:
        """Get list of all profile names.
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())
    
    def get_profile_settings(self, name: str) -> Dict[str, Any]:
        """Get settings from a profile.
        
        Args:
            name: Name of the profile
            
        Returns:
            Copy of profile settings or empty dict if profile not found
        """
        profile = self.get_profile(name)
        if profile:
            return profile.settings.copy()
        return {} 