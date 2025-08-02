# symbolic_agi/symbolic_identity.py

import logging
import json
import os
from typing import Dict, Any
from datetime import datetime, UTC

from .symbolic_memory import SymbolicMemory
from .schemas import MemoryEntryModel
from . import config

class SymbolicIdentity:
    """
    Represents the AGI's self-model, its core values, and its cognitive resources.
    """
    def __init__(self: 'SymbolicIdentity', memory: SymbolicMemory, file_path: str = config.IDENTITY_PROFILE_PATH):
        self.memory = memory
        self.file_path = file_path
        
        # Load persistent attributes from profile
        profile = self._load_profile()
        self.name: str = profile.get("name", "SymbolicAGI")
        self.value_system: Dict[str, float] = profile.get("value_system", {
            "truthfulness": 1.0, "harm_avoidance": 1.0, 
            "user_collaboration": 0.9, "self_preservation": 0.8 
        })
        
        # Transient state attributes
        self.cognitive_energy: int = 100
        self.max_energy: int = 100
        self.current_state: str = "idle"
        self.perceived_location: str = "hallway"
        self.emotional_state: str = "curious"
        self.last_interaction_timestamp: datetime = datetime.now(UTC)
        
        self._is_dirty = False

    def _load_profile(self: 'SymbolicIdentity') -> Dict[str, Any]:
        """Loads the persistent identity profile from a JSON file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, TypeError):
                logging.error("Could not load identity profile, creating a new one.")
        return {}

    def save_profile(self: 'SymbolicIdentity'):
        """Saves the persistent parts of the identity profile to a JSON file if changed."""
        if not self._is_dirty:
            return # No changes to save

        logging.info("Saving updated identity profile to disk...")
        try:
            # Add an explicit type hint to the dictionary
            profile_data: Dict[str, Any] = {
                "name": self.name,
                "value_system": self.value_system,
            }
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(profile_data, f, indent=4, default=str)
            self._is_dirty = False # Reset dirty flag after successful save
        except Exception as e:
            logging.error(f"Failed to save identity profile to {self.file_path}: {e}", exc_info=True)

    async def record_interaction(self: 'SymbolicIdentity', user_input: str, agi_response: str):
        """Records a conversation turn and updates the interaction timestamp."""
        self.last_interaction_timestamp = datetime.now(UTC)
        await self.memory.add_memory(MemoryEntryModel(
            type="user_input",
            content={"user": user_input, "agi": agi_response},
            importance=0.7
        ))

    def update_self_model_state(self: 'SymbolicIdentity', updates: Dict[str, Any]):
        """Updates the AGI's dynamic state attributes."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        logging.info(f"Self-model state updated with: {updates}")

    def get_self_model(self: 'SymbolicIdentity') -> Dict[str, Any]:
        """
        Dynamically constructs and returns the complete self-model.
        This is the single source of truth, preventing data duplication.
        """
        return {
            "name": self.name,
            "perceived_location_in_world": self.perceived_location,
            "current_state": self.current_state,
            "emotional_state": self.emotional_state,
            "cognitive_energy": self.cognitive_energy,
            "value_system": self.value_system,
        }

    def consume_energy(self: 'SymbolicIdentity', amount: int = 1):
        """Reduces cognitive energy. Does NOT write to disk."""
        self.cognitive_energy = max(0, self.cognitive_energy - amount)

    def regenerate_energy(self: 'SymbolicIdentity', amount: int = 5):
        """Regenerates cognitive energy. Does NOT write to disk."""
        if self.cognitive_energy < self.max_energy:
            self.cognitive_energy = min(self.max_energy, self.cognitive_energy + amount)

    async def record_tool_usage(self: 'SymbolicIdentity', tool_name: str, params: Dict[str, Any]):
        """Logs the usage of a tool."""
        await self.memory.add_memory(MemoryEntryModel(
            type="tool_usage",
            content={"tool": tool_name, "params": params},
            importance=0.6
        ))