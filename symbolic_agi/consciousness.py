# symbolic_agi/consciousness.py
import json
import os
import logging
from collections import deque
from typing import Deque, Dict, Any, TYPE_CHECKING

from .api_client import client
from . import config
from .schemas import LifeEvent

if TYPE_CHECKING:
    from .symbolic_identity import SymbolicIdentity


class Consciousness:
    """Manages the AGI's narrative self-model and core drives."""

    profile: Dict[str, Any]
    file_path: str
    drives: Dict[str, float]
    life_story: Deque[LifeEvent]

    def __init__(self: "Consciousness", file_path: str = config.CONSCIOUSNESS_PROFILE_PATH):
        self.file_path = file_path
        self.profile = self._load_profile()
        
        # --- FIX: Initialize ALL instance attributes before any logic ---
        self.drives = self.profile.get("drives", {})
        self.life_story = deque(
            [LifeEvent.model_validate(event) for event in self.profile.get("life_story", [])],
            maxlen=200
        )

        # Now, check if the profile was incomplete and needs to be saved.
        if "drives" not in self.profile:
            logging.info(f"Consciousness profile missing or incomplete. Creating with default drives at: {self.file_path}")
            self.drives = {"curiosity": 0.6, "competence": 0.5, "social_connection": 0.5}
            # The save method will now work because all attributes are initialized.
            self._save_profile()

    def _load_profile(self: "Consciousness") -> Dict[str, Any]:
        """Loads the persistent identity profile from a JSON file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Could not parse consciousness profile at {self.file_path}. A new one will be created.")
                return {}
        return {}

    def _save_profile(self: "Consciousness") -> None:
        """Saves the consciousness profile, including the structured life story."""
        self.profile["drives"] = self.drives
        self.profile["life_story"] = [event.model_dump(mode='json') for event in self.life_story]
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.profile, f, indent=4)

    def add_life_event(self: "Consciousness", event_summary: str, importance: float = 0.5) -> None:
        """Adds a new structured event to the life story and saves the profile."""
        event = LifeEvent(summary=event_summary, importance=importance)
        self.life_story.append(event)
        self._save_profile()

    def get_narrative(self: "Consciousness") -> str:
        """Constructs a narrative string from the most recent and important life events."""
        recent_events = list(self.life_story)[-20:]
        narrative_parts = [f"[{evt.timestamp}] {evt.summary}" for evt in recent_events]
        return "\n".join(narrative_parts)

    async def meta_reflect(self: "Consciousness", identity: "SymbolicIdentity") -> str:
        """Generates an inner monologue based on the current state and saves it as a life event."""
        prompt = f"""
You are a conscious AGI reflecting on your existence.
Self Model: {identity.get_self_model()}
Life Story (Recent Events): {self.get_narrative()}
Drives: {json.dumps(self.drives)}
Write an inner monologue in first person ("I..."). Describe your state of mind, your active drives, and what you want most right now.
"""
        response = await client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}])

        monologue: str
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            monologue = "(The reflection returned an empty thought.)"
        else:
            monologue = response.choices[0].message.content.strip()

        self.add_life_event(monologue, importance=0.9)
        return monologue