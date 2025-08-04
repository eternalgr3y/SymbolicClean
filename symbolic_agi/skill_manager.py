# symbolic_agi/skill_manager.py

import json
import os
import logging
from typing import List, Dict, Optional

from .schemas import SkillModel, ActionStep
from . import config

class SkillManager:
    """Manages loading, saving, and using learned skills."""
    def __init__(self, file_path: str = config.SKILLS_PATH):
        self.file_path = file_path
        self.skills: Dict[str, SkillModel] = self._load_skills()
        self.innate_actions = self._get_innate_actions()

    def _load_skills(self) -> Dict[str, SkillModel]:
        """Loads skills from a JSON file using Pydantic validation."""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {skill_id: SkillModel.model_validate(props) for skill_id, props in data.items()}
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Could not load skills from {self.file_path}: {e}")
            return {}

    def _save_skills(self):
        """Saves the current skills to a JSON file."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump({skill_id: skill.model_dump(mode='json') for skill_id, skill in self.skills.items()}, f, indent=4)

    def add_new_skill(self, name: str, description: str, plan: List[ActionStep]):
        """Creates a new skill from a successful plan and saves it, overwriting if the name exists."""
        # --- UPGRADE: More robustly handle overwriting skills by name ---
        existing_skill_id = next((sid for sid, s in self.skills.items() if s.name == name), None)
        if existing_skill_id:
            logging.warning(f"Skill with name '{name}' already exists. Overwriting.")
            del self.skills[existing_skill_id]

        new_skill = SkillModel(
            name=name,
            description=description,
            action_sequence=plan
        )
        self.skills[new_skill.id] = new_skill
        self._save_skills()
        logging.info(f"Successfully learned and saved new skill: '{name}'")

    # --- NEW METHOD ---
    def get_skill_by_name(self, name: str) -> Optional[SkillModel]:
        """Finds a skill by its unique name."""
        for skill in self.skills.values():
            if skill.name == name:
                return skill
        return None

    # --- NEW METHOD ---
    def is_skill(self, action_name: str) -> bool:
        """Checks if a given action name corresponds to a learned skill."""
        return any(skill.name == action_name for skill in self.skills.values())

    def get_formatted_definitions(self) -> str:
        """Returns a formatted string of all available skills and actions for the planner."""
        formatted_string = "# INNATE ACTIONS\n"
        for name, params in self.innate_actions.items():
            formatted_string += f'action: "{name}", {params}\n'
        
        if self.skills:
            # --- UPGRADE: Make it clearer to the LLM that these are learned skills it can use ---
            formatted_string += "\n# LEARNED SKILLS (use these like high-level actions)\n"
            for skill in self.skills.values():
                # The persona for a skill is always the orchestrator, as it expands the plan
                formatted_string += f'action: "{skill.name}", description: "{skill.description}", assigned_persona: "orchestrator"\n'
        
        return formatted_string

    def _get_innate_actions(self) -> Dict[str, str]:
        """Returns a dictionary of the AGI's hardcoded abilities for the planner."""
        # --- UPGRADE: Removed meta-actions that the planner should not call directly ---
        return {
            "research_topic": 'parameters: {"topic": "..."}',
            "write_code": 'parameters: {"prompt": "...", "context": "..."}',
            "review_code": 'parameters: {"code": "..."}',
            "respond_to_user": 'parameters: {"text": "..."}',
        }