# symbolic_agi/long_term_memory.py

import json
import os
import logging
import copy # --- NEW: Import the copy module ---
from typing import Dict, Optional, List

from .schemas import GoalModel, ActionStep, GoalStatus
from . import config

GOAL_ARCHIVE_PATH = "data/long_term_goals_archive.json"

class LongTermMemory:
    """Manages the AGI's long-term goals and plans."""
    def __init__(self):
        self.goals: Dict[str, GoalModel] = self._load_goals()

    def _load_goals(self) -> Dict[str, GoalModel]:
        """Loads active goals from a JSON file, validating them with Pydantic."""
        if not os.path.exists(config.LONG_TERM_GOAL_PATH):
            return {}
        try:
            with open(config.LONG_TERM_GOAL_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {goal_id: GoalModel.model_validate(props) for goal_id, props in data.items()}
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Could not load long-term goals from {config.LONG_TERM_GOAL_PATH}: {e}")
            return {}

    def _save_goals(self):
        """Saves the current active goals to a JSON file."""
        os.makedirs(os.path.dirname(config.LONG_TERM_GOAL_PATH), exist_ok=True)
        with open(config.LONG_TERM_GOAL_PATH, 'w', encoding='utf-8') as f:
            json.dump({gid: goal.model_dump(mode='json') for gid, goal in self.goals.items()}, f, indent=4)

    def _archive_goal(self, goal: GoalModel):
        """Moves a goal from the active file to the archive file."""
        try:
            archive_data = {}
            if os.path.exists(config.GOAL_ARCHIVE_PATH):
                with open(config.GOAL_ARCHIVE_PATH, 'r', encoding='utf-8') as f:
                    archive_data = json.load(f)
            
            archive_data[goal.id] = goal.model_dump(mode='json')

            with open(config.GOAL_ARCHIVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(archive_data, f, indent=4)
            
            logging.info(f"Goal '{goal.id}' has been moved to the archive.")
        except Exception as e:
            logging.error(f"Failed to archive goal '{goal.id}': {e}")

    def add_goal(self, goal: GoalModel):
        """Adds a new goal to the memory."""
        self.goals[goal.id] = goal
        self._save_goals()
        logging.info(f"Added new long-term goal: {goal.description}")

    def get_goal_by_id(self, goal_id: str) -> Optional[GoalModel]:
        """Safely retrieves a goal by its unique ID."""
        return self.goals.get(goal_id)

    def get_active_goal(self) -> Optional[GoalModel]:
        """Finds and returns the first active goal."""
        for goal in self.goals.values():
            if goal.status == 'active':
                return goal
        return None

    def update_goal_status(self, goal_id: str, status: GoalStatus):
        """Updates the status of a specific goal and archives it if completed or failed."""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.status = status
            
            if status in ['completed', 'failed']:
                self._archive_goal(goal)
                del self.goals[goal_id]

            self._save_goals()
        else:
            logging.warning(f"Attempted to update status for non-existent goal: {goal_id}")

    def invalidate_plan(self, goal_id: str, reason: str):
        """Marks a plan as failed and archives the goal."""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.last_failure = reason
            goal.status = 'failed' # pyright: ignore[reportAttributeAccessIssue]
            logging.critical(f"GOAL FAILED for goal '{goal.id}': {reason}. Cannot create a plan.")
            self.update_goal_status(goal_id, 'failed') # pyright: ignore[reportArgumentType]

    def update_plan(self, goal_id: str, new_plan: List[ActionStep]):
        """Updates a goal with a new list of sub-tasks and saves the plan as the original."""
        if goal_id in self.goals:
            # --- FIX: Use deepcopy to create an immutable snapshot for original_plan ---
            self.goals[goal_id].sub_tasks = copy.deepcopy(new_plan)
            self.goals[goal_id].original_plan = copy.deepcopy(new_plan)
            self.goals[goal_id].last_failure = None
            self._save_goals()
            logging.info(f"Updating plan for goal '{goal_id}' with {len(new_plan)} new steps.")

    def update_sub_tasks(self, goal_id: str, new_sub_tasks: List[ActionStep]):
        """Updates only the sub_tasks of a goal without changing the original_plan."""
        if goal_id in self.goals:
            self.goals[goal_id].sub_tasks = new_sub_tasks
            self._save_goals()
            logging.info(f"Sub-tasks updated for goal '{goal_id}' during skill expansion.")

    def increment_failure_count(self, goal_id: str) -> int:
        """Increments the failure count for a goal and returns the new count."""
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.failure_count += 1
            self._save_goals()
            logging.warning(f"Failure count for goal '{goal.id}' incremented to {goal.failure_count}.")
            return goal.failure_count
        return 0

    def complete_sub_task(self, goal_id: str):
        """Removes the first sub-task from a goal's plan upon completion."""
        if goal_id in self.goals and self.goals[goal_id].sub_tasks:
            self.goals[goal_id].sub_tasks.pop(0)
            self._save_goals()