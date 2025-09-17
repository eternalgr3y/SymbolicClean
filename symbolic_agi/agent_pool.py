# symbolic_agi/agent_pool.py

import logging
from typing import Any, Dict, List, Optional

from .message_bus import MessageBus
# --- FIX: Removed unused MessageModel import ---
from .skill_manager import SkillManager
from .symbolic_identity import SymbolicIdentity
from .symbolic_memory import SymbolicMemory

class DynamicAgentPool:
    """Manages a collection of sub-agents with different personas."""
    def __init__(self: 'DynamicAgentPool', bus: MessageBus) -> None:
        self.subagents: Dict[str, Dict[str, Any]] = {}
        self.bus: MessageBus = bus
        logging.info("[AgentPool] Initialized.")

    def add_agent(self: 'DynamicAgentPool', name: str, persona: str, memory: SymbolicMemory, skills: Optional[SkillManager] = None) -> None:
        """Adds a new sub-agent to the pool."""
        self.subagents[name] = {
            "name": name,
            "persona": persona.lower(),
            "skills": skills or SkillManager(),
            "identity": SymbolicIdentity(memory),
            "memory": memory,
            "state": {}
        }
        self.bus.subscribe(name)
        logging.info(f" [AgentPool] Added agent: {name} with persona: {persona.lower()}")

    def get_all(self: 'DynamicAgentPool') -> List[Dict[str, Any]]:
        """Returns all agents in the pool."""
        return list(self.subagents.values())

    def get_agents_by_persona(self: 'DynamicAgentPool', persona: str) -> List[str]:
        """Gets a list of agent names matching a specific persona."""
        return [agent["name"] for agent in self.subagents.values() if agent.get("persona") == persona]

    def get_persona_capabilities_prompt(self: 'DynamicAgentPool') -> str:
        """Generates a prompt describing available agent personas and their actions."""
        persona_actions: Dict[str, str] = {
            'research': "action must be 'research_topic'.",
            'coder': "action must be 'write_code'.",
            'qa': "action must be 'review_code'.",
            'orchestrator': "action can be 'create_long_term_goal_with_sub_tasks' or 'respond_to_user'."
        }
        available_personas: List[str] = sorted({agent['persona'] for agent in self.get_all()})
        prompt_lines: List[str] = ["The available personas and their valid actions are:"]
        prompt_lines.extend(
            f"- '{persona}': {persona_actions[persona]}"
            for persona in available_personas
            if persona in persona_actions
        )
        return "\n".join(prompt_lines)