# symbolic_agi/curriculum.py
from __future__ import annotations

import os
from typing import Any, List

try:
    import yaml
except Exception:
    yaml = None  # graceful fallback

DEFAULTS = [
    "Do a web search for 'AI news today', open one article, summarize to workspace.",
    "Open 'symbolic_agi/tool_plugin.py', propose a refactor note.",
    "Search 'best Python testing tips', write a 10-bullet cheatsheet.",
]

def load_curriculum(path: str = "docs/curriculum.yaml") -> List[str]:
    """Load a list of idle goals. Falls back to defaults if YAML or file missing."""
    if yaml is None or not os.path.exists(path):
        return DEFAULTS
    try:
        return _load_from_yaml(path)
    except Exception:
        return DEFAULTS

def _load_from_yaml(path: str) -> List[str]:
    """Load curriculum from YAML file with proper type handling."""
    if yaml is None:
        return DEFAULTS
        
    with open(path, "r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f) or {}  # type: ignore[attr-defined]
    
    # Ensure we have a dict and extract goals safely
    if not isinstance(data, dict):
        return DEFAULTS
        
    goals: Any = data.get("idle_goals", [])  # type: ignore[attr-defined]
    
    # Validate that goals is a list of strings
    if not isinstance(goals, list):
        return DEFAULTS
        
    # Filter to ensure all items are strings
    str_goals: List[str] = [goal for goal in goals if isinstance(goal, str)]  # type: ignore[misc]
    
    return str_goals or DEFAULTS
