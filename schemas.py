# symbolic_agi/schemas.py

from __future__ import annotations
from datetime import datetime, UTC, timedelta
from secrets import token_hex
from typing import Any, Dict, List, Literal, Optional, Annotated

from pydantic import BaseModel, Field

# --- CORE CONFIGURATION ---

class AGIConfig(BaseModel):
    name: str = "SymbolicAGI"
    scalable_agent_pool_size: int = 3
    meta_task_sleep_seconds: int = 10
    meta_task_timeout: int = 60
    motivational_drift_rate: float = 0.05
    memory_compression_window: timedelta = timedelta(days=1)
    social_interaction_threshold: timedelta = timedelta(hours=6)
    memory_forgetting_threshold: float = 0.2

# --- INTER-AGENT COMMUNICATION ---

class MessageModel(BaseModel):
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]

# --- AGI DATA MODELS ---

class EmotionalState(BaseModel):
    joy: float = 0.5
    sadness: float = 0.1
    anger: float = 0.1
    fear: float = 0.1
    surprise: float = 0.2
    disgust: float = 0.1
    trust: float = 0.5
    frustration: float = 0.2

    def clamp(self):
        for field in self.__class__.model_fields:
            value = getattr(self, field)
            setattr(self, field, max(0.0, min(1.0, value)))

class ActionStep(BaseModel):
    """Defines a single step in a plan, designed for delegation."""
    action: str
    parameters: Dict[str, Any]
    assigned_persona: Literal['research', 'coder', 'qa', 'orchestrator']
    risk: Optional[Literal['low', 'medium', 'high']] = 'low'

GoalStatus = Literal['active', 'paused', 'completed', 'failed']
GoalMode = Literal['code', 'docs']

class GoalModel(BaseModel):
    id: str = Field(default_factory=lambda: f"goal_{token_hex(8)}")
    description: str
    sub_tasks: List[ActionStep]
    status: GoalStatus = 'active'
    mode: GoalMode = 'code'
    last_failure: Optional[str] = None
    original_plan: Optional[List[ActionStep]] = None
    failure_count: int = 0
    max_failures: int = 3

# --- NEW: Planner Output Schema ---
class PlannerOutput(BaseModel):
    """Structured output from the planner, including its reasoning."""
    thought: str
    plan: List[ActionStep]

class SkillModel(BaseModel):
    """A structured representation of a learned skill with metadata."""
    id: str = Field(default_factory=lambda: f"skill_{token_hex(8)}")
    name: str
    description: str
    action_sequence: List[ActionStep]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    usage_count: int = 0
    effectiveness_score: float = 0.7 

class LifeEvent(BaseModel):
    """A structured representation of a significant event in the AGI's life story."""
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    summary: str
    importance: float = 0.5

PerceptionSource = Literal['workspace', 'microworld']
PerceptionType = Literal['file_created', 'file_modified', 'file_deleted', 'agent_appeared']

class PerceptionEvent(BaseModel):
    """Represents a single, passive observation of the environment."""
    source: PerceptionSource
    type: PerceptionType
    content: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

MemoryType = Literal[
    'user_input', 'action_result', 'reflection', 'goal', 'insight', 
    'self_modification', 'tool_usage', 'inner_monologue', 'debate', 
    'self_experiment', 'emotion', 'persona_fork', 'motivation_drift', 
    'skill_transfer', 'creativity', 'meta_insight', 'critical_error', 
    'meta_learning', 'self_explanation', 'cross_agent_transfer',
    'perception'
]

class MemoryEntryModel(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{token_hex(12)}")
    type: MemoryType
    content: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    importance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    embedding: Optional[List[float]] = None

class MetaEventModel(BaseModel):
    type: MemoryType
    data: Any
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())