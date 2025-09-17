# symbolic_agi/agi_controller.py
# cspell:ignore introspector Introspector autolearner cooldown cheatsheet saucedemo

from __future__ import annotations
import asyncio
import inspect
import logging
import random
import re
import traceback
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, cast, Callable

from symbolic_agi.schemas import (
    AGIConfig,
    GoalModel,
    ActionStep,
    MessageModel,
)
from symbolic_agi.symbolic_memory import SymbolicMemory
from symbolic_agi.long_term_memory import LongTermMemory
from symbolic_agi.skill_manager import SkillManager
from symbolic_agi.agent_pool import DynamicAgentPool
from symbolic_agi.message_bus import MessageBus
from symbolic_agi.tool_plugin import ToolPlugin
from symbolic_agi.planner import Planner
from symbolic_agi.api_client import client
from symbolic_agi.curriculum import load_curriculum

# Optional: present in some repos
try:
    from symbolic_agi.recursive_introspector import RecursiveIntrospector
except ImportError:
    RecursiveIntrospector = None  # Planner may not need it


# =========================
# Types & State containers
# =========================

class GoalStatus(Enum):
    NEW = auto()
    PLANNED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    NEEDS_REPLAN = auto()


class StepStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    OK = auto()
    ERROR = auto()
    SKIPPED = auto()


@dataclass
class StepResult:
    action: str
    assigned_persona: str
    status: StepStatus
    output: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class PlanResult:
    goal_id: str
    step_results: List[StepResult]
    overall_status: GoalStatus
    summary: str = ""


# =========================
# Agent Gateway
# =========================

class AgentGateway:
    def __init__(self, message_bus: MessageBus, receiver_id: str, response_timeout: float = 60.0) -> None:
        self.bus = message_bus
        self.receiver_id = receiver_id
        self.timeout = response_timeout
        self.log = logging.getLogger("AgentGateway")

    async def request(
        self,
        agent_name: str,
        message_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        correlation_id = str(uuid.uuid4())
        msg = MessageModel(
            sender_id=self.receiver_id,
            receiver_id=agent_name,
            message_type=message_type,
            payload={**(payload or {}), "correlation_id": correlation_id},
        )
        await self.bus.publish(msg)

        queue = self.bus.agent_queues.get(self.receiver_id)
        if queue is None:
            raise RuntimeError("Orchestrator is not subscribed to message bus.")

        expected_reply = f"{message_type}_result"
        while True:
            try:
                reply = await asyncio.wait_for(queue.get(), timeout=self.timeout)
            except asyncio.TimeoutError as e:
                raise TimeoutError(f"Agent '{agent_name}' timed out for '{message_type}'") from e

            if reply.message_type == expected_reply:
                return reply.payload


# =========================
# Plan Executor (skill-aware, no dupes, low complexity)
# =========================

class PlanExecutor:
    """
    Executes a plan step-by-step.
    Adds a small per-plan 'workspace' dict so tools can infer missing params
    (e.g., fetch_url can pick a URL from prior web_search results).
    """

    def __init__(self, agent_gateway: AgentGateway, tool_plugin: ToolPlugin, agent_pool: DynamicAgentPool, logger: logging.Logger, error_analyzer: ErrorAnalyzer) -> None:
        self.gateway = agent_gateway
        self.tools = tool_plugin
        self.pool = agent_pool
        self.log = logger
        self.error_analyzer = error_analyzer
        # Shared across steps for the current plan execution
        self.workspace: Dict[str, Any] = {}

    # ---------- Agent selection ----------

    def _pick_agent_for_persona(self, persona: str) -> Optional[str]:
        names = self.pool.get_agents_by_persona(persona.lower())
        return names[0] if names else None

    # ---------- Workspace helpers ----------

    def _update_workspace_from_tool(self, action: str, payload: Dict[str, Any]) -> None:
        """Persist the useful parts of a tool's response for downstream steps."""
        # payload is already guaranteed to be a dict by type annotation
        
        # Common keys to keep available
        for k in ("results", "url", "text", "content", "summary", "path", "data"):
            if k in payload and payload[k] is not None:
                self.workspace[k] = payload[k]

        # Convenience aliases for LLMs that refer to generic names
        if "text" in payload:
            self.workspace["data_text"] = payload["text"]
        if "results" in payload and not self.workspace.get("urls"):
            # extract urls list if present
            if urls := [
                str(cast(Dict[str, Any], item)["url"]) 
                for item in cast(List[Any], payload["results"] or [])
                if isinstance(item, dict) and cast(Dict[str, Any], item).get("url")
            ]:
                self.workspace["urls"] = urls

        # Action-specific crumbs
        if action in {"read_url", "browse_webpage"} and payload.get("text"):
            # keep a friendly alias for summarizers
            self.workspace["page_text"] = payload["text"]

    async def _call_tool_method(self, method: Callable[..., Any], params: Dict[str, Any], workspace: Dict[str, Any]) -> Any:
        """Call a tool method, handling workspace argument and sync/async."""
        if asyncio.iscoroutinefunction(method):
            try:
                return await method(**(params or {}), workspace=workspace)
            except TypeError:
                return await method(**(params or {}))
        else:
            try:
                return await asyncio.to_thread(method, **(params or {}), workspace=workspace)
            except TypeError:
                return await asyncio.to_thread(method, **(params or {}))
    # ---------- Tool / agent execution primitives ----------

    async def _try_run_tool(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a ToolPlugin method, always passing the shared workspace."""
        if not hasattr(self.tools, action):
            return {"status": "failure", "description": f"Tool '{action}' not found"}

        method = getattr(self.tools, action)
        try:
            out = await self._call_tool_method(method, params, self.workspace)
            if isinstance(out, dict) and cast(Dict[str, Any], out).get("status") == "success":
                return out  # type: ignore  # Dynamic tool response
            reason = ""
            if isinstance(out, dict):
                dict_out = cast(Dict[str, Any], out)
                reason = dict_out.get("description") or dict_out.get("error") or ""
            return {"status": "failure", "description": reason or "Tool returned non-success."}
        except Exception as e:
            return {"status": "failure", "description": str(e)}

    async def _run_orchestrator_tool_step(self, action: str, params: Dict[str, Any]) -> StepResult:
        self.log.info("[PlanExecutor] tool-step action=%s persona=orchestrator", action)
        out = await self._try_run_tool(action, params)
        ok = out.get("status") == "success"
        if ok:
            self._update_workspace_from_tool(action, out)
        err = "" if ok else (out.get("description") or out.get("error") or "Tool failed")
        self.log.info("[PlanExecutor] step_result action=%s persona=orchestrator status=%s error=%s", action, "OK" if ok else "ERROR", err)
        return StepResult(action, "orchestrator", StepStatus.OK if ok else StepStatus.ERROR, output=out, error=None if ok else err)

    async def _run_agent_step(self, persona: str, action: str, params: Dict[str, Any]) -> StepResult:
        agent_name = self._pick_agent_for_persona(persona)
        if not agent_name:
            return StepResult(action, persona, StepStatus.ERROR, error=f"No agent available for persona '{persona}'")

        self.log.info("[PlanExecutor] agent-step action=%s persona=%s agent=%s", action, persona, agent_name)
        try:
            result_payload: Any = await self.gateway.request(agent_name=agent_name, message_type=action, payload=params or {})
        except (TimeoutError, asyncio.TimeoutError):
            # Fallback to orchestrator execution if agent times out
            self.log.warning(f"Agent {agent_name} timed out, falling back to orchestrator for action: {action}")
            return await self._run_orchestrator_tool_step(action, params)
        except Exception as e:
            return StepResult(action, persona, StepStatus.ERROR, error=str(e))

        # Check if the response is a valid success dict
        is_valid_dict = isinstance(result_payload, dict)
        if is_valid_dict:
            dict_payload = cast(Dict[str, Any], result_payload)
            ok = dict_payload.get("status") == "success"
        else:
            ok = False
        
        if ok:
            # Agents can also return content we might want to stash - we know it's a dict here
            self._update_workspace_from_tool(action, cast(Dict[str, Any], result_payload))
        
        # Extract error message
        if not ok and is_valid_dict:
            err = cast(Dict[str, Any], result_payload).get("error")
        elif not ok:
            err = "Agent call failed"
        else:
            err = None
        
        self.log.info("[PlanExecutor] step_result action=%s persona=%s status=%s error=%s", action, persona, "OK" if ok else "ERROR", err or "")
        return StepResult(action, persona, StepStatus.OK if ok else StepStatus.ERROR, output=result_payload, error=err)

    async def _run_step(self, step: ActionStep) -> StepResult:
        action = step.action
        persona = (step.assigned_persona or "orchestrator").lower()
        params = step.parameters or {}

        # Force certain actions to run as orchestrator if tool is available
        if hasattr(self.tools, action):
            return await self._run_orchestrator_tool_step(action, params)
        
        if persona == "orchestrator":
            return await self._run_orchestrator_tool_step(action, params)
        return await self._run_agent_step(persona, action, params)

    async def _safe_run_step(self, step: ActionStep) -> StepResult:
        try:
            return await self._run_step(step)
        except Exception as e:
            self.log.exception("Step '%s' crashed:", step.action)
            return StepResult(step.action, (step.assigned_persona or "orchestrator").lower(), StepStatus.ERROR, error=str(e))

    # ---------- Main ----------

    async def execute(self, goal: GoalModel, plan_steps: List[ActionStep]) -> PlanResult:
        self.workspace = {}  # fresh per plan
        results: List[StepResult] = []
        failed_steps: List[StepResult] = []

        for step in plan_steps:
            rs = await self._safe_run_step(step)
            results.append(rs)
            
            if rs.status == StepStatus.ERROR:
                failed_steps.append(rs)
                self.log.error(f"Step '{rs.action}' failed: {rs.error}")
                
                # Analyze the error for recovery suggestions
                error_classification = self.error_analyzer.classify_error(rs.error or "Unknown error", step)
                self.log.info(f"Error classified as: {error_classification['type']} ({error_classification['severity']})")
                self.log.info(f"Recovery suggestion: {error_classification['suggestion']}")
                
                # For recoverable errors, we could implement retry logic here
                if error_classification['strategy'] == 'retry_with_backoff' and getattr(goal, 'retry_count', 0) < 2:
                    self.log.info(f"Attempting retry for recoverable error in step '{rs.action}'")
                    await asyncio.sleep(1)  # Simple backoff
                    retry_rs = await self._safe_run_step(step)
                    results[-1] = retry_rs  # Replace the failed result
                    if retry_rs.status != StepStatus.ERROR:
                        failed_steps.pop()  # Remove from failed list if retry succeeded
                        continue
                
                # Stop on first unrecoverable error; planner will handle replan
                return PlanResult(
                    goal_id=goal.id, 
                    step_results=results, 
                    overall_status=GoalStatus.NEEDS_REPLAN, 
                    summary=f"stopped_on_error: {error_classification['type']}"
                )

        # If we got here, all steps succeeded
        return PlanResult(goal_id=goal.id, step_results=results, overall_status=GoalStatus.COMPLETED, summary="ok")




# =========================
# Auto Learner
# =========================

class ErrorAnalyzer:
    """Analyzes failure patterns and suggests recovery strategies"""
    
    def __init__(self, logger: logging.Logger):
        self.log = logger
        
    def classify_error(self, error_str: str, step: ActionStep) -> Dict[str, str]:
        """Classify the type of error and suggest recovery strategy"""
        error_lower = error_str.lower()
        
        # Network/connectivity errors
        if any(keyword in error_lower for keyword in ['timeout', 'connection', 'network', 'http']):
            return {
                'type': 'network',
                'severity': 'recoverable',
                'strategy': 'retry_with_backoff',
                'suggestion': 'Retry the operation with exponential backoff'
            }
        
        # Parameter/input errors  
        if any(keyword in error_lower for keyword in ['missing', 'invalid', 'parameter', 'argument']):
            return {
                'type': 'parameter',
                'severity': 'critical',
                'strategy': 'fix_parameters',
                'suggestion': f'Fix parameters for action {step.action}'
            }
        
        # Permission/access errors
        if any(keyword in error_lower for keyword in ['permission', 'access', 'forbidden', '403', '401']):
            return {
                'type': 'access',
                'severity': 'critical', 
                'strategy': 'change_approach',
                'suggestion': 'Try alternative approach or different tool'
            }
        
        # Tool/function errors
        if any(keyword in error_lower for keyword in ['not found', 'unknown', 'unsupported']):
            return {
                'type': 'tool',
                'severity': 'critical',
                'strategy': 'substitute_tool',
                'suggestion': f'Find alternative to {step.action}'
            }
        
        # Default classification
        return {
            'type': 'unknown',
            'severity': 'moderate',
            'strategy': 'replan',
            'suggestion': 'Generate new plan with different approach'
        }
    
    def analyze_failure_pattern(self, goal: GoalModel, failed_steps: List[StepResult]) -> Dict[str, Any]:
        """Analyze patterns in failed steps to identify root causes"""
        
        if not failed_steps:
            return {'pattern': 'no_failures', 'recommendations': []}
        
        error_types: List[str] = []
        failed_actions: List[str] = []
        
        for step_result in failed_steps:
            if step_result.status == StepStatus.ERROR and step_result.error:
                # Create a minimal ActionStep for classification - cast persona safely
                persona = step_result.assigned_persona if step_result.assigned_persona in ['research', 'coder', 'qa', 'orchestrator'] else 'orchestrator'
                temp_step = ActionStep(action=step_result.action, assigned_persona=persona, parameters={})  # type: ignore
                classification = self.classify_error(step_result.error, temp_step)
                error_types.append(classification['type'])
                failed_actions.append(step_result.action)
        
        # Identify patterns
        recommendations: List[str] = []
        
        if len(set(failed_actions)) == 1:
            # Same action failing repeatedly
            action = failed_actions[0]
            recommendations.append(f'Action {action} consistently failing - consider alternative approach')
            
        if error_types.count('network') > len(error_types) // 2:
            # Mostly network errors
            recommendations.append('High network error rate - reduce web-dependent operations')
            
        if error_types.count('parameter') > 1:
            # Multiple parameter errors
            recommendations.append('Parameter issues detected - review goal decomposition')
        
        return {
            'pattern': f'{len(failed_steps)}_failures',
            'dominant_error_type': max(set(error_types), key=error_types.count) if error_types else 'none',
            'failed_actions': list(set(failed_actions)),
            'recommendations': recommendations
        }
    
    def suggest_recovery_strategies(self, goal: GoalModel, failure_analysis: Dict[str, Any]) -> List[str]:
        """Suggest specific recovery strategies based on failure analysis"""
        strategies: List[str] = []
        
        dominant_error = failure_analysis.get('dominant_error_type', 'unknown')
        failed_actions = failure_analysis.get('failed_actions', [])
        
        if dominant_error == 'network':
            strategies.extend([
                "Reduce dependency on web searches and online resources",
                "Use local knowledge and skills instead of fetching external content"
            ])
            
        elif dominant_error == 'parameter':
            strategies.extend([
                "Break down goal into smaller, more specific sub-tasks",
                "Ensure all required parameters are clearly defined"
            ])
            
        elif dominant_error == 'tool':
            strategies.append("Find alternative tools for the same purpose")
            if 'web_search' in failed_actions:
                strategies.append("Use direct knowledge instead of web search")
            if 'fetch_url' in failed_actions:
                strategies.append("Focus on analysis and reasoning rather than data fetching")
        
        # Goal-specific strategies
        goal_desc = goal.description.lower()
        if 'research' in goal_desc and 'network' in dominant_error:
            strategies.append("Use existing knowledge for research instead of web searches")
        
        if 'create' in goal_desc or 'write' in goal_desc:
            strategies.append("Focus on content generation using available skills")
            
        return strategies


class AutoLearner:
    def __init__(self, skill_manager: SkillManager, memory: SymbolicMemory, logger: logging.Logger) -> None:
        self.skills = skill_manager
        self.memory = memory
        self.log = logger

    @staticmethod
    def _filter(steps: List[ActionStep]) -> List[ActionStep]:
        return [s for s in steps if s.action != "review_plan"]

    @staticmethod
    def _safe_name(text: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_") or "Skill"
        return base[:48]

    def learn_from(self, goal: GoalModel, original_steps: List[ActionStep], step_results: List[StepResult]) -> Optional[str]:
        if not step_results or step_results[-1].status != StepStatus.OK:
            return None
        steps = self._filter(original_steps)
        if not steps:
            return None

        base = self._safe_name(goal.description)
        name = base
        if self.skills.get_skill_by_name(name):
            name = f"{base}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

        self.skills.add_new_skill(name=name, description=goal.description, plan=steps)

        with suppress(ImportError):
            from .schemas import MemoryEntryModel
            entry = MemoryEntryModel(
                type='skill_transfer',
                content={
                    "skill_name": name,
                    "goal_description": goal.description,
                    "message": f"Auto-learned skill '{name}' from goal '{goal.description}'.",
                    "tags": ["skill", "learning", "autonomous"]
                }
            )
            # Store task temporarily to prevent garbage collection
            _ = asyncio.create_task(self.memory.add_memory(entry))

        self.log.info(f"Auto-learned new skill: {name}")
        return name


# =========================
# Event Sink (optional helper)
# =========================

class EventSink:
    def __init__(self, logger: logging.Logger, memory: SymbolicMemory) -> None:
        self.log = logger
        self.memory = memory

    def emit(self, level: str, msg: str, **kv: Any) -> None:
        line = f"{msg} | " + " ".join(f"{k}={v}" for k, v in kv.items())
        getattr(self.log, level.lower(), self.log.info)(line)
        with suppress(ImportError):
            from .schemas import MemoryEntryModel
            entry = MemoryEntryModel(
                type='inner_monologue',
                content={"message": line, "tags": ["event"]}
            )
            # Create and forget - let the task run in background
            _ = asyncio.create_task(self.memory.add_memory(entry))


# =========================
# Helpers to reduce complexity in __init__
# =========================

def _instantiate_with_signature(cls: Any, candidates: Dict[str, Any], logger: logging.Logger) -> Optional[Any]:
    """Create an instance of cls by filtering candidates to its __init__ signature."""
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        kwargs = {k: v for k, v in candidates.items() if k in params}
        return cls(**kwargs)
    except Exception as e:
        logger.warning(f"{getattr(cls, '__name__', str(cls))} init failed: {e}")
        return None


# =========================
# SymbolicAGI Orchestrator
# =========================

class SymbolicAGI:
    def __init__(self, cfg: Optional[AGIConfig] = None) -> None:
        self.cfg = cfg or AGIConfig()
        self.name = getattr(self.cfg, "agent_name", "SymbolicAGI")

        # Core subsystems
        self.message_bus = MessageBus()
        self.agent_pool = DynamicAgentPool(self.message_bus)
        self.memory = SymbolicMemory(client=client)
        self.ltm = LongTermMemory()
        self.skill_manager = SkillManager()
        self.tool_plugin = ToolPlugin()

        # Introspector (adaptive)
        self.introspector = None
        if RecursiveIntrospector is not None:
            self.introspector = _instantiate_with_signature(
                RecursiveIntrospector,
                {
                    "llm_client": client,
                    "api_client": client,
                    "client": client,
                    "identity": self.name,
                    "logger": logging.getLogger("RecursiveIntrospector"),
                    "skill_manager": self.skill_manager,
                    "agent_pool": self.agent_pool,
                    "tool_plugin": self.tool_plugin,
                },
                logging.getLogger(self.name),
            )

        # Planner (adaptive)
        planner_instance = _instantiate_with_signature(
            Planner,
            {
                "introspector": self.introspector,
                "skill_manager": self.skill_manager,
                "agent_pool": self.agent_pool,
                "tool_plugin": self.tool_plugin,
            },
            logging.getLogger(self.name),
        )
        self.planner = planner_instance or Planner(
            introspector=cast(Any, self.introspector),
            skill_manager=self.skill_manager,
            agent_pool=self.agent_pool,
            tool_plugin=self.tool_plugin
        )  # last resort

        # Logging & events
        self.logger = logging.getLogger(self.name)
        self.events = EventSink(self.logger, self.memory)

        # Inboxes
        self.inbox_queue = self.message_bus.subscribe(self.name)

        # Gateways & executors
        self.gateway = AgentGateway(self.message_bus, receiver_id=self.name, response_timeout=60.0)
        self.autolearner = AutoLearner(self.skill_manager, self.memory, self.logger)
        self.error_analyzer = ErrorAnalyzer(self.logger)
        self.executor = PlanExecutor(self.gateway, self.tool_plugin, self.agent_pool, self.logger, self.error_analyzer)

        # Internal bookkeeping
        self._active_plans: Dict[str, List[ActionStep]] = {}   # goal_id -> steps
        self._replan_attempts: Dict[str, int] = {}             # goal_id -> attempts
        self._max_replans: int = 2
        self._memory_tasks: List[asyncio.Task[Any]] = []       # Background memory tasks
        self._last_seed_time: float = 0.0
        self._seed_cooldown: float = 45.0  # seconds

        # Background tasks
        self._bg_tasks: List[asyncio.Task[Any]] = []

    def _create_default_agents_async(self) -> None:
        """Create default agents for each persona to handle delegated tasks."""
        personas = ["research", "coder", "qa"]
        
        for persona in personas:
            agent_name = f"{persona}_agent"
            # Create both the pool entry AND spawn the actual Agent instance
            self.agent_pool.add_agent(
                name=agent_name,
                persona=persona,
                memory=self.memory,
                skills=self.skill_manager
            )
            
            # Import and create actual Agent instance
            from .agent import Agent
            from .api_client import client
            
            agent_instance = Agent(agent_name, self.message_bus, client)
            # Start the agent in the background
            agent_task = asyncio.create_task(agent_instance.run())
            self._bg_tasks.append(agent_task)
            
            self.logger.info(f"Created and started agent: {agent_name} with persona: {persona}")

    # ---------- Lifecycle ----------

    async def startup_validation(self) -> None:
        """Run any async validation/initialization that requires awaiting."""
        with suppress(Exception):
            if hasattr(self.tool_plugin, "ensure_workspace_ready"):
                maybe = getattr(self.tool_plugin, "ensure_workspace_ready")()
                if asyncio.iscoroutine(maybe):
                    await maybe
                elif callable(getattr(self.tool_plugin, "ensure_workspace_ready", None)):
                    maybe()

        # Initialize default agents for different personas (moved here for async support)
        self._create_default_agents_async()

        await asyncio.sleep(0)
        self.events.emit("info", "startup_validation_ok", agent=self.name)

    def start_background_tasks(self) -> None:
        """Start long-running tasks (sync on purpose)."""
        self._bg_tasks.append(asyncio.create_task(self._memory_autosave_loop()))
        self.events.emit("info", "background_tasks_started", agent=self.name)

    async def shutdown(self) -> None:
        for t in self._bg_tasks:
            t.cancel()
        await asyncio.gather(*self._bg_tasks, return_exceptions=True)
        self.events.emit("info", "shutdown_complete", agent=self.name)

    async def _memory_autosave_loop(self) -> None:
        try:
            while True:
                try:
                    if hasattr(self.memory, "save_if_dirty"):
                        getattr(self.memory, "save_if_dirty")()
                except Exception as e:
                    self.logger.warning(f"Memory autosave failed: {e}")
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.logger.info("Autosave task cancelled.")
            raise

    # ---------- Autonomy Helpers ----------

    def _skills_block_for_planner(self) -> str:
        """Expose learned skills to the planner in a simple text block."""
        with suppress(Exception):
            if hasattr(self.skill_manager, "get_formatted_definitions"):
                return self.skill_manager.get_formatted_definitions()
        return ""

    def _file_manifest_for_planner(self) -> str:
        """Append available skills to the manifest so LLM can see macro actions."""
        file_section = ""
        with suppress(Exception):
            if hasattr(self.tool_plugin, "list_workspace_files"):
                files = getattr(self.tool_plugin, "list_workspace_files")()
                if isinstance(files, list):
                    file_list = cast(List[str], files)  # Assume string list for workspace files
                    file_section = "\n".join(file_list)
                else:
                    file_section = ""

        skills = self._skills_block_for_planner()
        if skills:
            return f"{file_section}\n\n# AVAILABLE SKILLS & ACTIONS\n{skills}"
        return file_section

    def _store_plan(self, goal_id: str, steps: List[ActionStep]) -> None:
        self._active_plans[goal_id] = steps

    def _get_plan(self, goal_id: str) -> Optional[List[ActionStep]]:
        return self._active_plans.get(goal_id)

    def _bump_replan(self, goal_id: str) -> int:
        n = self._replan_attempts.get(goal_id, 0) + 1
        self._replan_attempts[goal_id] = n
        return n

    def _clear_goal_state(self, goal_id: str) -> None:
        self._active_plans.pop(goal_id, None)
        self._replan_attempts.pop(goal_id, None)

    def _create_and_add_goal(self, description: str) -> GoalModel:
        """Create a goal with standard defaults and add it to long-term memory."""
        goal = GoalModel(description=description, sub_tasks=[], mode="docs")
        self.ltm.add_goal(goal)
        self._last_seed_time = asyncio.get_running_loop().time()
        self.events.emit("info", "idle_goal_seeded", goal_id=goal.id, desc=goal.description)
        return goal

    def _seed_idle_goal(self) -> Optional[GoalModel]:
        # Cooldown
        now = asyncio.get_running_loop().time()
        if now - self._last_seed_time < self._seed_cooldown:
            return None
        if self.ltm.get_active_goal():
            return None

        # 30% chance to generate an adaptive goal based on learned skills
        if random.random() < 0.3:
            if adaptive_goal := self._generate_adaptive_goal():
                self.logger.info(f"Generated adaptive goal: {adaptive_goal}")
                return self._create_and_add_goal(adaptive_goal)

        # Fallback to curriculum templates
        templates = load_curriculum()
        desc = random.choice(templates)
        return self._create_and_add_goal(desc)

    def _generate_adaptive_goal(self) -> Optional[str]:
        """Generate goals based on learned skills and recent failures."""
        all_skills = self.skill_manager.list_skills()
        skills = [skill.name for skill in all_skills]
        
        if not skills:
            return None
            
        # Try different adaptive strategies
        strategies = [
            self._generate_skill_combination_goal,
            self._generate_skill_improvement_goal,
            self._generate_exploration_goal,
        ]
        
        for strategy in strategies:
            try:
                if goal := strategy(skills):
                    return goal
            except Exception as e:
                self.logger.debug(f"Strategy {strategy.__name__} failed: {e}")
        
        return None

    def _generate_skill_combination_goal(self, skills: List[str]) -> Optional[str]:
        """Create goals that combine multiple learned skills."""
        if len(skills) < 2:
            return None
        
        # Pick 2-3 random skills to combine
        num_skills = min(3, random.randint(2, len(skills)))
        selected_skills = random.sample(skills, num_skills)
        
        combination_templates = [
            f"Use skills {' and '.join(selected_skills[:-1])} and {selected_skills[-1]} in sequence to accomplish a complex task.",
            f"Create a workflow that applies {selected_skills[0]} then {selected_skills[1]} to process data.",
            f"Combine the capabilities of {' and '.join(selected_skills)} to solve a multi-step problem.",
        ]
        
        return random.choice(combination_templates)

    def _generate_skill_improvement_goal(self, skills: List[str]) -> Optional[str]:
        """Create goals that improve or test existing skills."""
        if not skills:
            return None
            
        skill = random.choice(skills)
        improvement_templates = [
            f"Test skill '{skill}' with different parameters and document the results.",
            f"Create a more robust version of skill '{skill}' with better error handling.",
            f"Analyze the effectiveness of skill '{skill}' and suggest improvements.",
            f"Create documentation and examples for how to use skill '{skill}'.",
        ]
        
        return random.choice(improvement_templates)

    def _generate_exploration_goal(self, skills: List[str]) -> Optional[str]:
        """Create goals that explore new capabilities based on existing skills."""
        exploration_templates = [
            "Explore a new domain or API that could complement existing skills.",
            "Research and implement a new tool or capability for the skill library.", 
            "Analyze recent failures to identify gaps in capabilities and address them.",
            "Create a comprehensive test suite for all learned skills.",
            "Document the current state of the AGI system and its capabilities.",
        ]
        
        return random.choice(exploration_templates)

    async def _ensure_plan(self, goal: GoalModel) -> Tuple[List[ActionStep], str]:
        if cached := self._get_plan(goal.id):
            return cached, "cached"

        file_manifest = self._file_manifest_for_planner()
        planner_output = await self.planner.decompose_goal_into_plan(
            goal.description, file_manifest=file_manifest, mode=getattr(goal, "mode", "docs")
        )
        steps = planner_output.plan if hasattr(planner_output, "plan") else []
        self._store_plan(goal.id, steps)
        return steps, "new"

    async def _try_replan(self, goal: GoalModel, failed_steps: Optional[List[StepResult]] = None) -> bool:
        attempts = self._bump_replan(goal.id)
        if attempts > self._max_replans:
            return False

        # Analyze failure patterns if we have failed steps
        failure_analysis = None
        if failed_steps:
            failure_analysis = self.error_analyzer.analyze_failure_pattern(goal, failed_steps)
            self.logger.info(f"Failure pattern analysis: {failure_analysis}")
            
            # Get specific recovery strategies
            recovery_strategies = self.error_analyzer.suggest_recovery_strategies(goal, failure_analysis)
            self.logger.info(f"Recovery strategies: {recovery_strategies}")
            
            # Store failure analysis in goal for future reference
            if hasattr(goal, 'last_failure'):
                goal.last_failure = f"Pattern: {failure_analysis['pattern']}, Dominant: {failure_analysis['dominant_error_type']}"

        file_manifest = self._file_manifest_for_planner()
        
        # Enhanced prompt with failure analysis
        replan_context = ""
        if failure_analysis and failure_analysis['recommendations']:
            strategies = self.error_analyzer.suggest_recovery_strategies(goal, failure_analysis)
            replan_context = f"\n\nPrevious attempt failed with pattern: {failure_analysis['pattern']}. " + \
                           f"Error type: {failure_analysis['dominant_error_type']}. " + \
                           f"Recommendations: {'; '.join(failure_analysis['recommendations'])}. " + \
                           f"Recovery strategies: {'; '.join(strategies)}"
        
        try:
            # If we have failure analysis, include it in the planning context
            enhanced_description = goal.description + replan_context
            
            planner_output = await self.planner.decompose_goal_into_plan(
                enhanced_description, file_manifest=file_manifest, mode=getattr(goal, "mode", "docs")
            )
            steps = planner_output.plan if hasattr(planner_output, "plan") else []
            if not steps:
                return False
            self._store_plan(goal.id, steps)
            self.events.emit("info", "replanned_goal", goal_id=goal.id, attempts=attempts, 
                           failure_pattern=failure_analysis['pattern'] if failure_analysis else None)
            return True
        except Exception as e:
            self.logger.warning(f"Replan failed: {e}")
            return False

    # ---------- Main Cognitive Cycle ----------

    async def handle_autonomous_cycle(self) -> Dict[str, Any]:
        try:
            return await self._handle_autonomous_cycle_inner()
        except asyncio.CancelledError:
            self.logger.info("Autonomous cycle cancelled.")
            raise
        except Exception as e:
            self.logger.error(f"Critical error in autonomous cycle: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)
            return {"description": f"Error: {e}"}

    async def _handle_autonomous_cycle_inner(self) -> Dict[str, Any]:
        active_goal = self.ltm.get_active_goal()
        if not active_goal:
            self._seed_idle_goal()
            return {"description": "Idle; seeded goal if cooldown elapsed."}

        steps, plan_source = await self._ensure_plan(active_goal)
        if not steps:
            self.ltm.update_goal_status(active_goal.id, status="failed")
            self._clear_goal_state(active_goal.id)
            return {"description": "Planner returned empty plan; goal failed."}

        self.events.emit("info", "plan_ready", goal_id=active_goal.id, source=plan_source, step_count=len(steps))

        plan_result = await self.executor.execute(active_goal, steps)

        if plan_result.overall_status == GoalStatus.COMPLETED:
            learned_name = self.autolearner.learn_from(active_goal, steps, plan_result.step_results)
            self.ltm.update_goal_status(active_goal.id, status="completed")
            self._clear_goal_state(active_goal.id)
            resp = "Goal completed."
            if learned_name:
                resp += f" Learned skill: {learned_name}."
            return {"description": resp, "response_text": resp}

        # Collect failed steps for analysis
        failed_steps = [step for step in plan_result.step_results if step.status == StepStatus.ERROR]
        
        if await self._try_replan(active_goal, failed_steps):
            return {"description": "Plan failed; replanned."}

        self.ltm.update_goal_status(active_goal.id, status="failed")
        self._clear_goal_state(active_goal.id)
        return {"description": "Goal failed and archived after replans."}
