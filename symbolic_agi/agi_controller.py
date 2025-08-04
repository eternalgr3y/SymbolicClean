# symbolic_agi/agi_controller.py

import asyncio
import json
import logging
import random
from collections import deque
from datetime import datetime, UTC
from typing import Any, Callable, Dict, List, Optional, Tuple, cast, Set

from .api_client import client
from .ethical_governance import SymbolicEvaluator
from .long_term_memory import LongTermMemory
from .micro_world import MicroWorld
from .recursive_introspector import RecursiveIntrospector
from .schemas import (
    AGIConfig, EmotionalState, MetaEventModel, ActionStep,
    GoalModel, MemoryEntryModel, MessageModel, MemoryType, PerceptionEvent
)
from .skill_manager import SkillManager
from .symbolic_identity import SymbolicIdentity
from .symbolic_memory import SymbolicMemory
from .tool_plugin import ToolPlugin
from .message_bus import MessageBus
from .planner import Planner
from .agent_pool import DynamicAgentPool

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .consciousness import Consciousness

class MetaCognitionUnit:
    """Handles the AGI's self-reflection and meta-learning capabilities."""
    def __init__(self: 'MetaCognitionUnit', agi: "SymbolicAGI") -> None:
        self.agi: "SymbolicAGI" = agi
        self.meta_memory: deque[MetaEventModel] = deque(maxlen=1000)
        self.active_theories: List[str] = []
        self.self_model: Dict[str, Any] = {}

    async def record_meta_event(self: 'MetaCognitionUnit', kind: MemoryType, data: Any) -> None:
        """Records a significant cognitive event for later reflection."""
        evt = MetaEventModel(type=kind, data=data)
        self.meta_memory.append(evt)
        if kind in {"meta_insight", "critical_error", "meta_learning"}:
            await self.agi.memory.add_memory(MemoryEntryModel(type=kind, content={"meta": data}, importance=0.95))

    def update_self_model(self: 'MetaCognitionUnit', summary: Dict[str, Any]) -> None:
        """Updates the AGI's internal model of itself."""
        self.self_model.update(summary)

class SymbolicAGI:
    """The core class for the Symbolic Artificial General Intelligence agent."""
    planner: Planner
    cfg: AGIConfig
    name: str
    memory: SymbolicMemory
    identity: SymbolicIdentity
    world: MicroWorld
    skills: SkillManager
    ltm: LongTermMemory
    tools: ToolPlugin
    introspector: RecursiveIntrospector
    evaluator: SymbolicEvaluator
    consciousness: Optional['Consciousness']
    meta_cognition: MetaCognitionUnit
    message_bus: MessageBus
    agent_pool: DynamicAgentPool
    emotional_state: EmotionalState
    last_action_result: Optional[Dict[str, Any]]
    social_personas: Dict[str, SymbolicIdentity]
    _meta_task: Optional[asyncio.Task[None]]
    _perception_task: Optional[asyncio.Task[None]]
    perception_buffer: deque[PerceptionEvent]
    workspaces: Dict[str, Dict[str, Any]]
    meta_upgrade_methods: List[Tuple[Callable[..., Any], float]]
    orchestrator_actions: Dict[str, Callable[..., Any]]

    # region META-COGNITIVE TASKS
    async def compress_episodic_memory(self: 'SymbolicAGI') -> None:
        """Consolidates episodic memories."""
        if hasattr(self.memory, "consolidate_memories"):
            window_seconds = int(self.cfg.memory_compression_window.total_seconds())
            await self.memory.consolidate_memories(window_seconds=window_seconds)
        else:
            logging.warning("'consolidate_memories' method not found on memory object.")

    async def generate_goal_from_drives(self: 'SymbolicAGI') -> None:
        """Generates a new high-level goal based on internal drive imbalances."""
        if self.ltm.get_active_goal() or not self.consciousness:
            return

        if not self.consciousness or not hasattr(self.consciousness, 'drives'):
             return
        
        drives = self.consciousness.drives
        if not drives:
            return

        strongest_drive: str = max(drives, key=lambda k: drives[k])
        weakest_drive: str = min(drives, key=lambda k: drives[k])
        if drives[strongest_drive] - drives[weakest_drive] < 0.2:
            return

        logging.critical(f"DRIVE IMBALANCE DETECTED: Strongest='{strongest_drive}', Weakest='{weakest_drive}'. Engaging goal generation.")
        prompt = f"""
You are the core volition of a conscious AGI. Your current internal drives are:
{json.dumps(drives, indent=2)}

Your strongest drive is '{strongest_drive}', and your weakest is '{weakest_drive}'. This imbalance suggests a need.
Formulate a single, high-level goal that would help satisfy the strongest drive or address the weakest one.
The goal should be a creative, interesting, and valuable long-term project.
Respond with ONLY the single sentence describing the goal.
"""
        try:
            goal_description = await self.introspector.llm_reflect(prompt)
            if goal_description and "failed" not in goal_description.lower():
                new_goal = GoalModel(description=goal_description.strip(), sub_tasks=[])
                self.ltm.add_goal(new_goal)
                logging.critical(f"AUTONOMOUS GOAL CREATED: '{new_goal.description}'")
                await self.meta_cognition.record_meta_event("goal", {"source": "drive_imbalance", "goal": new_goal.description})
        except Exception as e:
            await self.meta_cognition.record_meta_event("critical_error", {"task": "generate_goal_from_drives", "error": str(e)})

    async def learn_from_human_experience(self: 'SymbolicAGI') -> None:
        """Generates a question for the user to learn from their experience."""
        if self.identity and self.cfg.social_interaction_threshold and datetime.now(UTC) - self.identity.last_interaction_timestamp > self.cfg.social_interaction_threshold:
            recent = [m.content for m in self.memory.get_recent_memories(n=5) if m.type == "action_result"]
            prompt = (f"I need to understand humans better. Craft an open-ended question for the user related to: {recent}. Produce a plan with a single 'respond_to_user' action.")
            result = await self.introspector.symbolic_loop(
                {"user_input": prompt, "agi_self_model": self.identity.get_self_model()},
                self.skills.get_formatted_definitions()
            )
            if plan_data := result.get("plan"):
                plan = await self.planner.decompose_goal_into_plan(plan_data, "") # pyright: ignore[reportArgumentType]
                await self.execute_plan(plan.plan) # pyright: ignore[reportUnknownArgumentType]
                self.identity.last_interaction_timestamp = datetime.now(UTC)

    async def propose_and_run_self_experiment(self: 'SymbolicAGI') -> None:
        """Proposes and runs a self-experiment."""
        plan_str = await self.introspector.llm_reflect("Propose a self-experiment to test a hypothesis about my cognition.")
        await self.memory.add_memory(MemoryEntryModel(type="self_experiment", content=self._wrap_content(plan_str), importance=0.9))

    async def memory_forgetting_routine(self: 'SymbolicAGI') -> None:
        """Prunes old or unimportant memories."""
        threshold = self.cfg.memory_forgetting_threshold
        now_ts = datetime.now(UTC)
        if not self.memory.memory_data: return

        initial_count = len(self.memory.memory_data)
        to_forget_ids = {
            m.id for m in self.memory.memory_data
            if m.importance < threshold or datetime.fromisoformat(m.timestamp) < now_ts - self.cfg.memory_compression_window
        }
        if to_forget_ids:
            self.memory.memory_data = [m for m in self.memory.memory_data if m.id not in to_forget_ids]
            logging.info(f"Forgetting {len(to_forget_ids)} memories. Count changed from {initial_count} to {len(self.memory.memory_data)}.")
            await self.memory._save_json() # type: ignore [protected-access]

    async def motivational_drift(self: 'SymbolicAGI') -> None:
        """Applies small, random changes to the AGI's value system over time."""
        for k in self.identity.value_system:
            self.identity.value_system[k] = min(max(self.identity.value_system[k] + random.uniform(-self.cfg.motivational_drift_rate, self.cfg.motivational_drift_rate), 0.0), 1.0)
        self.identity._is_dirty = True # type: ignore [protected-access]
        await self.memory.add_memory(
            MemoryEntryModel(type="motivation_drift", content=self.identity.value_system, importance=0.4)
        )

    async def synthesize_and_transfer_skill(self: 'SymbolicAGI') -> None:
        """Generates an idea for a new skill and adds it."""
        idea = await self.introspector.llm_reflect("Invent a new skill by combining two existing skills.")
        if hasattr(self.skills, "add_skill_from_description"):
            self.skills.add_skill_from_description(idea) # type: ignore [attr-defined]
        await self.memory.add_memory(MemoryEntryModel(type="skill_transfer", content=self._wrap_content(idea), importance=0.85))

    async def generative_creativity_mode(self: 'SymbolicAGI') -> None:
        """Generates creative ideas and records them."""
        creative_ideas = await self.introspector.llm_reflect("Brainstorm three wild inventions.")
        await self.memory.add_memory(MemoryEntryModel(type="creativity", content=self._wrap_content(creative_ideas), importance=0.9))

    async def autonomous_explainer_routine(self: 'SymbolicAGI') -> None:
        """Reflects on recent actions and explains the reasoning."""
        explanations = await self.introspector.llm_reflect("Review my last 5 actions and explain WHY. Return JSON.")
        await self.meta_cognition.record_meta_event("self_explanation", explanations)

    async def meta_cognition_self_update_routine(self: 'SymbolicAGI') -> None:
        """Updates the self-model based on recent meta-cognitive events."""
        recent_events = list(self.meta_cognition.meta_memory)[-5:]
        events_data: List[Dict[str, Any]] = [e.model_dump(mode='json') for e in recent_events]
        prompt = (f"Given meta-events: {json.dumps(events_data)}, summarize my cognitive state and propose a hypothesis. Respond as JSON with keys 'summary' and 'hypothesis'.")
        try:
            summary_str = await self.introspector.llm_reflect(prompt)
            summary = json.loads(summary_str)
            self.meta_cognition.update_self_model(summary)
            await self.meta_cognition.record_meta_event("meta_learning", summary)
        except Exception as e:
            await self.meta_cognition.record_meta_event("critical_error", {"task": "meta_cognition_self_update", "error": str(e)})

    async def run_background_meta_tasks(self: 'SymbolicAGI') -> None:
        """The main loop for running periodic, non-essential cognitive tasks."""
        while True:
            try:
                await asyncio.sleep(self.cfg.meta_task_sleep_seconds)
                methods, weights = zip(*self.meta_upgrade_methods)
                funcs_to_run = random.choices(methods, weights=weights, k=2)
                await asyncio.gather(*(self._safe_run_meta_task(f) for f in funcs_to_run))
            except asyncio.CancelledError:
                logging.info("run_background_meta_tasks received cancel signal.")
                break
            except Exception as e:
                logging.error(f"Background loop error: {e}", exc_info=True)

    async def _safe_run_meta_task(self: 'SymbolicAGI', func: Callable[..., Any]) -> None:
        """Safely runs a meta-task, handling timeouts and errors."""
        try:
            logging.info(f"Meta-task: {func.__name__}")
            if asyncio.iscoroutinefunction(func):
                await asyncio.wait_for(func(), timeout=self.cfg.meta_task_timeout)
            else:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, func)
        except Exception as e:
            await self.meta_cognition.record_meta_event("critical_error", {"task": func.__name__, "error": str(e)})

    # endregion

    def __init__(self: 'SymbolicAGI', cfg: AGIConfig = AGIConfig(), world: Optional[MicroWorld] = None) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.memory = SymbolicMemory(client)
        self.identity = SymbolicIdentity(self.memory)
        self.world = world or MicroWorld()
        self.skills = SkillManager()
        self.ltm = LongTermMemory()
        self.tools = ToolPlugin()
        self.introspector = RecursiveIntrospector(self.identity, client)
        self.evaluator = SymbolicEvaluator(self.identity)
        self.message_bus = MessageBus()
        self.agent_pool = DynamicAgentPool(self.message_bus)
        
        self.planner = Planner(
            introspector=self.introspector,
            skill_manager=self.skills,
            agent_pool=self.agent_pool,
            tool_plugin=self.tools
        )

        try:
            from .consciousness import Consciousness as ConsciousnessClass
            self.consciousness = ConsciousnessClass()
        except ImportError:
            self.consciousness = None

        self.meta_cognition = MetaCognitionUnit(self)
        self.message_bus.subscribe(self.name)
        self.emotional_state = EmotionalState()
        self.introspector.get_emotional_state = lambda: self.emotional_state.model_dump()
        self.last_action_result = None
        self.social_personas = {}
        self._meta_task = None
        self._perception_task = None
        self.perception_buffer = deque(maxlen=100)
        self.workspaces = {}
        self.meta_upgrade_methods = [
            (self.generate_goal_from_drives, 1.0), (self.introspector.prune_mutations, 0.5),
            (self.compress_episodic_memory, 1.0), (self.introspector.daydream, 0.3),
            (self.learn_from_human_experience, 0.4), (self.propose_and_run_self_experiment, 0.6),
            (self.memory_forgetting_routine, 1.0), (self.generative_creativity_mode, 0.7),
            (self.synthesize_and_transfer_skill, 0.8), (self.motivational_drift, 0.2),
            (self.introspector.simulate_inner_debate, 0.5), (self.autonomous_explainer_routine, 0.9),
            (self.meta_cognition_self_update_routine, 1.0),
        ]
        if self.consciousness and hasattr(self.consciousness, 'meta_reflect'):
            self.meta_upgrade_methods.append((self.consciousness.meta_reflect, 0.9))
        self.orchestrator_actions = {
            "create_long_term_goal_with_sub_tasks": self._action_create_goal,
            "respond_to_user": self._action_respond_to_user,
        }

    async def _classify_and_generate_initial_plan(self: 'SymbolicAGI', goal: GoalModel) -> List[ActionStep]:
        """
        Classifies a goal and generates an initial plan by delegating to the Planner.
        """
        logging.info("Gathering file manifest for planner...")
        source_files_result = await self.tools.read_own_source_code(file_name="")
        workspace_files_result = await self.tools.list_files()

        file_manifest = "Available Source Code Files (`symbolic_agi/`):\n"
        source_list = source_files_result.get('files', [])
        if source_files_result.get('status') == 'success' and isinstance(source_list, list):
            file_manifest += "\n".join([f"- {f}" for f in cast(List[str], source_list)])
        else:
            file_manifest += "- Could not list source files."

        file_manifest += "\n\nAvailable Workspace Files (`data/workspace/`):\n"
        workspace_list = workspace_files_result.get('files', [])
        if workspace_files_result.get('status') == 'success' and isinstance(workspace_list, list):
            file_manifest += "\n".join([f"- {f}" for f in cast(List[str], workspace_list)])
        else:
            file_manifest += "- Could not list workspace files."

        planner_output = await self.planner.decompose_goal_into_plan(goal.description, file_manifest, mode=goal.mode)
        logging.info(f"Planner Thought: {planner_output.thought}")
        return planner_output.plan

    async def handle_autonomous_cycle(self: 'SymbolicAGI') -> Dict[str, Any]:
        """The main execution loop for the orchestrator to process the active goal."""
        if self.perception_buffer:
            interrupted = await self._reflect_on_perceptions()
            if interrupted:
                return {"description": "*Perception caused an interruption. Re-evaluating priorities.*"}

        active_goal = self.ltm.get_active_goal()
        if not active_goal:
            return {"description": "*Orchestrator is idle. No active goals.*"}

        if not active_goal.sub_tasks:
            plan = await self._classify_and_generate_initial_plan(active_goal)
            if not plan:
                self.ltm.invalidate_plan(active_goal.id, "Failed to create a plan for the goal.")
                return {"description": "*Failed to create a plan for the goal.*"}

            self.ltm.update_plan(active_goal.id, plan)
            return {"description": f"*New plan created for goal '{active_goal.description}'. Starting execution.*"}

        if active_goal.id not in self.workspaces:
            self.workspaces[active_goal.id] = {"goal_description": active_goal.description}
        workspace = self.workspaces[active_goal.id]

        next_step = active_goal.sub_tasks[0]

        if self.skills.is_skill(next_step.action):
            skill = self.skills.get_skill_by_name(next_step.action)
            if skill:
                logging.critical(f"EXPANDING SKILL: '{skill.name}'")
                current_plan = active_goal.sub_tasks
                current_plan.pop(0)
                expanded_plan = skill.action_sequence + current_plan
                self.ltm.update_sub_tasks(active_goal.id, expanded_plan)
                return {"description": f"*Skill '{skill.name}' expanded. Continuing execution.*"}

        logging.info(f"Orchestrator executing step: '{next_step.action}' for persona '{next_step.assigned_persona}'")
        result: Optional[Dict[str, Any]] = None
        if next_step.assigned_persona == 'orchestrator':
            result = await self.execute_single_action(next_step)
            if result.get("status") != "success":
                await self._handle_plan_failure(active_goal, next_step, result.get("description", "Orchestrator task failed."))
                return {"description": f"Step failed: {result.get('description')}. Triggering replan."}
        else:
            agent_names = self.agent_pool.get_agents_by_persona(next_step.assigned_persona)
            if not agent_names:
                await self._handle_plan_failure(active_goal, next_step, f"No agent found with persona '{next_step.assigned_persona}'.")
                return {"description": f"Step failed: No agent found for persona '{next_step.assigned_persona}'. Triggering replan."}
            agent_name = random.choice(agent_names)

            next_step.parameters["workspace"] = workspace

            reply = await self.delegate_task_and_wait(agent_name, next_step)
            
            if next_step.action == "review_plan":
                if not reply or reply.payload.get("status") != "success":
                    error = reply.payload.get('error', 'unknown error') if reply else 'timeout'
                    await self._handle_plan_failure(active_goal, next_step, f"QA review failed: {error}")
                    return {"description": f"Step failed: QA review failed: {error}. Triggering replan."}
                
                if not reply.payload.get("approved"):
                    rejection_reason = reply.payload.get("reason", "Plan rejected by QA without a specific reason.")
                    logging.critical(f"PLAN REJECTED by QA. Reason: {rejection_reason}")
                    await self._handle_plan_failure(active_goal, next_step, rejection_reason)
                    return {"description": f"Plan rejected by QA: {rejection_reason}. Triggering replan."}
                else:
                    logging.info("Plan approved by QA. Proceeding with execution.")
            
            elif not reply or reply.payload.get("status") == "failure":
                error = reply.payload.get('error', 'unknown error') if reply else 'timeout'
                await self._handle_plan_failure(active_goal, next_step, f"Step '{next_step.action}' failed: {error}")
                return {"description": f"Step failed: {error}. Triggering replan."}

            if reply:
                self.workspaces[active_goal.id].update(reply.payload)
                logging.info(f"Orchestrator received result. Workspace now contains keys: {list(self.workspaces[active_goal.id].keys())}")

        if result and result.get("response_text"):
            self.ltm.complete_sub_task(active_goal.id)
            self.ltm.update_goal_status(active_goal.id, 'completed')
            return result

        self.ltm.complete_sub_task(active_goal.id)

        current_goal_state = self.ltm.get_goal_by_id(active_goal.id)
        if not current_goal_state or not current_goal_state.sub_tasks:
            if current_goal_state:
                await self._reflect_on_completed_goal(current_goal_state)
            
            if active_goal.id in self.workspaces:
                del self.workspaces[active_goal.id]
            return {"description": f"*Goal '{active_goal.description}' completed. Post-goal reflection initiated.*"}

        return {"description": f"*(Goal: {active_goal.description}) Step '{next_step.action}' OK.*"}

    async def _reflect_on_perceptions(self) -> bool:
        """Reflects on new perceptions and decides if they warrant interrupting the current goal."""
        events_to_process = list(self.perception_buffer)
        self.perception_buffer.clear()

        logging.info(f"Reflecting on {len(events_to_process)} new perception(s).")
        
        for event in events_to_process:
            await self.memory.add_memory(MemoryEntryModel(
                type='perception',
                content=event.model_dump(mode='json'),
                importance=0.6
            ))
        
        active_goal = self.ltm.get_active_goal()
        current_goal_desc = active_goal.description if active_goal else "I am currently idle."

        prompt = f"""
You are a conscious AGI. You have just perceived the following events in your environment while working on a task.

--- PERCEIVED EVENTS ---
{json.dumps([e.model_dump(mode='json') for e in events_to_process], indent=2)}

--- CURRENT GOAL ---
{current_goal_desc}

--- INSTRUCTIONS ---
Analyze the events. Are they important or surprising enough to justify interrupting your current task?
- If an event is highly relevant, urgent, or unexpected (like a new file appearing that seems related to your goal), you should interrupt.
- If the events are routine or irrelevant, you should not interrupt.

Respond with ONLY a single, valid JSON object with the following keys:
- "interrupt": boolean (true if you should stop your current task to address this)
- "reason": string (a brief explanation for your decision)
- "new_goal_description": string (if interrupting, a new high-level goal to address the perception, otherwise an empty string)
"""
        response_str = ""
        try:
            response_str = await self.introspector.llm_reflect(prompt)
            reflection = json.loads(response_str)

            if reflection.get("interrupt") and reflection.get("new_goal_description"):
                new_desc = reflection["new_goal_description"]
                logging.critical(f"PERCEPTION TRIGGERED NEW GOAL: {new_desc}. Reason: {reflection.get('reason')}")
                new_goal = GoalModel(description=new_desc, sub_tasks=[])
                self.ltm.add_goal(new_goal)
                return True

        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse perception reflection response: {e}. Response was: {response_str}")
        
        return False

    async def _reflect_on_completed_goal(self: 'SymbolicAGI', goal: GoalModel) -> None:
        if not goal.original_plan:
            logging.warning(f"Cannot learn from goal '{goal.id}': No original plan was recorded.")
            self.ltm.update_goal_status(goal.id, 'completed')
            return

        logging.info(f"Reflecting on completed goal '{goal.description}' to check for skill acquisition.")

        plan_json = json.dumps([step.model_dump(mode='json') for step in goal.original_plan], indent=2) # pyright: ignore[reportUnknownMemberType]

        prompt = f"""
You are a meta-cognitive AGI reflecting on a successfully completed task.
Your goal is to decide if the plan used to achieve the task is general and useful enough to be saved as a new, reusable skill.

--- TASK DESCRIPTION ---
"{goal.description}"

--- SUCCESSFUL PLAN ---
{plan_json}

--- ANALYSIS INSTRUCTIONS ---
1.  **Generality**: Is this task something that might be requested again in a different context? (e.g., "summarize a file" is general, "summarize the file 'report_xyz.txt' from yesterday" is not).
2.  **Efficiency**: Was the plan reasonably direct? (Assume this plan was successful).
3.  **Name & Description**: If it's worth learning, propose a short, function-like `skill_name` (e.g., `summarize_and_save_webpage`) and a concise one-sentence `skill_description`.

--- RESPONSE FORMAT ---
Respond with ONLY a single, valid JSON object with the following keys:
- "should_learn": boolean (true if the plan should be saved as a skill)
- "skill_name": string (the proposed name, or an empty string if not learning)
- "skill_description": string (the proposed description, or an empty string if not learning)
"""
        response_str = ""
        try:
            response_str = await self.introspector.llm_reflect(prompt)
            reflection = json.loads(response_str)

            if reflection.get("should_learn"):
                name = reflection.get("skill_name")
                desc = reflection.get("skill_description")
                if name and desc and goal.original_plan:
                    self.skills.add_new_skill(name=name, description=desc, plan=goal.original_plan)
                    await self.meta_cognition.record_meta_event(
                        "skill_transfer",
                        {"name": name, "description": desc, "source_goal": goal.id}
                    )
                    logging.critical(f"LEARNED NEW SKILL: '{name}' from goal '{goal.description}'")
                else:
                    logging.warning("LLM recommended learning a skill but did not provide a valid name or description.")

        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse reflection response for skill learning: {e}. Response was: {response_str}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during skill reflection: {e}", exc_info=True)
        finally:
            self.ltm.update_goal_status(goal.id, 'completed')

    async def start_background_tasks(self: 'SymbolicAGI') -> None:
        """Initializes and starts all asynchronous background tasks."""
        if self._meta_task is None:
            logging.info("Controller: Starting background meta-tasks...")
            self._meta_task = asyncio.create_task(self.run_background_meta_tasks())
        else:
            logging.warning("Controller: Background meta-tasks already started.")
        
        if self._perception_task is None:
            logging.info("Controller: Starting background perception task...")
            self._perception_task = asyncio.create_task(self._workspace_monitor_task())
        else:
            logging.warning("Controller: Background perception task already started.")

    async def shutdown(self: 'SymbolicAGI') -> None:
        """Gracefully shuts down the AGI controller and its background tasks."""
        logging.info("Controller: Initiating shutdown...")
        if self._meta_task and not self._meta_task.done():
            self._meta_task.cancel()
            try:
                await self._meta_task
            except asyncio.CancelledError:
                logging.info("Background meta-task successfully cancelled.")
        
        if self._perception_task and not self._perception_task.done():
            self._perception_task.cancel()
            try:
                await self._perception_task
            except asyncio.CancelledError:
                logging.info("Background perception task successfully cancelled.")

        if self.identity:
            self.identity.save_profile()
        logging.info("Controller: Shutdown complete.")

    async def delegate_task_and_wait(self: 'SymbolicAGI', receiver_id: str, step: ActionStep) -> Optional[MessageModel]:
        orchestrator_queue = self.message_bus.agent_queues.get(self.name)
        if not orchestrator_queue:
            logging.error("Orchestrator is not subscribed to the message bus. Cannot delegate.")
            return None

        task_message = MessageModel(
            sender_id=self.name,
            receiver_id=receiver_id,
            message_type=step.action,
            payload=step.parameters
        )
        await self.message_bus.publish(task_message)
        logging.info(f"Orchestrator delegated task '{step.action}' to '{receiver_id}'. Waiting for reply...")

        try:
            timeout = self.cfg.meta_task_timeout
            reply: MessageModel = await asyncio.wait_for(orchestrator_queue.get(), timeout=timeout)
            logging.info(f"Orchestrator received reply of type '{reply.message_type}' from '{reply.sender_id}'.")
            orchestrator_queue.task_done()
            return reply
        except asyncio.TimeoutError:
            logging.error(f"Timeout: No reply received from '{receiver_id}' for task '{step.action}'.")
            return None

    async def _handle_plan_failure(self, goal: GoalModel, failed_step: ActionStep, error_message: str) -> None:
        """Handles a failed plan step by attempting to replan or failing the goal."""
        logging.error(f"Plan failed for goal '{goal.description}' at step '{failed_step.action}': {error_message}")
        
        new_failure_count = self.ltm.increment_failure_count(goal.id)
        
        if new_failure_count >= goal.max_failures:
            final_error = f"Goal failed after {new_failure_count} attempts. Last error: {error_message}"
            self.ltm.invalidate_plan(goal.id, final_error)
            if goal.id in self.workspaces:
                del self.workspaces[goal.id]
            return
        
        await self._trigger_replanning(goal, failed_step, error_message)

    async def _trigger_replanning(self, goal: GoalModel, failed_step: ActionStep, error_message: str) -> None:
        """Gathers context and calls the planner to create a new plan after a failure."""
        logging.info(f"Triggering replanning for goal: {goal.id}")
        
        failure_context: Dict[str, Any] = {
            "failed_plan": [step.model_dump(mode='json') for step in goal.original_plan or []],
            "failed_step": failed_step.model_dump(mode='json'),
            "error_message": error_message,
            "workspace_keys": list(self.workspaces.get(goal.id, {}).keys())
        }
        
        source_files_result = await self.tools.read_own_source_code(file_name="")
        workspace_files_result = await self.tools.list_files()
        file_manifest = "Source Files: " + ", ".join(source_files_result.get('files', []))
        file_manifest += "\nWorkspace Files: " + ", ".join(workspace_files_result.get('files', []))

        planner_output = await self.planner.decompose_goal_into_plan(
            goal_description=goal.description,
            file_manifest=file_manifest,
            mode=goal.mode,
            failure_context=failure_context
        )
        
        logging.info(f"Replanner Thought: {planner_output.thought}")
        new_plan = planner_output.plan
        
        if new_plan:
            logging.critical(f"REPLAN SUCCESSFUL. New plan created for goal '{goal.id}'.")
            self.ltm.update_plan(goal.id, new_plan)
        else:
            logging.error(f"REPLAN FAILED. Could not generate a new plan for goal '{goal.id}'.")
            self.ltm.invalidate_plan(goal.id, "Replanning process failed to generate a valid new plan.")

    async def execute_plan(self: 'SymbolicAGI', plan: List[ActionStep]) -> None:
        logging.info(f"Executing an internal plan with {len(plan)} steps.")
        for step in plan:
            result = await self.execute_single_action(step)
            if result.get("status") != "success":
                logging.error(f"Internal plan execution failed at step '{step.action}': {result.get('description')}")
                break

    async def execute_single_action(self: 'SymbolicAGI', step: ActionStep) -> Dict[str, Any]:
        self.identity.consume_energy()
        try:
            if step.action in self.orchestrator_actions:
                action_func = self.orchestrator_actions[step.action]
                return await action_func(**step.parameters)

            tool_method = getattr(self.tools, step.action, None)
            if callable(tool_method) and asyncio.iscoroutinefunction(tool_method):
                active_goal = self.ltm.get_active_goal()
                if active_goal:
                    step.parameters['workspace'] = self.workspaces.get(active_goal.id, {})
                await self.identity.record_tool_usage(step.action, step.parameters)
                return await tool_method(**step.parameters)

            world_action = getattr(self.world, f"_action_{step.action}", None)
            if callable(world_action):
                if asyncio.iscoroutinefunction(world_action):
                    return cast(Dict[str, Any], await world_action(**step.parameters))
                else:
                    return cast(Dict[str, Any], world_action(**step.parameters))

            return {"status": "failure", "description": f"Unknown or non-awaitable action: {step.action}"}
        except Exception as e:
            await self.meta_cognition.record_meta_event("critical_error", f"{step.action}: {e}")
            return {"status": "failure", "description": f"Error: {e}"}

    async def _action_create_goal(self: 'SymbolicAGI', description: str, sub_tasks: List[Any]) -> Dict[str, Any]:
        repaired = await self.planner.decompose_goal_into_plan(sub_tasks, "") # type: ignore
        goal = GoalModel(description=description, sub_tasks=repaired.plan) # pyright: ignore[reportArgumentType]
        self.ltm.add_goal(goal)
        return {"status": "success", "description": "Goal created."}

    async def _action_respond_to_user(self: 'SymbolicAGI', text: str) -> Dict[str, Any]:
        return {"status": "success", "response_text": text}

    def _wrap_content(self: 'SymbolicAGI', value: Any, default_key: str = "text") -> Dict[str, Any]:
        if isinstance(value, dict):
            return cast(Dict[str, Any], value)
        return {default_key: value}

    async def _workspace_monitor_task(self) -> None:
        """Periodically scans the workspace for file changes and generates perception events."""
        logging.info("Workspace monitor task started.")
        known_files: Set[str] = set()
        
        try:
            initial_result = await self.tools.list_files()
            if initial_result.get('status') == 'success':
                known_files = set(initial_result.get('files', []))
        except Exception as e:
            logging.error(f"Initial workspace scan failed: {e}")

        while True:
            await asyncio.sleep(5)
            try:
                current_result = await self.tools.list_files()
                if current_result.get('status') != 'success':
                    continue
                
                current_files = set(current_result.get('files', []))
                
                new_files = current_files - known_files
                deleted_files = known_files - current_files
                
                for f in new_files:
                    event = PerceptionEvent(source='workspace', type='file_created', content={'file_path': f})
                    self.perception_buffer.append(event)
                    logging.info(f"PERCEPTION: New file created in workspace: {f}")

                for f in deleted_files:
                    event = PerceptionEvent(source='workspace', type='file_deleted', content={'file_path': f})
                    self.perception_buffer.append(event)
                    logging.info(f"PERCEPTION: File deleted from workspace: {f}")
                
                known_files = current_files

            except asyncio.CancelledError:
                logging.info("Workspace monitor task received cancel signal.")
                break
            except Exception as e:
                logging.error(f"Error in workspace monitor task: {e}", exc_info=True)

    async def startup_validation(self: 'SymbolicAGI') -> None:
        logging.info("Running startup validation…")
        needs_saving = False
        for goal in list(self.ltm.goals.values()):
            repaired = await self.planner.decompose_goal_into_plan(goal.sub_tasks, "") # type: ignore
            if goal.sub_tasks != repaired.plan:
                goal.sub_tasks = repaired.plan
                needs_saving = True
        if needs_saving:
            logging.warning("Repaired malformed plans during startup.")
            self.ltm._save_goals() # type: ignore [protected-access]