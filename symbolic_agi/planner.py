# symbolic_agi/planner.py

import json
import logging
from typing import Any, Dict, Optional

from .recursive_introspector import RecursiveIntrospector
from .schemas import ActionStep, GoalMode, PlannerOutput
from .skill_manager import SkillManager
from .tool_plugin import ToolPlugin
from .agent_pool import DynamicAgentPool

# Constants
RESPONSE_FORMAT = '{"thought": "...", "plan": [{"action": "...", "parameters": {}, "assigned_persona": "..."}]}'

class Planner:
    """
    A dedicated class for creating and repairing plans for the AGI.
    It uses an introspector to reason about goals and available capabilities.
    """
    def __init__(
        self,
        introspector: RecursiveIntrospector,
        skill_manager: SkillManager,
        agent_pool: DynamicAgentPool,
        tool_plugin: ToolPlugin
    ):
        self.introspector = introspector
        self.skills = skill_manager
        self.agent_pool = agent_pool
        self.tools = tool_plugin

    def _build_tools_prompt(self, mode: GoalMode) -> str:
        """Build the tools prompt based on the goal mode."""
        docs_only_tools = """
- 'read_file', 'write_file', 'list_files', 'read_core_file', 'analyze_data', 'read_own_source_code', 'web_search', 'browse_webpage'
"""
        code_allowed_tools = """
- 'read_file', 'write_file', 'list_files', 'read_core_file', 'analyze_data', 'read_own_source_code', 'execute_python_code', 'web_search', 'browse_webpage'
"""
        
        return f"""
The 'orchestrator' persona can use the following tools directly:
{docs_only_tools if mode == 'docs' else code_allowed_tools}
"""

    def _build_failure_prompt(self, goal_description: str, failure_context: Dict[str, Any], 
                             persona_prompt: str, available_skills: str, 
                             orchestrator_tools_prompt: str, file_manifest: str) -> str:
        """Build prompt for replanning after failure."""
        
        return f"""
You are an expert troubleshooter AGI. A previous attempt to achieve a goal failed. Your task is to analyze the failure and create a new, corrected plan.

--- ORIGINAL GOAL ---
{goal_description}

--- FAILED PLAN CONTEXT ---
{json.dumps(failure_context, indent=2)}

--- AVAILABLE CAPABILITIES ---
{persona_prompt}
{available_skills}
{orchestrator_tools_prompt}

--- AVAILABLE FILES ---
{file_manifest}

--- INSTRUCTIONS ---
1.  **Analyze**: The previous plan failed. The error message was: "{failure_context.get('error_message', 'N/A')}". This means you used the wrong tool or the wrong parameters.
2.  **Correction Rule**: To read a core configuration file like `consciousness_profile.json`, you MUST use the `read_core_file` tool. To read a file in the workspace, you MUST use the `read_file` tool.
3.  **New Strategy**: Your new plan MUST start with the correct tool to read the required file. Then, it MUST use the `analyze_data` tool to extract the necessary information.
4.  **Respond**: Decompose the corrected approach into a JSON array of action steps. Respond ONLY with the raw JSON object: {RESPONSE_FORMAT}
"""

    def _build_initial_prompt(self, goal_description: str, mode: GoalMode,
                             persona_prompt: str, available_skills: str,
                             orchestrator_tools_prompt: str, file_manifest: str) -> str:
        """Build prompt for initial planning."""
        
        return f"""
You are a master project manager AGI. Your task is to decompose a high-level goal into a series of concrete, logical steps.

# GOAL MODE: {mode.upper()}
{'You are in "docs" mode. You MUST NOT use the "write_code" or "execute_python_code" actions.' if mode == 'docs' else ''}

# AVAILABLE CAPABILITIES
{persona_prompt}
{available_skills}
{orchestrator_tools_prompt}

# AVAILABLE FILES
{file_manifest}

# MANDATORY LOGIC FOR FILE ANALYSIS
To answer a question about a file, you MUST follow this exact two-step process:
1.  Use `read_file` (for workspace files) or `read_core_file` (for config files) to get the file's content.
2.  Use `analyze_data` with the `data` parameter set to the key `content` and a `query` to extract the specific information. The output of `analyze_data` is then available in the workspace under the key "answer".

Goal: "{goal_description}"

--- INSTRUCTIONS ---
1.  **Think**: First, write a step-by-step "thought" process for how you will achieve the goal, following the mandatory logic above.
2.  **Plan**: Based on your thought process, create a JSON array of action steps.
3.  **Respond**: Format your entire response as a single JSON object: {RESPONSE_FORMAT}
"""

    async def _generate_plan_with_retry(self, master_prompt: str) -> Optional[Dict[str, Any]]:
        """Generate plan with JSON repair retry logic."""
        plan_str = ""
        
        for attempt in range(2):
            try:
                if attempt == 0:
                    plan_str = await self.introspector.llm_reflect(master_prompt)
                else:
                    logging.warning(f"Malformed JSON response detected. Attempting repair on: {plan_str[:200]}")
                    forceful_prompt = f"""
The following text is NOT valid JSON.
--- BROKEN TEXT ---
{plan_str}
---
FIX THIS. Respond ONLY with the corrected, raw JSON object in the format {RESPONSE_FORMAT}.
"""
                    plan_str = await self.introspector.llm_reflect(forceful_prompt)

                if "```json" in plan_str:
                    plan_str = plan_str.partition("```json")[2].partition("```")[0]
                
                return json.loads(plan_str)

            except json.JSONDecodeError as e:
                if attempt >= 1:
                    logging.error(f"Final repair attempt failed to produce valid JSON: {e}. Response was: {plan_str}")
                    return None
        
        return None

    def _validate_and_build_output(self, planner_output_dict: Dict[str, Any], 
                                  goal_description: str, failure_context: Optional[Dict[str, Any]]) -> PlannerOutput:
        """Validate plan and build final output with optional QA review step."""
        try:
            validated_plan = [ActionStep.model_validate(item) for item in planner_output_dict.get("plan", [])]
            thought = planner_output_dict.get("thought", "No thought recorded.")
            
            if validated_plan and failure_context is None:
                logging.info(f"Plan generated with {len(validated_plan)} steps. Adding QA review step.")
                review_step = ActionStep(
                    action="review_plan",
                    parameters={
                        "original_goal": goal_description,
                        "plan_to_review": [step.model_dump() for step in validated_plan]
                    },
                    assigned_persona="qa"
                )
                return PlannerOutput(thought=thought, plan=[review_step] + validated_plan)
            
            return PlannerOutput(thought=thought, plan=validated_plan)

        except Exception as e:
            logging.error(f"Failed to validate repaired plan structure: {e}")
            return PlannerOutput(thought=f"Failed to validate plan structure: {e}", plan=[])

    async def decompose_goal_into_plan(
        self, 
        goal_description: str, 
        file_manifest: str,
        mode: GoalMode = 'code',
        failure_context: Optional[Dict[str, Any]] = None
    ) -> PlannerOutput:
        """
        Uses an LLM to generate a plan, then validates and repairs it to ensure it is always valid JSON.
        """
        
        available_skills = self.skills.get_formatted_definitions()
        persona_prompt = self.agent_pool.get_persona_capabilities_prompt()
        orchestrator_tools_prompt = self._build_tools_prompt(mode)

        if failure_context:
            logging.critical(f"REPLANNING for goal: '{goal_description}'")
            master_prompt = self._build_failure_prompt(
                goal_description, failure_context, persona_prompt, 
                available_skills, orchestrator_tools_prompt, file_manifest
            )
        else:
            logging.info(f"Decomposing goal: '{goal_description}'")
            master_prompt = self._build_initial_prompt(
                goal_description, mode, persona_prompt,
                available_skills, orchestrator_tools_prompt, file_manifest
            )

        planner_output_dict = await self._generate_plan_with_retry(master_prompt)
        
        if planner_output_dict is None:
            return PlannerOutput(thought="Planner returned no output.", plan=[])

        return self._validate_and_build_output(planner_output_dict, goal_description, failure_context)