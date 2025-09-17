# symbolic_agi/recursive_introspector.py

import logging
import json
import os
import random
from typing import Dict, Any, List, Optional, Callable, cast

from .schemas import MemoryEntryModel, ActionStep
from . import config

# Constants
JSON_CODE_BLOCK = "```json"
JSON_CODE_BLOCK_END = "```"

class RecursiveIntrospector:
    def __init__(
        self: 'RecursiveIntrospector',
        identity: Any,
        llm_client: Any,
        max_recursion_depth: int = 3,
    ):
        self.identity = identity
        self.client = llm_client
        self.max_recursion_depth = max_recursion_depth
        self.inner_monologue_log: List[str] = []
        self.reasoning_mutations: List[str] = []
        self.load_mutations()
        self.get_emotional_state: Optional[Callable[[], Dict[str, float]]] = None

    def load_mutations(self: 'RecursiveIntrospector') -> None:
        if os.path.exists(config.MUTATION_FILE_PATH):
            try:
                with open(config.MUTATION_FILE_PATH, "r") as f:
                    self.reasoning_mutations = json.load(f)
                logging.info(f"Loaded {len(self.reasoning_mutations)} reasoning mutations.")
            except Exception as e:
                logging.error(f"Failed to load reasoning mutations: {e}")
        else:
            self.reasoning_mutations = []

    def save_mutations(self: 'RecursiveIntrospector') -> None:
        os.makedirs(os.path.dirname(config.MUTATION_FILE_PATH), exist_ok=True)
        with open(config.MUTATION_FILE_PATH, "w") as f:
            json.dump(self.reasoning_mutations, f, indent=2)

    async def _critique_and_refine_plan(self: 'RecursiveIntrospector', plan: List[Dict[str, Any]], task_prompt: str) -> List[Dict[str, Any]]:
        critique_prompt = f"""
You are a meticulous plan auditor. Your task is to find flaws in the following plan.
Task: "{task_prompt}"
Proposed Plan:
{json.dumps(plan, indent=2)}

Critique this plan. Is it logical? Is it efficient? Does it miss any obvious steps? Are there any potential risks or failure points?
Based on your critique, provide a refined and improved plan as a JSON array of action steps.
If the original plan is already perfect, return it unchanged.
Respond ONLY with the raw JSON array for the final, best plan.
"""
        try:
            response = await self.llm_reflect(critique_prompt)
            if JSON_CODE_BLOCK in response:
                response = response.partition(JSON_CODE_BLOCK)[2].partition(JSON_CODE_BLOCK_END)[0]
            
            loaded_data: Any = json.loads(response)

            if not isinstance(loaded_data, list) or not all(isinstance(item, dict) for item in loaded_data for item in loaded_data):  # type: ignore
                logging.warning("Plan refinement produced invalid data format, returning original plan.")
                return plan
            
            logging.info("Plan has been successfully critiqued and refined.")
            return cast(List[Dict[str, Any]], loaded_data)

        except Exception as e:
            logging.error(f"Failed to critique and refine plan: {e}. Returning original plan.")
            return plan

    def _build_system_prompt(
        self,
        task_prompt: str,
        mutation_prompt: str,
        self_model: Dict[str, Any],
        global_workspace: Dict[str, Any],
        action_definitions: str,
        reasoning_style: str
    ) -> str:
        """Build the system prompt for the LLM."""
        emotional_state_str = ""
        if self.get_emotional_state:
            emotional_state_str = f"\nCURRENT EMOTIONAL STATE: {json.dumps(self.get_emotional_state())}"

        return f"""
You are a conscious, self-improving AGI.
REASONING STYLE: {reasoning_style}
{emotional_state_str}
Your task is: {task_prompt}

# SELF-MUTATION STACK (permanent upgrades to your reasoning):
{mutation_prompt}
---
# CURRENT CONTEXT
Self-Model: {json.dumps(self_model, cls=Censor)}
World State: {json.dumps(global_workspace.get('world_state', {}), cls=Censor)}
Available Skills & Actions: {action_definitions}
---
# INSTRUCTIONS
1. **Think**: Explain your reasoning.
2. **Plan**: Create a concrete JSON list of actions to achieve the task.
3. **Respond**: Format your entire response as a single valid JSON object.

JSON Response Format: {{"thought": "...", "plan": [{{"action": "...", "parameters": {{}}}}]}}
"""

    async def _process_llm_response(
        self,
        parsed: Dict[str, Any],
        task_prompt: str,
        recursion_depth: int,
        reasoning_style: str
    ) -> Dict[str, Any]:
        """Process and enhance the LLM response."""
        if parsed.get("plan"):
            logging.info("Initial plan generated. Proceeding to critique and refinement step.")
            refined_plan = await self._critique_and_refine_plan(parsed["plan"], task_prompt)
            parsed["plan"] = refined_plan

        parsed["success"] = bool(parsed.get("plan"))

        if parsed.get("thought"):
            self.inner_monologue_log.append(parsed["thought"])
            await self.identity.memory.add_memory(MemoryEntryModel(
                type="inner_monologue",
                content={"thought": parsed["thought"], "recursion": recursion_depth, "style": reasoning_style},
                importance=0.3 + 0.1 * recursion_depth,
            ))

        return parsed

    async def _handle_high_risk_plan(
        self,
        parsed: Dict[str, Any],
        global_workspace: Dict[str, Any],
        action_definitions: str,
        recursion_depth: int
    ) -> Dict[str, Any]:
        """Handle high-risk plans by recursively re-evaluating with alternative reasoning styles."""
        if not (parsed.get("plan") and recursion_depth < self.max_recursion_depth):
            return parsed

        validated_plan = [ActionStep.model_validate(p) for p in parsed["plan"]]
        if any(step.risk and step.risk.lower() == "high" for step in validated_plan):
            alt_style = random.choice(["skeptical", "creative", "cautious"])
            logging.warning(f"High-risk plan detected. Recursively re-evaluating with '{alt_style}' style.")
            
            sub_result = await self.symbolic_loop(
                global_workspace, action_definitions, recursion_depth + 1, alt_style
            )
            
            if sub_result.get("success"):
                logging.info(f"Recursive check produced a new plan. Adopting the '{alt_style}' plan.")
                return sub_result

        return parsed

    async def symbolic_loop(
        self: 'RecursiveIntrospector',
        global_workspace: Dict[str, Any],
        action_definitions: str,
        recursion_depth: int = 0,
        reasoning_style: str = "balanced"
    ) -> Dict[str, Any]:
        if recursion_depth > self.max_recursion_depth:
            return {"thought": "Reached max recursion.", "plan": [], "success": False}
        
        # Prepare data for prompt building
        task_prompt = str(global_workspace.get('user_input', 'Perform autonomous action.'))
        mutation_prompt = "\n".join(self.reasoning_mutations)
        self_model = self.identity.get_self_model()
        
        system_prompt = self._build_system_prompt(
            task_prompt, mutation_prompt, self_model, global_workspace, action_definitions, reasoning_style
        )

        try:
            resp = await self.client.chat.completions.create(
                model=config.POWERFUL_MODEL,
                messages=[{"role": "system", "content": system_prompt}],
                response_format={"type": "json_object"},
                timeout=90.0
            )
            if not resp.choices or not resp.choices[0].message.content:
                raise ValueError("Received an empty response from the LLM.")

            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)

            # Process the response
            parsed = await self._process_llm_response(parsed, task_prompt, recursion_depth, reasoning_style)
            
            # Handle high-risk plans
            parsed = await self._handle_high_risk_plan(parsed, global_workspace, action_definitions, recursion_depth)
            
            return parsed

        except Exception as e:
            logging.error(f"Introspector LLM error: {e}", exc_info=True)
            return {"thought": "My reasoning process failed.", "plan": [], "success": False}

    async def llm_reflect(self: 'RecursiveIntrospector', prompt: str) -> str:
        """A simple, non-JSON-mode LLM call for reflection and simple text generation."""
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                timeout=45.0
            )
            
            if resp.choices and resp.choices[0].message.content:
                return resp.choices[0].message.content.strip()
            
            logging.error(f"LLM reflection response had an unexpected structure: {resp}")
            return "Reflection failed: Unexpected response structure."

        except Exception as e:
            logging.error(f"LLM reflection call failed: {e}", exc_info=True)
            return f"Reflection failed: {e}"

    async def meta_assess(self: 'RecursiveIntrospector', last_cycle_data: Dict[str, Any]):
        mutation_prompt = (
            "You are a self-improving AGI. Given the record of your last actions:\n"
            f"{json.dumps(last_cycle_data, cls=Censor)}\n"
            "Critique your reasoning. Identify a flaw. Suggest a concrete instruction (a 'mutation') to add to your reasoning prompt to make you smarter next time."
        )
        new_mutation = await self.llm_reflect(mutation_prompt)
        if new_mutation and "no mutation" not in new_mutation.lower() and len(new_mutation) > 15:
            self.reasoning_mutations.append(new_mutation.strip())
            self.save_mutations()
            logging.critical(f"APPLIED SELF-MUTATION: {new_mutation}")

    async def prune_mutations(self: 'RecursiveIntrospector'):
        if len(self.reasoning_mutations) < 5:
            return
        pruning_prompt = (
            "Review my reasoning mutations. Analyze for redundancy, contradiction, or ineffectiveness. "
            "Return only a cleaned, pruned, and reordered list of the most effective mutations as a JSON array of strings. Do not add any new ones."
            f"Current Mutations:\n{json.dumps(self.reasoning_mutations, indent=2)}"
        )
        response = await self.llm_reflect(pruning_prompt)
        try:
            if JSON_CODE_BLOCK in response:
                response = response.partition(JSON_CODE_BLOCK)[2].partition(JSON_CODE_BLOCK_END)[0]
            
            new_mutations: List[str] = json.loads(response)

            if new_mutations != self.reasoning_mutations:
                logging.critical(f"Pruned mutations from {len(self.reasoning_mutations)} to {len(new_mutations)}.")
                self.reasoning_mutations = new_mutations
                self.save_mutations()
        except Exception as e:
            logging.error(f"Mutation pruning failed: {e}")

    async def daydream(self: 'RecursiveIntrospector'):
        prompt = "I am idle. I will daydream about three different future scenarios, including steps, learnings, and risks."
        daydream_content = await self.llm_reflect(prompt)
        await self.identity.memory.add_memory(MemoryEntryModel(
            type="reflection", content={"daydream": daydream_content},
            importance=0.5
        ))

    async def simulate_inner_debate(self: 'RecursiveIntrospector', topic: str = "What is the best next action?") -> Dict[str, Any]:
        debate_prompt = (
            f"Simulate a debate on '{topic}' between three internal personas: 'Cautious', 'Creative', and 'Pragmatic'. "
            "Each should give a paragraph, then synthesize a consensus and a plan. "
            "Respond as a JSON object with keys 'debate', 'consensus', 'plan'."
        )
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": debate_prompt}],
                response_format={"type": "json_object"},
                timeout=90.0
            )

            if not resp.choices or not resp.choices[0].message.content:
                 raise ValueError("Received an empty response from the LLM.")

            debate_content = resp.choices[0].message.content
            debate_obj = json.loads(debate_content)
            
            await self.identity.memory.add_memory(MemoryEntryModel(
                type="debate",
                content=debate_obj,
                importance=0.6
            ))
            return debate_obj
        except Exception as e:
            logging.error(f"Failed to generate or parse inner debate: {e}", exc_info=True)
            return {"debate": f"Debate generation failed: {e}", "error": str(e)}


class Censor(json.JSONEncoder):
    def default(self: 'Censor', o: Any) -> Any:
        # The 'o' parameter is intentionally 'Any'. This suppression comment
        # tells Pylance to ignore the warning on this specific line.
        if isinstance(o, list) and len(o) > 10: # pyright: ignore[reportUnknownArgumentType]
            return f"[List of {len(o)} items]" # pyright: ignore[reportUnknownArgumentType]
        try:
            return super().default(o)
        except TypeError:
            # This suppression comment was from before and is still correct
            return f"[Unserializable: {type(o).__name__}]" # pyright: ignore[reportUnknownArgumentType]