# symbolic_agi/agent.py

import asyncio
import logging
import json
from typing import Dict, Any, TYPE_CHECKING

from openai import AsyncOpenAI

from .schemas import MessageModel

if TYPE_CHECKING:
    from .message_bus import MessageBus

# Constants
NO_CONTENT_ERROR = "No content returned from LLM."

class Agent:
    def __init__(self, name: str, message_bus: "MessageBus", api_client: AsyncOpenAI):
        self.name = name
        self.persona = name.split('_')[-2].lower() if '_' in name else 'specialist'
        self.bus = message_bus
        self.client = api_client
        self.inbox = self.bus.subscribe(self.name)
        self.running = True
        self.skills: Dict[str, Any] = self._initialize_skills()
        logging.info(f"Agent '{self.name}' initialized with persona '{self.persona}' and skills: {list(self.skills.keys())}")

    def _initialize_skills(self) -> Dict[str, Any]:
        if self.persona == 'coder':
            return {"write_code": self.skill_write_code}
        elif self.persona == 'research':
            return {"research_topic": self.skill_research_topic}
        elif self.persona == 'qa':
            # --- UPGRADE: Add the new review_plan skill ---
            return {"review_code": self.skill_review_code, "review_plan": self.skill_review_plan}
        return {}

    async def run(self):
        while self.running:
            try:
                message: MessageModel = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                await self.handle_message(message)
                self.inbox.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self.running = False
                logging.info(f"Agent '{self.name}' received cancel signal.")
                raise  # Re-raise CancelledError after cleanup
        logging.info(f"Agent '{self.name}' has shut down.")

    async def handle_message(self, message: MessageModel):
        logging.info(f"Agent '{self.name}' received message of type '{message.message_type}' from '{message.sender_id}'.")
        
        if message.message_type in self.skills:
            skill_to_run = self.skills[message.message_type]
            result_payload = await skill_to_run(message.payload)
            
            reply = MessageModel(
                sender_id=self.name,
                receiver_id=message.sender_id,
                message_type=f"{message.message_type}_result",
                payload=result_payload
            )
            await self.bus.publish(reply)
        elif message.message_type == "new_skill_broadcast":
            logging.info(f"Agent '{self.name}' learned about a new skill: {message.payload.get('skill_name')}")
        else:
            logging.warning(f"Agent '{self.name}' does not know how to handle message type '{message.message_type}'.")

    # --- NEW METHOD ---
    async def skill_review_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reviews a plan for logical flaws, inefficiency, or misinterpretation of the original goal.
        """
        goal = params.get("original_goal", "No goal provided.")
        plan = params.get("plan_to_review", [])
        
        plan_str = json.dumps(plan, indent=2)

        prompt = f"""
You are a meticulous and skeptical QA engineer. Your task is to review a proposed plan against the original goal.

--- ORIGINAL GOAL ---
"{goal}"

--- PROPOSED PLAN ---
{plan_str}

--- INSTRUCTIONS ---
1.  **Check for Correctness**: Does the plan actually achieve the original goal?
2.  **Check for Misinterpretation**: Did the planner misunderstand the user's intent (e.g., asked to write a document but the plan writes code)?
3.  **Check for Logical Flaws**: Are there any obviously illogical steps or incorrect tool usage?
4.  **Check for Inefficiency**: Is there a much simpler or more direct way to achieve the goal?

Based on your analysis, provide a final JSON object with two keys:
- "approved": A boolean (true if the plan is good, false if it needs to be redone).
- "reason": A brief, critical explanation for your decision. If rejecting, clearly state WHY the plan is flawed.

Respond ONLY with the valid JSON object.
"""
        try:
            resp = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"}
            )
            if resp.choices and resp.choices[0].message.content:
                review_data = json.loads(resp.choices[0].message.content)
                return {"status": "success", **review_data}
            return {"status": "failure", "error": NO_CONTENT_ERROR}
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    async def skill_write_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt", "Write a simple hello world python script.")
        context = params.get("context", "")
        workspace = params.get("workspace", {})
        full_context = f"{context}\n\nResearch Summary:\n{workspace.get('research_summary', 'N/A')}"

        llm_prompt = f"You are a master programmer. Based on the following context, write the Python code requested.\n\nContext: {full_context}\n\nRequest: {prompt}\n\nRespond with ONLY the raw Python code inside a ```python ... ``` block."
        
        try:
            resp = await self.client.chat.completions.create(model="gpt-4.1", messages=[{"role": "system", "content": llm_prompt}])
            content = resp.choices[0].message.content
            if content is None:
                return {"status": "failure", "error": NO_CONTENT_ERROR}
            code = content.strip()
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            return {"status": "success", "generated_code": code}
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    async def skill_research_topic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        topic = params.get("topic", "The history of artificial intelligence.")
        llm_prompt = f"You are a master researcher. Provide a concise but comprehensive summary of the following topic: {topic}"
        
        try:
            resp = await self.client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": llm_prompt}])
            content = resp.choices[0].message.content
            if content is None:
                return {"status": "failure", "error": NO_CONTENT_ERROR}
            summary = content.strip()
            return {"status": "success", "research_summary": summary}
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    async def skill_review_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        workspace = params.get("workspace", {})
        code_to_review = workspace.get("generated_code", "# No code provided to review.")

        llm_prompt = f"You are a master QA engineer. Review the following Python code for bugs, style issues, and potential improvements. Provide your feedback as a brief report.\n\nCode:\n```python\n{code_to_review}\n```"
        
        try:
            resp = await self.client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": llm_prompt}])
            content = resp.choices[0].message.content
            if content is None:
                return {"status": "failure", "error": NO_CONTENT_ERROR}
            review = content.strip()
            return {"status": "success", "code_review": review}
        except Exception as e:
            return {"status": "failure", "error": str(e)}