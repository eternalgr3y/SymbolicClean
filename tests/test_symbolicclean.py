import json
import pytest
from typing import Dict, Any
from pathlib import Path

# Import the classes from the SymbolicClean project
try:
    from symbolic_agi.schemas import EmotionalState, GoalModel, ActionStep, MessageModel
    from symbolic_agi import config
    from symbolic_agi.symbolic_memory import SymbolicMemory
    from symbolic_agi.long_term_memory import LongTermMemory
    from symbolic_agi.symbolic_identity import SymbolicIdentity
    from symbolic_agi.skill_manager import SkillManager
    from symbolic_agi.agent_pool import DynamicAgentPool
    from symbolic_agi.message_bus import MessageBus
    from symbolic_agi.planner import Planner, PlannerOutput
    from symbolic_agi.agent import Agent
    from symbolic_agi.tool_plugin import ToolPlugin
except ImportError:
    # If the SymbolicClean project is not available, skip the tests. This allows
    # the test file to be imported without failing in environments where the
    # project code isn't present (e.g. on first run).
    pytest.skip("SymbolicClean project not found in PYTHONPATH", allow_module_level=True)


class DummyIntrospector:
    """Dummy Introspector to simulate LLM reflections for Planner."""
    def __init__(self):
        self.call_count = 0

    def llm_reflect(self, prompt: str) -> str:
        """Return a dummy plan in JSON. If called a second time for repair, return a simple valid JSON."""
        self.call_count += 1
        if "FIX THIS" in prompt or self.call_count > 1:
            # Return a minimal valid JSON on repair attempt
            return '{"thought": "Repaired thought", "plan": []}'
        # Return an initial plan JSON with one action (to trigger adding QA step)
        return json.dumps({
            "thought": "Initial plan",
            "plan": [
                {"action": "write_code", "parameters": {}, "assigned_persona": "coder"}
            ]
        })


@pytest.mark.asyncio
async def test_emotional_state_clamp():
    """EmotionalState.clamp should bound all emotions between 0.0 and 1.0."""
    es = EmotionalState(joy=1.5, sadness=-0.2, anger=0.0, fear=2.0, surprise=0.5, disgust=1.2, trust=-0.1, frustration=0.0)
    es.clamp()
    # All values should now be within [0.0, 1.0]
    for _, value in es.model_dump().items():
        assert 0.0 <= value <= 1.0
    # Specific fields clamped
    assert es.joy == pytest.approx(1.0)    # type: ignore # pytest approx typing  # was 1.5
    assert es.sadness == pytest.approx(0.0)  # type: ignore # pytest approx typing  # was -0.2
    assert es.fear == pytest.approx(1.0)   # type: ignore # pytest approx typing  # was 2.0
    assert es.disgust == pytest.approx(1.0)  # type: ignore # pytest approx typing  # was 1.2
    assert es.trust == pytest.approx(0.0)  # type: ignore # pytest approx typing  # was -0.1


def test_long_term_memory_add_and_archive(tmp_path: Path) -> None:
    """LongTermMemory should save goals and archive completed/failed ones properly."""
    # Use temporary files for goal storage
    config.LONG_TERM_GOAL_PATH = str(tmp_path / "goals.json")
    config.GOAL_ARCHIVE_PATH = str(tmp_path / "archive.json")
    ltm = LongTermMemory()
    # Start with no goals
    assert ltm.get_active_goal() is None

    # Add a new goal
    goal = GoalModel(description="Test Goal", sub_tasks=[])
    ltm.add_goal(goal)
    # The goal should be retrievable and active
    assert ltm.get_goal_by_id(goal.id) is not None
    active_goal = ltm.get_active_goal()
    assert active_goal is not None
    assert active_goal.description == "Test Goal"

    # Mark goal as completed -> it should be archived and removed from active goals
    ltm.update_goal_status(goal.id, status='completed')
    assert ltm.get_goal_by_id(goal.id) is None  # removed from active
    # The archive file should contain the goal
    with open(config.GOAL_ARCHIVE_PATH, 'r') as f:
        archive = json.load(f)
    assert goal.id in archive and archive[goal.id]["description"] == "Test Goal"

    # Add another goal and mark as failed via invalidate_plan
    goal2 = GoalModel(description="Another Goal", sub_tasks=[])
    ltm.add_goal(goal2)
    ltm.invalidate_plan(goal2.id, reason="Plan failure")
    # Goal2 should be archived as failed
    with open(config.GOAL_ARCHIVE_PATH, 'r') as f:
        archive = json.load(f)
    assert goal2.id in archive and archive[goal2.id]["status"] == "failed"
    # And no active goals remain
    assert ltm.get_active_goal() is None


def test_symbolic_identity_state_and_energy(tmp_path: Path) -> None:
    """SymbolicIdentity should update transient state and handle energy correctly."""
    # Use temp file for identity profile
    profile_path = tmp_path / "identity.json"
    config.IDENTITY_PROFILE_PATH = str(profile_path)
    memory = SymbolicMemory(client=None)  # type: ignore # Test mock
    ident = SymbolicIdentity(memory=memory, file_path=config.IDENTITY_PROFILE_PATH)
    # Initial state defaults
    model = ident.get_self_model()
    assert model["name"] == "SymbolicAGI"
    assert model["cognitive_energy"] == 100
    assert model["emotional_state"] == "curious"

    # Update state via update_self_model_state
    ident.update_self_model_state({"current_state": "busy", "perceived_location": "lab"})
    model2 = ident.get_self_model()
    assert model2["current_state"] == "busy"
    assert model2["perceived_location_in_world"] == "lab"

    # Energy consumption and regeneration
    ident.cognitive_energy = 5
    ident.consume_energy(amount=3)
    assert ident.cognitive_energy == 2
    ident.consume_energy(amount=10)  # not go below 0
    assert ident.cognitive_energy == 0
    ident.regenerate_energy(amount=5)
    assert ident.cognitive_energy == 5
    ident.regenerate_energy(amount=100)  # not exceed max_energy (100)
    assert ident.cognitive_energy == 100


def test_skill_manager_learn_and_override(tmp_path: Path) -> None:
    """SkillManager should add new skills and override existing by name."""
    # Use temp file for skills storage
    config.SKILLS_PATH = str(tmp_path / "skills.json")
    sm = SkillManager()
    # Initially no skills
    assert sm.get_skill_by_name("TestSkill") is None

    # Helper function to create and add a skill
    def add_test_skill(name: str, description: str, action: str, persona: str) -> None:
        plan = [ActionStep(action=action, parameters={}, assigned_persona=persona)]  # type: ignore # Test flexibility
        sm.add_new_skill(name=name, description=description, plan=plan)

    # Add a new skill
    add_test_skill("TestSkill", "A test skill", "write_code", "coder")
    skill = sm.get_skill_by_name("TestSkill")
    assert skill is not None
    assert skill.description == "A test skill"
    assert skill.action_sequence[0].action == "write_code"

    # Add another skill with same name to test override
    add_test_skill("TestSkill", "Updated skill", "research_topic", "research")
    skill_updated = sm.get_skill_by_name("TestSkill")
    # The skill should be updated (new description and action sequence)
    assert skill_updated is not None
    assert skill_updated.description == "Updated skill"
    assert skill_updated.action_sequence[0].action == "research_topic"
    # Only one skill with that name exists (old one overwritten)
    skill_ids = [s.name for s in sm.skills.values()]
    assert skill_ids.count("TestSkill") == 1

    # get_formatted_definitions should list innate and learned skills
    fmt = sm.get_formatted_definitions()
    # It should contain a section for learned skills
    assert "# LEARNED SKILLS" in fmt
    assert 'action: "TestSkill", description: "Updated skill"' in fmt


def test_agent_pool_and_persona_registration():
    """DynamicAgentPool should register agents and allow lookup by persona."""
    bus = MessageBus()
    pool = DynamicAgentPool(bus)
    # Add agents of different personas
    memory = SymbolicMemory(client=None)  # type: ignore # Test mock
    pool.add_agent(name="UnitTest_Coder_1", persona="Coder", memory=memory)
    pool.add_agent(name="UnitTest_Research_1", persona="research", memory=memory)
    pool.add_agent(name="UnitTest_QA_1", persona="QA", memory=memory)
    # The agent personas should be lowercased in records
    agents = pool.get_all()
    personas = {agent["persona"] for agent in agents}
    assert personas == {"coder", "research", "qa"}

    # get_agents_by_persona should retrieve correct agent names
    coder_agents = pool.get_agents_by_persona("coder")
    assert any("Coder" in name for name in coder_agents)
    research_agents = pool.get_agents_by_persona("research")
    assert any("Research" in name for name in research_agents)
    qa_agents = pool.get_agents_by_persona("qa")
    assert any("QA" in name for name in qa_agents)

    # Persona capabilities prompt should list each persona and its valid action
    prompt = pool.get_persona_capabilities_prompt()
    assert "- 'coder': action must be 'write_code'." in prompt
    assert "- 'research': action must be 'research_topic'." in prompt
    assert "- 'qa': action must be 'review_code'." in prompt


@pytest.mark.asyncio
async def test_message_bus_publish_and_broadcast():
    """MessageBus should deliver messages to subscribed agent queues and broadcast to all."""
    bus = MessageBus()
    # Subscribe two agents (queues)
    q1 = bus.subscribe("AgentA")
    q2 = bus.subscribe("AgentB")
    # Publish a direct message to AgentA
    message = MessageModel(sender_id="Tester", receiver_id="AgentA", message_type="test", payload={"x": 1})
    await bus.publish(message)
    # AgentA's queue should have the message, AgentB's should be empty
    delivered = await q1.get()
    assert delivered.message_type == "test" and delivered.payload == {"x": 1}
    assert q1.empty()
    assert q2.empty()

    # Broadcast a message to all
    broadcast_msg = MessageModel(sender_id="Tester", receiver_id="ALL", message_type="announce", payload={"note": "hi"})
    await bus.broadcast(broadcast_msg)
    # Both AgentA and AgentB should get it (sender 'Tester' should not exclude any since not a subscribed agent here)
    delivered_a = await q1.get()
    delivered_b = await q2.get()
    assert delivered_a.message_type == "announce" and delivered_b.message_type == "announce"
    assert delivered_a.payload["note"] == "hi" and delivered_b.payload["note"] == "hi"


@pytest.mark.asyncio
async def test_planner_generates_plan_with_review(tmp_path: Path) -> None:
    """Planner.decompose_goal_into_plan should return a plan with an initial QA review step for new plans."""
    # Monkey-patch the workspace directory to a temp to avoid file side effects
    config.WORKSPACE_DIR = str(tmp_path)
    # Initialize planner with dummy introspector and real (but mostly unused) skill_manager, agent_pool, tool_plugin
    dummy_intel = DummyIntrospector()
    skill_manager = SkillManager(file_path=str(tmp_path / "skills.json"))
    agent_pool = DynamicAgentPool(MessageBus())
    tool_plugin = ToolPlugin()  # this will create the workspace dir
    planner = Planner(introspector=dummy_intel, skill_manager=skill_manager, agent_pool=agent_pool, tool_plugin=tool_plugin)  # type: ignore # Test mock

    goal_desc = "Test planning goal"
    file_manifest = ""  # assume no special files available
    result: PlannerOutput = await planner.decompose_goal_into_plan(goal_desc, file_manifest, mode='code')
    # The DummyIntrospector returns one action in the plan, so Planner should add a 'review_plan' step at the beginning
    plan_steps = result.plan
    assert plan_steps, "PlannerOutput plan should not be empty"
    first_step = plan_steps[0]
    # First step must be the QA review step
    assert first_step.action == "review_plan" and first_step.assigned_persona == "qa"
    # Its parameters should include the original goal and the plan to review (which should match the subsequent steps)
    assert "original_goal" in first_step.parameters and first_step.parameters["original_goal"] == goal_desc
    assert "plan_to_review" in first_step.parameters
    # The rest of the plan (after the QA step) should be the action(s) from the LLM's plan
    assert any(step.assigned_persona == "coder" for step in plan_steps[1:]), "Expected the original coder step in plan"


@pytest.mark.asyncio
async def test_agent_handle_message_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent.handle_message should invoke skill and send back a result message."""
    bus = MessageBus()
    # Create an agent with persona 'research' (to handle 'research_topic')
    agent_name = "TestAgent_Research_42"
    agent = Agent(name=agent_name, message_bus=bus, api_client=None)  # type: ignore # Test mock
    # Ensure orchestrator (receiver) is subscribed to get replies
    orchestrator_id = "SymbolicAGI"
    bus.subscribe(orchestrator_id)

    # Monkey-patch the agent's research skill to avoid actual LLM call
    result_payload = {"status": "success", "research_summary": "Result"}

    def dummy_research_topic(params: Dict[str, Any]) -> Dict[str, Any]:
        return result_payload
    agent.skills["research_topic"] = dummy_research_topic

    # Send a message of type 'research_topic' to the agent
    message = MessageModel(sender_id=orchestrator_id, receiver_id=agent_name, message_type="research_topic", payload={"topic": "test"})
    # Instead of running agent.run loop, directly call handle_message
    await agent.handle_message(message)
    # After handling, the agent should have published a reply to the orchestrator
    orch_queue = bus.agent_queues.get(orchestrator_id)
    assert orch_queue is not None
    reply = await orch_queue.get()
    # The reply should have the correct sender, receiver, type, and payload
    assert reply.sender_id == agent_name
    assert reply.receiver_id == orchestrator_id
    assert reply.message_type == "research_topic_result"
    assert reply.payload == result_payload

    # If agent gets a message with unknown type, it should log a warning and not send a reply
    # (We'll check that the orchestrator queue remains empty after processing an unknown message)
    unknown_msg = MessageModel(sender_id=orchestrator_id, receiver_id=agent_name, message_type="unknown_action", payload={})
    # Call handle_message for unknown type
    await agent.handle_message(unknown_msg)
    # Orchestrator queue should remain empty (no new message added)
    assert orch_queue.empty()