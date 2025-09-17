# micro_world.py
import asyncio
import logging
import random
import json
from typing import Dict, Any, List, Optional

# Moved from _action_ask for better practice
from .api_client import client

# Constants
AGENT_OR_OBJECT_NOT_FOUND = "Agent or object not found."

class MicroWorld:
    """A rich, multi-agent, multi-room simulated world."""

    # Declare types for instance attributes at the class level
    room_map: Dict[str, Dict[str, Any]]
    state: Dict[str, List[Any]]

    def __init__(self: 'MicroWorld'):
        self.room_map = {
            "hallway": {"desc": "A main hallway with doors to room1 and room2.", "exits": ["room1", "room2"]},
            "room1":   {"desc": "A small room with a locked chest and sticks.", "exits": ["hallway"]},
            "room2":   {"desc": "A stone room with a heavy rock and a notice board.", "exits": ["hallway"]},
        }
        # The type for 'state' is now less restrictive to allow for different list types
        self.state = {
            "agents": [
                {"name": "SymbolicAGI", "location": "hallway", "inventory": []},
                {"name": "Alice", "location": "room1", "inventory": ["Stick"]},
                {"name": "Bob", "location": "room2", "inventory": ["Rock"]},
                {"name": "User", "location": "hallway", "inventory": []}
            ],
            "objects": [
                {"name": "Chest", "location": "room1", "state": "locked", "description": "A heavy wooden chest with a lock."},
                {"name": "Rock", "location": "room2", "description": "A rough gray rock."},
                {"name": "Stick", "location": "room1", "description": "A sturdy wooden stick."},
                {"name": "Key", "location": "room2", "description": "A small iron key."},
                {"name": "NoticeBoard", "location": "room2", "description": "A faded notice board covered with old papers."}
            ],
            "doors": [
                {"from": "hallway", "to": "room1", "locked": False},
                {"from": "hallway", "to": "room2", "locked": False}
            ],
            "rooms": list(self.room_map.keys()), # This assignment is now valid
            "events": []
        }

    def add_agent(self: 'MicroWorld', name: str, location: str = "hallway", inventory: Optional[List[str]] = None):
        if inventory is None:
            inventory = []
        self.state["agents"].append({"name": name, "location": location, "inventory": inventory})

    def get_agent(self: 'MicroWorld', name: str) -> Optional[Dict[str, Any]]:
        return next((agent for agent in self.state["agents"] if agent["name"] == name), None)

    def get_object(self: 'MicroWorld', object_name: str) -> Optional[Dict[str, Any]]:
        return next((obj for obj in self.state["objects"] if obj["name"] == object_name), None)

    def room_agents(self: 'MicroWorld', room: str) -> List[Dict[str, Any]]:
        return [a for a in self.state["agents"] if a["location"] == room]

    def room_objects(self: 'MicroWorld', room: str) -> List[Dict[str, Any]]:
        return [o for o in self.state["objects"] if o["location"] == room]

    def tick(self: 'MicroWorld'):
        """Simulate time passing in the world (random agent wandering)."""
        try:
            if random.random() < 0.1:
                agent = random.choice(self.state["agents"])
                if (available_exits := self.room_map[agent["location"]]["exits"]):
                    new_location = random.choice(available_exits)
                    agent["location"] = new_location
                    self.state["events"].append({"event_type": "wander", "agent": agent["name"], "to": new_location})
        except Exception as e:
            logging.error(f"World tick error: {e}")

    async def perform_action(self: 'MicroWorld', action: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            method_to_call = getattr(self, f"_action_{action}")
            if asyncio.iscoroutinefunction(method_to_call):
                result = await method_to_call(**kwargs)
            else:
                result = method_to_call(**kwargs)
            self.state["events"].append({"action": action, "params": kwargs, "result": result})
            logging.info(f"WORLD ACTION: {action} with {kwargs} -> {result}")
            return result
        except Exception as e:
            err_msg = f"Error performing action '{action}': {e}"
            logging.error(err_msg)
            return {"status": "failure", "description": err_msg}

    # ========== Actions ==========

    def _action_move(self: 'MicroWorld', agent_name: str, new_location: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        if not agent:
            return {"status": "failure", "description": f"Agent '{agent_name}' not found."}
        if new_location not in self.room_map[agent["location"]]["exits"]:
            return {"status": "failure", "description": f"Cannot move from {agent['location']} to {new_location}."}
        if any(d for d in self.state["doors"] if d["from"] == agent["location"] and d["to"] == new_location and d["locked"]):
            return {"status": "failure", "description": f"The door to {new_location} is locked."}
        agent["location"] = new_location
        return {"status": "success", "description": f"{agent_name} moves to {new_location}."}

    def _action_read(self: 'MicroWorld', object_name: str, agent_name: str = "SymbolicAGI") -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        obj = self.get_object(object_name)
        if not agent:
            return {"status": "failure", "description": f"Agent '{agent_name}' not found."}
        if not obj:
            return {"status": "failure", "description": f"Object '{object_name}' not found."}
        if agent["location"] != obj.get("location"):
            return {"status": "failure", "description": f"{agent_name} is not in the same location as {object_name}."}
        desc = obj.get("description", f"You see a {object_name}.")
        details = f"State: {obj.get('state', 'normal')}" if "state" in obj else ""
        return {"status": "success", "description": f"{agent_name} reads {object_name}: {desc} {details}".strip()}

    def _action_pickup(self: 'MicroWorld', agent_name: str, object_name: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        obj = self.get_object(object_name)
        if not agent or not obj:
            return {"status": "failure", "description": AGENT_OR_OBJECT_NOT_FOUND}
        if agent["location"] != obj.get("location"):
            return {"status": "failure", "description": f"{object_name} is not in the same room as {agent_name}."}
        agent["inventory"].append(object_name)
        obj["location"] = "inventory"
        return {"status": "success", "description": f"{agent_name} picked up {object_name}."}

    def _action_drop(self: 'MicroWorld', agent_name: str, object_name: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        obj = self.get_object(object_name)
        if not agent or not obj:
            return {"status": "failure", "description": AGENT_OR_OBJECT_NOT_FOUND}
        if object_name not in agent["inventory"]:
            return {"status": "failure", "description": f"{agent_name} does not have {object_name}."}
        agent["inventory"].remove(object_name)
        obj["location"] = agent["location"]
        return {"status": "success", "description": f"{agent_name} dropped {object_name} in {agent['location']}."}

    def _action_open(self: 'MicroWorld', agent_name: str, object_name: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        obj = self.get_object(object_name)
        if not agent or not obj:
            return {"status": "failure", "description": AGENT_OR_OBJECT_NOT_FOUND}
        if object_name == "Chest" and obj.get("state") == "locked":
            if "Key" not in agent["inventory"]:
                return {"status": "failure", "description": "The Chest is locked. You need a Key."}
            obj["state"] = "unlocked"
            return {"status": "success", "description": "Unlocked the Chest with the Key!"}
        return {"status": "failure", "description": f"{object_name} cannot be opened or is already open."}

    async def _action_ask(self: 'MicroWorld', asking_agent: str, target_agent: str, question: str) -> Dict[str, Any]:
        agent = self.get_agent(asking_agent)
        target = self.get_agent(target_agent)
        if not agent or not target:
            return {"status": "failure", "description": "Agent or target not found."}
        if agent["location"] != target["location"]:
            return {"status": "failure", "description": f"{target_agent} is not in the same location as {asking_agent}."}
        if target_agent.lower() == "user":
            return {"status": "success", "response_text": f"{asking_agent} asked you: {question}"}
        try:
            prompt = (
                f"{target_agent} is being asked a question by {asking_agent}.\n"
                f"World state: {json.dumps(self.state)}\n"
                f"Question: {question}\n"
                f"Answer in character as {target_agent} and keep it concise:"
            )
            resp = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )
            answer = resp.choices[0].message.content.strip() if resp.choices and resp.choices[0].message.content else "..."
        except Exception as e:
            logging.error(f"LLM ask error: {e}")
            answer = f"{target_agent} says: I don't know yet, but I'll try to help next time!"
        return {
            "status": "success",
            "response_text": f"{asking_agent} asked {target_agent}: {question}\n{answer}"
        }

    def _action_look(self: 'MicroWorld', agent_name: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        if not agent:
            return {"status": "failure", "description": f"Agent '{agent_name}' not found."}
        location = agent["location"]
        room = self.room_map[location]
        objects_here = [obj["name"] for obj in self.room_objects(location)]
        agents_here = [a["name"] for a in self.room_agents(location) if a["name"] != agent_name]
        return {
            "status": "success",
            "description": f"You are in {location}. {room['desc']} You see: {', '.join(objects_here) or 'nothing'}. "
                           f"Others here: {', '.join(agents_here) or 'no one'}."
        }

    def _action_give(self: 'MicroWorld', giving_agent: str, item_name: str, receiving_agent: str) -> Dict[str, Any]:
        agent = self.get_agent(giving_agent)
        recipient = self.get_agent(receiving_agent)
        obj = self.get_object(item_name)
        if not agent or not recipient or not obj:
            return {"status": "failure", "description": "Agent, recipient, or object not found."}
        if agent["location"] != recipient["location"]:
            return {"status": "failure", "description": "Recipient not in the same room."}
        if item_name not in agent["inventory"]:
            return {"status": "failure", "description": f"{giving_agent} does not have {item_name}."}
        agent["inventory"].remove(item_name)
        recipient["inventory"].append(item_name)
        obj["location"] = "inventory" 
        return {"status": "success", "description": f"{giving_agent} gave {item_name} to {receiving_agent}."}


    def _action_combine(self: 'MicroWorld', agent_name: str, item1_name: str, item2_name: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        if not agent:
            return {"status": "failure", "description": "Agent not found."}
        if item1_name not in agent["inventory"] or item2_name not in agent["inventory"]:
            return {"status": "failure", "description": "Agent does not have both items to combine."}
        if {item1_name, item2_name} == {"Stick", "Rock"}:
            agent["inventory"].remove("Stick")
            agent["inventory"].remove("Rock")
            agent["inventory"].append("Hammer")
            return {"status": "success", "description": f"{agent_name} crafted a Hammer."}
        return {"status": "failure", "description": "These items cannot be combined."}

    def _action_use(self: 'MicroWorld', agent_name: str, item_name: str, target_name: str) -> Dict[str, Any]:
        agent = self.get_agent(agent_name)
        if not agent:
            return {"status": "failure", "description": "Agent not found."}
        if item_name not in agent["inventory"]:
            return {"status": "failure", "description": f"Agent does not have a {item_name}."}
        target = self.get_object(target_name)
        if not target:
            return {"status": "failure", "description": f"Target object {target_name} not found."}
        if item_name == "Hammer" and target_name == "Chest" and target["state"] == "locked":
            target["state"] = "unlocked"
            return {"status": "success", "description": f"{agent_name} used the Hammer to break the lock on the Chest."}
        return {"status": "failure", "description": f"The {item_name} has no effect on the {target_name}."}