# scripts/run_agi.py

import asyncio
import logging
import traceback
from typing import List

from symbolic_agi.agent import Agent
from symbolic_agi.schemas import AGIConfig, GoalModel
from symbolic_agi.agi_controller import SymbolicAGI
from symbolic_agi.api_client import client

def setup_logging():
    """Sets up a detailed logger with UTF-8 encoding."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - (%(filename)s:%(lineno)d) - %(message)s",
        handlers=[
            logging.FileHandler("agi_session.log", mode='w', encoding='utf-8'),
        ]
    )

async def main():
    """Initializes the AGI and runs the main interactive/autonomous loop."""
    agi = None
    agent_tasks: List[asyncio.Task[None]] = []
    main_loop_task = None
    try:
        setup_logging()

        print("="*50)
        print("--- INITIALIZING SYMBOLIC AGI SYSTEM (PERSISTENT MODE) ---")
        print("--- All logs will be written to agi_session.log ---")
        print("="*50)
        
        agi = SymbolicAGI(cfg=AGIConfig())
        
        await agi.startup_validation()
        
        await agi.start_background_tasks()

        specialist_definitions = [
            {"name": f"{agi.name}_Coder_0", "persona": "coder"},
            {"name": f"{agi.name}_Research_0", "persona": "research"},
            {"name": f"{agi.name}_QA_0", "persona": "qa"},
        ]

        for agent_def in specialist_definitions:
            agi.agent_pool.add_agent(
                name=agent_def["name"], 
                persona=agent_def["persona"], 
                memory=agi.memory
            )
            specialist_agent = Agent(name=agent_def["name"], message_bus=agi.message_bus, api_client=client)
            task = asyncio.create_task(specialist_agent.run())
            agent_tasks.append(task)
        
        print(f"--- {len(agent_tasks)} SPECIALIST AGENTS ONLINE ---")
        print("--- AGI CORE ONLINE. NOW FULLY AUTONOMOUS & PERSISTENT. ---")
        print("You can enter a goal at any time. If idle, the AGI may generate its own.")
        print("="*50 + "\n")

        async def autonomous_loop():
            """The main cognitive heartbeat of the AGI."""
            while True:
                try:
                    if active_goal := agi.ltm.get_active_goal():
                        print(f"\nAGI is now working on goal: '{active_goal.description}'...")
                        
                        result = await agi.handle_autonomous_cycle()
                        
                        status_message = result.get("description", "Cycle finished.")
                        
                        print(f"AGI status: {status_message}")

                        if response_text := result.get("response_text"):
                            print(f"\nAGI: {response_text}\n")
                    else:
                        pass
                    
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    print("Autonomous loop cancelled.")
                    break
                except Exception as e:
                    print(f"\nCRITICAL ERROR in autonomous loop: {e}\nCheck agi_session.log for details.")
                    logging.error(f"Error in autonomous loop: {e}", exc_info=True)
                    await asyncio.sleep(15)

        main_loop_task = asyncio.create_task(autonomous_loop())

        while True:
            user_input = await asyncio.to_thread(input, "Enter a new goal (or press Ctrl+C to exit):\n> ")
            if user_input.strip():
                # --- UPGRADE: Intelligently set the goal mode ---
                goal_mode = 'docs' if 'document' in user_input.lower() else 'code'
                new_goal = GoalModel(
                    description=user_input.strip(), 
                    sub_tasks=[],
                    mode=goal_mode
                )
                agi.ltm.add_goal(new_goal)
                print(f"Goal '{new_goal.description}' has been added to the queue (Mode: {goal_mode}).")
            else:
                print("No input received. AGI continues autonomous operation.")

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n--- User initiated shutdown (Ctrl+C). ---")
        logging.info("\n--- User initiated shutdown (Ctrl+C). ---")
    except Exception:
        logging.error("A critical error occurred in the main runner:")
        traceback.print_exc()
    finally:
        if main_loop_task:
            main_loop_task.cancel()
        
        if agi:
            await agi.shutdown()
        
        for task in agent_tasks:
            task.cancel()
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        print("All agents have been shut down.")
        logging.info("All agents have been shut down.")

if __name__ == "__main__":
    asyncio.run(main())