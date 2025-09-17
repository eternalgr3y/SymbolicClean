# scripts/run_agi.py

import asyncio
import logging

from symbolic_agi.agi_controller import SymbolicAGI
from symbolic_agi.schemas import AGIConfig

LOG_FILE = "agi_session.log"

async def main() -> None:
    # Pretty banner
    print("==================================================")
    print("--- INITIALIZING SYMBOLIC AGI SYSTEM (PERSISTENT MODE) ---")
    print(f"--- All logs will be written to {LOG_FILE} ---")
    print("==================================================")

    # Logging to file + console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )

    agi = SymbolicAGI(cfg=AGIConfig())

    # async init
    await agi.startup_validation()

    # NOTE: start_background_tasks is sync on purpose — do NOT await it
    agi.start_background_tasks()

    # Simple heartbeat loop
    try:
        while True:
            result = await agi.handle_autonomous_cycle()
            if (desc := result.get("description") or result.get("response_text") or ""):
                print(f"> {desc}")
            await asyncio.sleep(5)  # pacing
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: shutting down…")
    finally:
        await agi.shutdown()
        print("All agents have been shut down.")

if __name__ == "__main__":
    asyncio.run(main())
