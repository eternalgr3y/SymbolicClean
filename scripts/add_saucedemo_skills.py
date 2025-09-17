# scripts/add_saucedemo_skills.py

from symbolic_agi.skill_manager import SkillManager
from symbolic_agi.schemas import ActionStep

# Constants
SAUCE_STATE_FILE = "sauce_state.json"

if __name__ == "__main__":
    sm = SkillManager()

    plan = [
        ActionStep(
            action="login_site_with_playwright",
            parameters={
                "url": "https://www.saucedemo.com/",
                "username": "standard_user",
                "password": "secret_sauce",
                "username_selector": "#user-name",
                "password_selector": "#password",
                "submit_selector": "#login-button",
                "wait_for_selector": "#inventory_container",
                "headless": True,
                "screenshot_name": "sauce_login.png",
                "storage_state_name": SAUCE_STATE_FILE,
            },
            assigned_persona="orchestrator",
        ),
        ActionStep(
            action="open_with_storage_state_and_screenshot",
            parameters={
                "url": "https://www.saucedemo.com/inventory.html",
                "storage_state_name": SAUCE_STATE_FILE,
                "screenshot_name": "sauce_inventory.png",
                "wait_for": "#inventory_container",
                "headless": True,
            },
            assigned_persona="orchestrator",
        ),
        ActionStep(
            action="scrape_saucedemo_inventory_to_csv",
            parameters={
                "storage_state_name": SAUCE_STATE_FILE,
                "csv_name": "sauce_inventory.csv",
                "headless": True,
            },
            assigned_persona="orchestrator",
        ),
    ]

    sm.add_new_skill(
        name="SauceDemoLoginAndScrape",
        description="Login to SauceDemo, open inventory, scrape to CSV.",
        plan=plan,
    )
    print("Skill 'SauceDemoLoginAndScrape' added.")
