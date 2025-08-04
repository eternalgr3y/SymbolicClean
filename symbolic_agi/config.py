# symbolic_agi/config.py
"""
Centralized configuration for the Symbolic AGI project.
"""

# --- LLM & Embedding Models ---
POWERFUL_MODEL = "gpt-4.1"
FAST_MODEL = "gpt-3.5-turbo" # For less critical background tasks
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# --- File Paths ---
# UPGRADE: Added all path constants to one place.
DATA_DIR = "data" # <-- NEW
FAISS_INDEX_PATH = "data/symbolic_mem.index"
SYMBOLIC_MEMORY_PATH = "data/symbolic_mem.json"
LONG_TERM_GOAL_PATH = "data/long_term_goals.json"
GOAL_ARCHIVE_PATH = "data/long_term_goals_archive.json"
IDENTITY_PROFILE_PATH = "data/identity_profile.json"
MUTATION_FILE_PATH = "data/reasoning_mutations.json"
CONSCIOUSNESS_PROFILE_PATH = "data/consciousness_profile.json"
SKILLS_PATH = "data/learned_skills.json"
WORKSPACE_DIR = "data/workspace"

# --- Behavioral & Ethical Tuning ---
# UPGRADE: Added tuning parameters.
SECONDS_OF_SILENCE_BEFORE_AUTONOMY = 20.0
PLAN_EVALUATION_THRESHOLD = 0.6
SELF_MODIFICATION_THRESHOLD = 0.99