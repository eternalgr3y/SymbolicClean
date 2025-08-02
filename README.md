# SymbolicAGI

## Project Overview
SymbolicAGI is a modular, self-improving AGI framework designed around symbolic reasoning, trust-based agent coordination, and continuous meta-cognitive self-improvement. 

## Quickstart (Setup & Run)
1. **Setup Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate      # On Windows: .venv\Scripts\activate
    pip install -e ".[dev]"
    pre-commit install
    ```
2. **Start Redis (required):**
    ```bash
    docker-compose up -d redis
    ```
3. **Run the AGI:**
    ```bash
    python scripts/run_agi.py
    ```

## Docs & Info
- Detailed docs in `docs/ARCHITECTURE.md`
