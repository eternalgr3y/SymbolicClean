# symbolic_agi/api_client.py

"""
Initializes and provides a shared asynchronous OpenAI client.
"""

import os
from openai import AsyncOpenAI

def get_openai_client() -> AsyncOpenAI:
    """
    Initializes and returns a singleton instance of the AsyncOpenAI client.
    
    It's recommended to use a single client instance for the application's lifetime
    as it manages underlying connection pools.
    """
    if not (api_key := os.getenv("OPENAI_API_KEY")):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    return AsyncOpenAI(api_key=api_key)

# Shared client instance
client = get_openai_client()