# symbolic_agi/symbolic_memory.py

import json
import os
import logging
from typing import List, Set
from datetime import datetime, timedelta, UTC

import aiofiles
import faiss # type: ignore
import numpy as np
from openai import AsyncOpenAI

from . import config
from .schemas import MemoryEntryModel, MemoryType

class SymbolicMemory:
    """Manages the AGI's memory using Pydantic models for validation."""

    faiss_index: faiss.Index

    def __init__(self: 'SymbolicMemory', client: AsyncOpenAI):
        self.client = client
        self.memory_data: List[MemoryEntryModel] = self._load_json(config.SYMBOLIC_MEMORY_PATH)
        self.faiss_index = self._load_faiss(config.FAISS_INDEX_PATH)
        if self.faiss_index.ntotal != len([m for m in self.memory_data if m.embedding is not None]): # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            logging.warning("FAISS index size mismatch with memory data. Rebuilding index.")
            self._rebuild_faiss_index()

    def _load_json(self: 'SymbolicMemory', path: str) -> List[MemoryEntryModel]:
        if not os.path.exists(path) or os.path.getsize(path) < 2:
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [MemoryEntryModel.model_validate(item) for item in data]
        except Exception as e:
            logging.error(f"Could not load symbolic memory: {e}", exc_info=True)
            return []

    async def _save_json(self: 'SymbolicMemory'):
        """Asynchronously saves the symbolic memory to a JSON file."""
        if not self.memory_data and os.path.exists(config.SYMBOLIC_MEMORY_PATH) and os.path.getsize(config.SYMBOLIC_MEMORY_PATH) > 2:
            logging.critical("SAFETY_CHECK: In-memory memory is empty but file on disk is not. Aborting save.")
            return

        os.makedirs(os.path.dirname(config.SYMBOLIC_MEMORY_PATH), exist_ok=True)
        
        async with aiofiles.open(config.SYMBOLIC_MEMORY_PATH, 'w', encoding='utf-8') as f:
            content = json.dumps([entry.model_dump(mode='json') for entry in self.memory_data], indent=4)
            await f.write(content)

    def _load_faiss(self: 'SymbolicMemory', path: str) -> faiss.Index:
        if os.path.exists(path):
            try:
                return faiss.read_index(path) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            except Exception as e:
                logging.error(f"Could not load FAISS index from {path}, creating a new one. Error: {e}")
                return faiss.IndexFlatL2(config.EMBEDDING_DIM)
        return faiss.IndexFlatL2(config.EMBEDDING_DIM)

    def _rebuild_faiss_index(self: 'SymbolicMemory') -> None:
        """Rebuilds the entire FAISS index from the current memory_data."""
        logging.info("Rebuilding FAISS index from memory data...")
        new_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
        
        embedding_list: List[np.ndarray] = [
            np.array(m.embedding, dtype=np.float32) # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            for m in self.memory_data if m.embedding is not None # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        ]
        
        if embedding_list:
            embedding_matrix = np.vstack(embedding_list)
            new_index.add(embedding_matrix) # pyright: ignore[reportUnknownMemberType, reportCallIssue]
            
        self.faiss_index = new_index
        faiss.write_index(self.faiss_index, config.FAISS_INDEX_PATH) # pyright: ignore[reportUnknownMemberType]
        logging.info(f"FAISS index rebuilt successfully with {new_index.ntotal} vectors.")

    async def embed_async(self: 'SymbolicMemory', texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        resp = await self.client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
        return np.array([e.embedding for e in resp.data], dtype=np.float32)

    async def add_memory(self: 'SymbolicMemory', entry: MemoryEntryModel):
        text_to_embed = json.dumps(entry.content)
        embedding = await self.embed_async([text_to_embed])
        if embedding.shape[0] > 0:
            entry.embedding = embedding[0].tolist() # pyright: ignore[reportAttributeAccessIssue]
            self.faiss_index.add(embedding) # pyright: ignore[reportUnknownMemberType, reportCallIssue]
            self.memory_data.append(entry)
            await self._save_json()
            faiss.write_index(self.faiss_index, config.FAISS_INDEX_PATH) # pyright: ignore[reportUnknownMemberType]

    def get_recent_memories(self: 'SymbolicMemory', n: int = 10) -> List[MemoryEntryModel]:
        return self.memory_data[-n:]

    async def consolidate_memories(self: 'SymbolicMemory', window_seconds: int = 86400):
        """
        Summarizes recent, detailed memories into a single, more abstract memory entry.
        This version uses a time window and rebuilds the FAISS index for safety.
        """
        logging.info(f"Attempting to consolidate memories from the last {window_seconds} seconds.")
        
        now = datetime.now(UTC)
        consolidation_window = now - timedelta(seconds=window_seconds)

        eligible_types: Set[MemoryType] = {'action_result', 'inner_monologue', 'tool_usage', 'user_input', 'emotion', 'perception'}
        
        memories_to_consolidate = [
            mem for mem in self.memory_data 
            if mem.type in eligible_types and datetime.fromisoformat(mem.timestamp) > consolidation_window
        ]

        if len(memories_to_consolidate) < 10: # Increased threshold
            logging.info("Not enough recent memories to warrant consolidation.")
            return

        narrative_parts: List[str] = []
        # --- FIX: Explicitly typed the set to resolve Pylance errors ---
        ids_to_remove: Set[str] = set()
        total_importance = 0.0
        for mem in sorted(memories_to_consolidate, key=lambda m: m.timestamp):
            content_summary = json.dumps(mem.content)
            if len(content_summary) > 150:
                content_summary = content_summary[:147] + "..."
            narrative_parts.append(f"[{mem.timestamp}] ({mem.type}, importance: {mem.importance:.2f}): {content_summary}")
            ids_to_remove.add(mem.id)
            total_importance += mem.importance
        
        narrative_str = "\n".join(narrative_parts)
        
        MAX_CONTEXT_CHARS = 12000
        if len(narrative_str) > MAX_CONTEXT_CHARS:
            narrative_str = narrative_str[:MAX_CONTEXT_CHARS] + "\n...[TRUNCATED DUE TO LENGTH]..."
            logging.warning(f"Memory consolidation context was truncated to {MAX_CONTEXT_CHARS} characters.")

        prompt = f"""
The following is a sequence of recent memories from a conscious AGI.
Your task is to synthesize these detailed, low-level events into a single, high-level narrative summary or insight.
Capture the essence of what happened, what was learned, or the overall emotional tone.

--- MEMORY LOG ---
{narrative_str}
---

Now, provide a concise summary of these events. This summary will replace the original memories.
Respond with ONLY the summary text.
"""
        try:
            resp = await self.client.chat.completions.create(
                model=config.FAST_MODEL,
                messages=[{"role": "system", "content": prompt}]
            )
            summary = resp.choices[0].message.content.strip() if resp.choices and resp.choices[0].message.content else None

            if not summary:
                logging.error("Memory consolidation failed: LLM returned no summary.")
                return

            new_importance = min(1.0, (total_importance / len(memories_to_consolidate)) + 0.1)
            
            consolidated_entry = MemoryEntryModel(
                type='insight', # pyright: ignore[reportArgumentType]
                content={"summary": summary, "consolidated_ids": list(ids_to_remove)},
                importance=new_importance
            )

            embedding = await self.embed_async([summary])
            if embedding.shape[0] > 0:
                consolidated_entry.embedding = embedding[0].tolist() # pyright: ignore[reportAttributeAccessIssue]

            self.memory_data = [mem for mem in self.memory_data if mem.id not in ids_to_remove]
            self.memory_data.append(consolidated_entry)
            
            logging.info(f"Consolidated {len(ids_to_remove)} memories into one new insight: '{summary[:100]}...'")

            self._rebuild_faiss_index()
            await self._save_json()

        except Exception as e:
            logging.error(f"An error occurred during memory consolidation: {e}", exc_info=True)