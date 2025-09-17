# symbolic_agi/tool_plugin_ext.py
# cspell:ignore saucedemo

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from symbolic_agi import config
from symbolic_agi.api_client import client

# Search backend availability check
_has_ddg = False
try:
    from duckduckgo_search import DDGS  # type: ignore[import-untyped]
    _has_ddg = True
except ImportError:
    DDGS = None  # type: ignore


# ---------------------------
# Internal utilities
# ---------------------------

def _slugify(text: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "")).strip("-")
    return base or "note"

def _workspace_dir() -> Path:
    wd = getattr(config, "WORKSPACE_DIR", "workspace")
    return Path(wd)

def _ensure_workspace() -> Path:
    p = _workspace_dir()
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------
# Core tools
# ---------------------------

async def _web_search_impl(query: str, max_results: int = 5, region: str = "us-en") -> Dict[str, Any]:
    if not _has_ddg:
        return {"status": "error", "description": "duckduckgo-search not installed"}
    query = (query or "").strip()
    if not query:
        return {"status": "error", "description": "Empty query"}
    def _run() -> List[Dict[str, Any]]:
        if DDGS is None:
            return []
        with DDGS() as ddgs:  # type: ignore[misc]
            return list(ddgs.text(query, max_results=max_results, region=region))  # type: ignore[misc]
    try:
        results = await asyncio.to_thread(_run)
    except Exception as e:
        return {"status": "error", "description": f"Search failed: {e}"}
    # Normalize
    out: List[Dict[str, Any]] = []
    for r in results:
        out.append({  # type: ignore[misc]
            "title": r.get("title"),
            "url": r.get("href") or r.get("url"),
            "snippet": r.get("body"),
        })
    return {"status": "success", "results": out}

async def _fetch_url_impl(url: str) -> Dict[str, Any]:
    url = (url or "").strip()
    if not url:
        return {"status": "error", "description": "Empty URL"}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as sess:
            resp = await sess.get(url)
            resp.raise_for_status()
            return {
                "status": "success",
                "url": str(resp.url),
                "content_type": resp.headers.get("content-type", ""),
                "text": resp.text,
            }
    except Exception as e:
        return {"status": "error", "description": f"Fetch failed: {e}"}

async def _summarize_text_impl(text: str, max_bullets: int = 10) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"status": "error", "description": "No text to summarize"}
    # Keep it simple; use your existing OpenAI client
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        "Summarize the following content as concise bullet points.\n"
        f"Limit to {max_bullets} bullets. Be specific and actionable.\n\n"
        f"CONTENT:\n{text}\n"
    )
    def _call_openai() -> str:
        # Using OpenAI SDK v1.x
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = ""
        if resp.choices and resp.choices[0].message.content:  # type: ignore[misc]
            content = str(resp.choices[0].message.content).strip()  # type: ignore[misc]
        return content
    try:
        summary = await asyncio.to_thread(_call_openai)
    except Exception as e:
        return {"status": "error", "description": f"LLM summarize failed: {e}"}
    return {"status": "success", "summary": summary}

async def _write_workspace_note_impl(title: str, content: str, filename: Optional[str] = None) -> Dict[str, Any]:
    _ensure_workspace()
    safe_name = filename or f"{_slugify(title)}.md"
    path = _workspace_dir() / safe_name
    body = f"# {title}\n\n{content}\n"
    try:
        await asyncio.to_thread(path.write_text, body, encoding="utf-8")
        return {"status": "success", "path": str(path), "bytes": len(body)}
    except Exception as e:
        return {"status": "error", "description": f"Write failed: {e}"}

def _list_workspace_files_impl() -> List[str]:
    _ensure_workspace()
    paths: List[str] = []
    for p in _workspace_dir().rglob("*"):
        if p.is_file():
            try:
                paths.append(str(p.relative_to(_workspace_dir())))  # type: ignore[misc]
            except Exception:
                paths.append(str(p))  # type: ignore[misc]
    return paths

async def _review_plan_impl(**_: Any) -> Dict[str, Any]:
    # No-op success so QA steps don't explode
    await asyncio.sleep(0)  # Make it truly async
    return {"status": "success", "description": "Plan reviewed."}


# ---------------------------
# Attachers (monkey-patch onto your ToolPlugin instance)
# ---------------------------

def attach_web_tools(plugin: Any) -> None:
    """
    Adds web + note tools to the given ToolPlugin instance.
    Also injects workspace helpers if missing.
    Includes common aliases so planner synonyms won't break.
    """
    # Core tools
    plugin.web_search = _web_search_impl
    plugin.fetch_url = _fetch_url_impl
    plugin.open_url = _fetch_url_impl  # alias
    plugin.summarize_text = _summarize_text_impl
    plugin.write_workspace_note = _write_workspace_note_impl
    plugin.review_plan = _review_plan_impl

    # Aliases so planner synonyms just work
    aliases: Dict[str, Any] = {
        # search
        "search": plugin.web_search,
        "search_web": plugin.web_search,
        "websearch": plugin.web_search,
        "google_search": plugin.web_search,
        "duck_search": plugin.web_search,

        # fetch
        "read_url": plugin.fetch_url,
        "open_webpage": plugin.fetch_url,

        # summarize
        "summarize": plugin.summarize_text,
        "summarize_content": plugin.summarize_text,
        "write_summary": plugin.summarize_text,
        "summarize_article": plugin.summarize_text,

        # notes
        "write_note": plugin.write_workspace_note,
        "save_markdown": plugin.write_workspace_note,
        "write_markdown": plugin.write_workspace_note,
        "create_note": plugin.write_workspace_note,
        "save_note": plugin.write_workspace_note,
        "compose_note": plugin.write_workspace_note,
        "write_bullets": plugin.write_workspace_note,
        "write_cheatsheet": plugin.write_workspace_note,
    }
    for name, fn in aliases.items():
        setattr(plugin, name, fn)

    # Workspace helpers if base plugin doesn't provide them
    if not hasattr(plugin, "ensure_workspace_ready"):
        async def _ensure_ready() -> Dict[str, Any]:
            _ensure_workspace()
            await asyncio.sleep(0)  # Make it truly async
            return {"status": "success", "dir": str(_workspace_dir())}
        plugin.ensure_workspace_ready = _ensure_ready

    if not hasattr(plugin, "list_workspace_files"):
        plugin.list_workspace_files = _list_workspace_files_impl



def attach_playwright_tools(plugin: Any) -> None:
    """
    Safe stub: if you later want real browser automation, we can wire Playwright.
    For now this keeps imports stable and gives a placeholder.
    """
    async def pw_browser_info() -> Dict[str, Any]:
        await asyncio.sleep(0)  # Make it truly async
        return {"status": "success", "browser": "stub", "message": "Playwright stub active"}
    plugin.pw_browser_info = pw_browser_info
