# symbolic_agi/tool_plugin.py

import asyncio
import io
import logging
import os
import requests
from bs4 import BeautifulSoup
from contextlib import redirect_stdout
from datetime import datetime, UTC
from duckduckgo_search import DDGS
from typing import Dict, Any, List, Optional

from .api_client import client
from . import config

try:
    import memory_profiler # type: ignore
except ImportError:
    memory_profiler = None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class ToolPlugin:
    """A collection of real-world tools for the AGI."""
    
    def __init__(self: 'ToolPlugin'):
        os.makedirs(config.WORKSPACE_DIR, exist_ok=True)

    def _get_safe_path(self: 'ToolPlugin', file_path: str, base_dir: str) -> str:
        base_path = os.path.abspath(base_dir)
        safe_filename = os.path.basename(file_path)
        target_path = os.path.abspath(os.path.join(base_path, safe_filename))
        
        if os.path.commonpath([base_path]) != os.path.commonpath([base_path, target_path]):
            raise PermissionError(f"File access denied: path is outside the designated directory '{base_dir}'.")
            
        return target_path

    async def analyze_data(self: 'ToolPlugin', data: str, query: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Analyzes a string of data (e.g., file content, web scrape) to answer a specific query.
        Can reference data from the workspace by passing the key name as the 'data' parameter.
        """
        logging.info(f"Analyzing data with query: '{query}'")
        
        workspace: Dict[str, Any] = kwargs.get('workspace', {})
        # --- FIX: Removed redundant isinstance check ---
        if data in workspace:
            logging.info(f"Resolving 'data' parameter from workspace key '{data}'...")
            data = str(workspace[data])

        prompt = f"""
You are a data analysis expert. Your task is to answer a specific query based ONLY on the provided data.
Do not use any external knowledge. If the answer cannot be found in the data, state that clearly.

--- DATA ---
{data}
---

--- QUERY ---
{query}

---

Respond with ONLY the direct answer to the query.
"""
        try:
            resp = await client.chat.completions.create(
                model=config.FAST_MODEL,
                messages=[{"role": "system", "content": prompt}]
            )
            if resp.choices and resp.choices[0].message.content:
                answer = resp.choices[0].message.content.strip()
                return {"status": "success", "answer": answer}
            else:
                return {"status": "failure", "error": "No answer returned from LLM."}
        except Exception as e:
            return {"status": "failure", "error": str(e)}

    async def execute_python_code(self: 'ToolPlugin', code: Optional[str] = None, timeout_seconds: int = 10, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes Python code in a restricted sandbox.
        Can reference code from the workspace by passing the key name as the 'code' parameter.
        """
        output_buffer = io.StringIO()
        try:
            workspace: Dict[str, Any] = kwargs.get('workspace', {})
            if isinstance(code, str) and code in workspace:
                code = str(workspace[code])
                logging.info(f"Resolved 'code' parameter from workspace key.")
            
            if code is None:
                return {"status": "failure", "description": "execute_python_code was called without code and no suitable code was found in the workspace."}

            logging.warning(f"Executing sandboxed Python code:\n---\n{code}\n---")
            
            safe_builtins: Dict[str, Any]
            if isinstance(__builtins__, dict):
                safe_builtins = __builtins__.copy() # pyright: ignore[reportUnknownVariableType]
            else:
                safe_builtins = __builtins__.__dict__.copy() # pyright: ignore[reportUnknownVariableType]

            restricted_functions = ['open', '__import__', 'eval', 'exit', 'quit']
            for func_name in restricted_functions:
                if func_name in safe_builtins:
                    del safe_builtins[func_name]

            safe_globals: Dict[str, Any] = {
                'memory_profiler': memory_profiler,
                'requests': requests,
                '__builtins__': safe_builtins
            }
            
            if 'import ' in code and memory_profiler is None and 'memory_profiler' in code:
                 return {"status": "failure", "description": "Error: The 'memory_profiler' library is not installed in the environment. Please install it with 'pip install memory-profiler'."}

            async def run_code() -> None:
                with redirect_stdout(output_buffer):
                    exec(str(code), safe_globals)

            await asyncio.wait_for(run_code(), timeout=timeout_seconds)
            
            output = output_buffer.getvalue()
            logging.info(f"Code execution successful. Output:\n{output}")
            return {"status": "success", "output": output}

        except asyncio.TimeoutError:
            logging.error("Code execution timed out.")
            return {"status": "failure", "description": "Error: Code execution took too long and was terminated."}
        except Exception as e:
            error_message = f"An error occurred during code execution: {type(e).__name__}: {e}"
            logging.error(error_message, exc_info=True)
            return {"status": "failure", "description": error_message, "output": output_buffer.getvalue()}

    async def read_own_source_code(self: 'ToolPlugin', file_name: str, **kwargs: Any) -> Dict[str, Any]:
        source_dir = os.path.join(PROJECT_ROOT, 'symbolic_agi')
        
        if not file_name:
            logging.info("Listing source code files.")
            try:
                files = [f for f in os.listdir(source_dir) if f.endswith('.py')]
                return {"status": "success", "files": files}
            except Exception as e:
                return {"status": "failure", "description": f"An error occurred while listing source files: {e}"}

        logging.info(f"Performing self-reflection by reading source code: {file_name}")
        try:
            safe_file_path = self._get_safe_path(file_name, source_dir)
            with open(safe_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "content": f"Source code for '{file_name}':\n\n{content}"}
        except PermissionError as e:
            return {"status": "failure", "description": str(e)}
        except FileNotFoundError:
            return {"status": "failure", "description": f"Source file '{file_name}' not found."}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred while reading source file: {e}"}

    async def read_core_file(self: 'ToolPlugin', file_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Reads a core AGI configuration file from the 'data/' directory.
        This tool has a strict whitelist of readable files for safety.
        """
        logging.info(f"Attempting to read core AGI file: {file_name}")
        
        allowed_files = [
            'consciousness_profile.json',
            'identity_profile.json',
            'long_term_goals.json',
            'learned_skills.json',
            'reasoning_mutations.json'
        ]
        
        if file_name not in allowed_files:
            return {"status": "failure", "description": f"Permission denied: '{file_name}' is not in the list of readable core files."}
            
        try:
            safe_file_path = self._get_safe_path(file_name, config.DATA_DIR)
            with open(safe_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "content": content}
        except PermissionError as e:
            return {"status": "failure", "description": str(e)}
        except FileNotFoundError:
            return {"status": "failure", "description": f"Core file '{file_name}' not found in '{config.DATA_DIR}'. You might need to use 'list_files' on the workspace instead."}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred while reading core file: {e}"}

    async def list_files(self: 'ToolPlugin', directory: str = ".", **kwargs: Any) -> Dict[str, Any]:
        try:
            base_path = os.path.abspath(config.WORKSPACE_DIR)
            target_path = os.path.abspath(os.path.join(base_path, directory))
            if os.path.commonpath([base_path]) != os.path.commonpath([base_path, target_path]):
                 raise PermissionError("File access denied: path is outside the workspace directory.")
            
            files = os.listdir(target_path)
            return {"status": "success", "files": files}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred: {e}"}

    async def write_file(self: 'ToolPlugin', file_path: str, content: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        try:
            workspace: Dict[str, Any] = kwargs.get('workspace', {})
            if isinstance(content, str) and content in workspace:
                content = str(workspace[content])
                logging.info(f"Resolved 'content' parameter from workspace key '{content[:20]}...'")

            if content is None:
                return {"status": "failure", "description": "write_file was called without content and no suitable content was found in the workspace."}

            safe_file_path = self._get_safe_path(file_path, config.WORKSPACE_DIR)
            with open(safe_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"status": "success", "description": f"Successfully wrote to '{file_path}'."}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred while writing file: {e}"}

    async def read_file(self: 'ToolPlugin', file_path: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            safe_file_path = self._get_safe_path(file_path, config.WORKSPACE_DIR)
            with open(safe_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"status": "success", "content": content}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred while reading file: {e}"}

    async def analyze_image(self: 'ToolPlugin', image_url: str, prompt: str = "Describe this image in detail.", **kwargs: Any) -> Dict[str, Any]:
        logging.info(f"Analyzing image from URL: {image_url}")
        try:
            response = await client.chat.completions.create(
                model=config.POWERFUL_MODEL,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url,},},],}],
                max_tokens=500, timeout=45.0
            )
            if response.choices and response.choices[0].message.content:
                return {"status": "success", "description": response.choices[0].message.content}
            return {"status": "failure", "description": "Image analysis returned no content."}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred during image analysis: {e}"}

    async def browse_webpage(self: 'ToolPlugin', url: str, **kwargs: Any) -> Dict[str, Any]:
        logging.info(f"Browse webpage: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = '\n'.join(chunk for chunk in (phrase.strip() for line in soup.get_text().splitlines() for phrase in line.split("  ")) if chunk)
            return {"status": "success", "content": text[:8000]} if text else {"status": "failure", "description": "Could not extract text."}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred while Browse webpage: {e}"}

    async def web_search(self: 'ToolPlugin', query: str, num_results: int = 3, **kwargs: Any) -> Dict[str, Any]:
        logging.info(f"Executing REAL web search for query: '{query}'")
        try:
            with DDGS() as ddgs:
                results: List[Dict[str, str]] = [r for r in ddgs.text(query, max_results=num_results)]
            return {"status": "success", "data": "\n\n".join([f"Title: {res['title']}\nSnippet: {res['body']}\nURL: {res['href']}" for res in results])} if results else {"status": "success", "data": "No results found."}
        except Exception as e:
            return {"status": "failure", "description": f"An error occurred during web search: {e}"}

    async def get_current_datetime(self: 'ToolPlugin', _timezone_str: str = "UTC", **kwargs: Any) -> Dict[str, Any]:
        try:
            return {"status": "success", "data": datetime.now(UTC).isoformat()}
        except Exception as e:
            return {"status": "failure", "description": f"Could not get current time: {e}"}

    async def execute_tool(self: 'ToolPlugin', tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(self, tool_name):
            tool_method = getattr(self, tool_name)
            return await tool_method(**parameters)
        else:
            return {"status": "failure", "description": f"Error: Tool '{tool_name}' not found."}