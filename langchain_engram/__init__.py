"""LangChain integration for Weaviate Engram agent memory.

Public surfaces:

- `EngramMiddleware`: drop-in long-term memory for `create_agent`.
- `create_memory_tools`: `search_memories` / `add_memories` tools for an agent.
- `EngramStore`: a LangGraph `BaseStore` backed by Engram (experimental).
"""

from langchain_engram._version import __version__
from langchain_engram.middleware import EngramMiddleware
from langchain_engram.store import EngramStore
from langchain_engram.tools import create_memory_tools

__all__ = [
    "EngramMiddleware",
    "EngramStore",
    "__version__",
    "create_memory_tools",
]
