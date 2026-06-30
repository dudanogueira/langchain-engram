"""LangChain tools that let an agent read and write Engram memory explicitly.

`create_memory_tools` returns two tools an agent can call on its own:

- `search_memories` — recall facts about the user.
- `add_memories` — deliberately remember something ("remember that ...").

The `user_id` scope is fixed by the caller (or resolved from the agent runtime
context) rather than supplied by the model, so the model cannot read or write
another user's memories.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, tool

from langchain_engram._client import build_client

if TYPE_CHECKING:
    from engram import EngramClient

UserIdResolver = Callable[[], str]


def create_memory_tools(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    user_id: str | UserIdResolver | None = None,
    context_key: str = "user_id",
    group: str | None = None,
    properties: dict[str, str] | None = None,
    retrieval_config: Any | None = None,
    client: EngramClient | None = None,
) -> list[BaseTool]:
    """Create `search_memories` and `add_memories` tools bound to one scope.

    Args:
        api_key: Engram API key. Falls back to `ENGRAM_API_KEY`. Ignored when
            `client` is provided.
        base_url: Override the Engram API base URL.
        timeout: Per-request timeout in seconds.
        user_id: The memory owner. A string binds a fixed user; a callable
            `() -> str` resolves it per call. If `None`, the value is read from
            the agent runtime context under `context_key`.
        context_key: Runtime context key holding the user id when `user_id` is
            not supplied directly.
        group: Optional Engram group scope applied to reads and writes.
        properties: Optional Engram scope properties applied to reads and writes.
        retrieval_config: Optional Engram retrieval configuration for searches.
        client: A pre-built `EngramClient` to reuse instead of constructing one.

    Returns:
        A list containing the `search_memories` and `add_memories` tools.
    """
    shared_client: EngramClient | None = client

    def get_client() -> EngramClient:
        nonlocal shared_client
        if shared_client is None:
            shared_client = build_client(api_key, base_url=base_url, timeout=timeout)
        return shared_client

    def resolve_user_id() -> str:
        if isinstance(user_id, str):
            return user_id
        if callable(user_id):
            return user_id()
        resolved = _runtime_context_value(context_key)
        if not resolved:
            msg = (
                "Could not resolve an Engram user id. Pass `user_id=...` to "
                "`create_memory_tools`, or provide it in the agent runtime "
                f"context under `{context_key!r}`."
            )
            raise ValueError(msg)
        return resolved

    @tool
    def search_memories(query: str) -> str:
        """Search the user's long-term memory for information relevant to `query`.

        Use this to recall the user's preferences, past decisions, and facts they
        have shared before answering.
        """
        results = get_client().memories.search(
            query=query,
            user_id=resolve_user_id(),
            group=group,
            retrieval_config=retrieval_config,
            properties=properties,
        )
        contents = [m.content for m in results if m.content]
        if not contents:
            return "No relevant memories found."
        return "\n".join(f"- {c}" for c in contents)

    @tool
    def add_memories(text: str) -> str:
        """Save `text` to the user's long-term memory for future conversations.

        Use this when the user shares a durable preference or fact worth
        remembering (for example "remember that I'm vegetarian").
        """
        run = get_client().memories.add(
            text,
            user_id=resolve_user_id(),
            group=group,
            properties=properties,
        )
        return f"Stored memory (run {run.run_id})."

    return [search_memories, add_memories]


def _runtime_context_value(key: str) -> str | None:
    """Best-effort read of a key from the active LangGraph runtime context."""
    try:
        from langgraph.runtime import get_runtime
    except ImportError:
        return None
    try:
        runtime = get_runtime()
    except (RuntimeError, LookupError):
        return None
    context = getattr(runtime, "context", None)
    if context is None:
        return None
    value = (
        context.get(key) if isinstance(context, dict) else getattr(context, key, None)
    )
    return str(value) if value else None
