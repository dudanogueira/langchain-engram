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


def create_memory_tools(  # noqa: C901 - thin closures over one bound scope
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    user_id: str | UserIdResolver | None = None,
    context_key: str = "user_id",
    conversation_id: str | UserIdResolver | None = None,
    conversation_id_context_key: str = "conversation_id",
    conversation_property: str = "conversation_id",
    scope_search_to_conversation: bool = False,
    group: str | None = None,
    properties: dict[str, str] | None = None,
    topics: list[str] | None = None,
    topics_context_key: str = "topics",
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
        conversation_id: Ties each saved memory to a conversation. A string binds
            a fixed conversation; a callable `() -> str` resolves it per call. If
            `None`, the value is read from the runtime context under
            `conversation_id_context_key`. Stored under the Engram scope property
            named `conversation_property`.
        conversation_id_context_key: Runtime context key holding the conversation
            id when `conversation_id` is not supplied directly.
        conversation_property: Engram scope property name under which the
            conversation id is stored. The actual name is defined when you create
            your Engram project/topic (commonly `conversation_id` or `session_id`);
            set this to match. Defaults to `conversation_id`.
        scope_search_to_conversation: When `True`, `search_memories` is filtered
            to the current conversation. When `False` (default), the conversation
            id only tags writes and search spans all of the user's memories.
        group: Optional Engram group scope applied to reads and writes.
        properties: Optional Engram scope properties applied to reads and writes.
        topics: Default Engram topics to restrict `search_memories` to. Overridden
            at runtime by a list under `topics_context_key` in the runtime context.
        topics_context_key: Runtime context key holding a per-call list of topics
            that overrides `topics`.
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

    def resolve_conversation_id() -> str | None:
        if isinstance(conversation_id, str):
            return conversation_id
        if callable(conversation_id):
            return conversation_id()
        return _runtime_context_value(conversation_id_context_key)

    def resolve_topics() -> list[Any] | None:
        override = _runtime_context_raw(topics_context_key)
        if override:
            return [override] if isinstance(override, str) else list(override)
        return topics

    def merge_properties(*, include_conversation: bool) -> dict[str, str] | None:
        props = dict(properties or {})
        cid = resolve_conversation_id()
        if include_conversation and cid:
            props[conversation_property] = cid
        return props or None

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
            topics=resolve_topics(),
            retrieval_config=retrieval_config,
            properties=merge_properties(
                include_conversation=scope_search_to_conversation
            ),
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
            properties=merge_properties(include_conversation=True),
        )
        return f"Stored memory (run {run.run_id})."

    return [search_memories, add_memories]


def _runtime_context_raw(key: str) -> Any:
    """Best-effort raw read of a key from the active LangGraph runtime context."""
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
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _runtime_context_value(key: str) -> str | None:
    """Best-effort read of a context key coerced to a non-empty string."""
    value = _runtime_context_raw(key)
    return str(value) if value else None
