"""Agent middleware that wires Weaviate Engram memory into `create_agent`.

`EngramMiddleware` gives a LangChain agent long-term memory with no changes to
the agent body:

- **Recall** (`wrap_model_call`): before each model call it searches Engram for
  memories relevant to the latest user turn and injects them into the system
  prompt.
- **Write** (`after_agent`): after the agent finishes it hands the conversation
  to Engram, whose asynchronous pipeline extracts, dedupes, and persists facts.

Memory failures never interrupt the agent — they are logged and swallowed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    convert_to_openai_messages,
)

from langchain_engram._client import build_async_client, build_client

if TYPE_CHECKING:
    from engram import AsyncEngramClient, EngramClient
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

UserIdResolver = Callable[[Any], str]

_DEFAULT_MEMORY_HEADER = (
    "Here are memories that may be relevant to the user's request. "
    "Use them when helpful, but do not mention them unless asked:"
)


class EngramMiddleware(AgentMiddleware):
    """Add Weaviate Engram long-term memory to a `create_agent` agent.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_engram import EngramMiddleware

        agent = create_agent(
            "anthropic:claude-sonnet-4-6",
            middleware=[EngramMiddleware(user_id="alice@example.com")],
        )
        agent.invoke({"messages": [{"role": "user", "content": "Use metric units."}]})
        ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        user_id: str | UserIdResolver | None = None,
        context_key: str = "user_id",
        conversation_id: str | UserIdResolver | None = None,
        conversation_id_context_key: str = "conversation_id",
        conversation_property: str = "conversation_id",
        scope_recall_to_conversation: bool = False,
        group: str | None = None,
        properties: dict[str, str] | None = None,
        topics: list[str] | None = None,
        topics_context_key: str = "topics",
        retrieval_config: Any | None = None,
        memory_header: str = _DEFAULT_MEMORY_HEADER,
        recall: bool = True,
        write: bool = True,
    ) -> None:
        """Initialize the middleware.

        Args:
            api_key: Engram API key. Falls back to `ENGRAM_API_KEY`.
            base_url: Override the Engram API base URL.
            timeout: Per-request timeout in seconds.
            user_id: The memory owner. A string binds a fixed user; a callable
                `(runtime) -> str` resolves it per invocation. If `None`, the
                value is read from the agent runtime context under `context_key`.
            context_key: Runtime context key holding the user id when `user_id`
                is not supplied directly.
            conversation_id: Ties each written memory to a conversation. A string
                binds a fixed conversation; a callable `(runtime) -> str` resolves
                it per invocation. If `None`, the value is read from the runtime
                context under `conversation_id_context_key` (so it can be passed at
                runtime via `agent.invoke(..., context={"conversation_id": ...})`).
                Stored under the Engram scope property named `conversation_property`.
            conversation_id_context_key: Runtime context key holding the
                conversation id when `conversation_id` is not supplied directly.
            conversation_property: Engram scope property name under which the
                conversation id is stored. The actual name is defined when you
                create your Engram project/topic (commonly `conversation_id` or
                `session_id`); set this to match. Defaults to `conversation_id`.
            scope_recall_to_conversation: When `True`, the `conversation_id` is
                added to the recall search filter. When `False` (default), the
                conversation id only tags writes and recall spans all of the
                user's memories (cross-conversation long-term memory). Note that
                this property is a filter, not a hard isolation boundary: true
                per-conversation isolation (and conversation-scoped reconciliation)
                requires the target Engram topic to be `scoped by` that property;
                otherwise reconciliation is user-level and unscoped memories surface
                in every conversation.
            group: Optional Engram group scope applied to reads and writes.
            properties: Optional Engram scope properties applied to reads and
                writes (for example `{"app": "support"}`).
            topics: Default Engram topics to restrict search to. Overridden at
                runtime by a list under `topics_context_key` in the runtime
                context.
            topics_context_key: Runtime context key holding a per-invocation list
                of topics that overrides `topics`.
            retrieval_config: Optional Engram retrieval configuration passed to
                search — a named type (`"vector"`, `"bm25"`, `"hybrid"`,
                `"fetch"`) or a retrieval model such as `HybridRetrieval(limit=5)`.
            memory_header: Sentence prepended to recalled memories in the prompt.
            recall: Whether to inject relevant memories before each model call.
            write: Whether to persist the conversation when the agent finishes.
        """
        super().__init__()
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._user_id = user_id
        self._context_key = context_key
        self._conversation_id = conversation_id
        self._conversation_id_context_key = conversation_id_context_key
        self._conversation_property = conversation_property
        self._scope_recall_to_conversation = scope_recall_to_conversation
        self._group = group
        self._properties = properties
        self._topics: list[Any] | None = topics
        self._topics_context_key = topics_context_key
        self._retrieval_config = retrieval_config
        self._memory_header = memory_header
        self._recall = recall
        self._write = write
        self._client: EngramClient | None = None
        self._async_client: AsyncEngramClient | None = None

    # -- client accessors ---------------------------------------------------

    def _sync_client(self) -> EngramClient:
        if self._client is None:
            self._client = build_client(
                self._api_key, base_url=self._base_url, timeout=self._timeout
            )
        return self._client

    def _get_async_client(self) -> AsyncEngramClient:
        if self._async_client is None:
            self._async_client = build_async_client(
                self._api_key, base_url=self._base_url, timeout=self._timeout
            )
        return self._async_client

    # -- recall -------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Inject relevant memories into the request, then call the model."""
        if not self._recall:
            return handler(request)
        try:
            user_id = self._resolve_user_id(request.runtime)
            query = _latest_user_text(request)
            if query:
                results = self._sync_client().memories.search(
                    query=query,
                    user_id=user_id,
                    group=self._group,
                    topics=self._resolve_topics(request.runtime),
                    retrieval_config=self._retrieval_config,
                    properties=self._recall_properties(request.runtime),
                )
                request = self._with_memories(request, [m.content for m in results])
        except Exception:
            logger.exception("Engram recall failed; continuing without memories")
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Async variant of `wrap_model_call`."""
        if not self._recall:
            return await handler(request)
        try:
            user_id = self._resolve_user_id(request.runtime)
            query = _latest_user_text(request)
            if query:
                results = await self._get_async_client().memories.search(
                    query=query,
                    user_id=user_id,
                    group=self._group,
                    topics=self._resolve_topics(request.runtime),
                    retrieval_config=self._retrieval_config,
                    properties=self._recall_properties(request.runtime),
                )
                request = self._with_memories(request, [m.content for m in results])
        except Exception:
            logger.exception("Engram recall failed; continuing without memories")
        return await handler(request)

    # -- write --------------------------------------------------------------

    def after_agent(self, state: Any, runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Persist the finished conversation to Engram (fire-and-forget)."""
        if not self._write:
            return None
        try:
            conversation = _conversation_input(state)
            if conversation:
                self._sync_client().memories.add(
                    conversation,
                    user_id=self._resolve_user_id(runtime),
                    group=self._group,
                    properties=self._write_properties(runtime),
                )
        except Exception:
            logger.exception("Engram write failed; conversation not persisted")
        return None

    async def aafter_agent(
        self, state: Any, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async variant of `after_agent`."""
        if not self._write:
            return None
        try:
            conversation = _conversation_input(state)
            if conversation:
                await self._get_async_client().memories.add(
                    conversation,
                    user_id=self._resolve_user_id(runtime),
                    group=self._group,
                    properties=self._write_properties(runtime),
                )
        except Exception:
            logger.exception("Engram write failed; conversation not persisted")
        return None

    # -- helpers ------------------------------------------------------------

    def _resolve_user_id(self, runtime: Runtime[Any] | None) -> str:
        if isinstance(self._user_id, str):
            return self._user_id
        if callable(self._user_id):
            return self._user_id(runtime)
        context = getattr(runtime, "context", None)
        resolved = _context_value(context, self._context_key)
        if not resolved:
            msg = (
                "Could not resolve an Engram user id. Pass `user_id=...` to "
                "`EngramMiddleware`, or provide it in the agent runtime context "
                f"under `{self._context_key!r}`."
            )
            raise ValueError(msg)
        return resolved

    def _resolve_conversation_id(self, runtime: Runtime[Any] | None) -> str | None:
        value = self._conversation_id
        if isinstance(value, str):
            return value
        if callable(value):
            return value(runtime)
        return _context_value(
            getattr(runtime, "context", None), self._conversation_id_context_key
        )

    def _resolve_topics(self, runtime: Runtime[Any] | None) -> list[Any] | None:
        override = _context_raw(
            getattr(runtime, "context", None), self._topics_context_key
        )
        if override:
            return [override] if isinstance(override, str) else list(override)
        return self._topics

    def _merge_properties(
        self, conversation_id: str | None, *, include_conversation: bool
    ) -> dict[str, str] | None:
        props = dict(self._properties or {})
        if include_conversation and conversation_id:
            props[self._conversation_property] = conversation_id
        return props or None

    def _recall_properties(self, runtime: Runtime[Any] | None) -> dict[str, str] | None:
        conversation_id = self._resolve_conversation_id(runtime)
        return self._merge_properties(
            conversation_id, include_conversation=self._scope_recall_to_conversation
        )

    def _write_properties(self, runtime: Runtime[Any] | None) -> dict[str, str] | None:
        conversation_id = self._resolve_conversation_id(runtime)
        return self._merge_properties(conversation_id, include_conversation=True)

    def _with_memories(
        self, request: ModelRequest, memories: list[str]
    ) -> ModelRequest:
        contents = [m for m in memories if m]
        if not contents:
            return request
        block = self._memory_header + "\n" + "\n".join(f"- {m}" for m in contents)
        existing = request.system_message
        if existing is not None:
            combined = _content_text(existing.content) + "\n\n" + block
        else:
            combined = block
        return request.override(system_message=SystemMessage(content=combined))


def _latest_user_text(request: ModelRequest) -> str | None:
    """Return the text of the most recent human message, if any."""
    for message in reversed(request.messages):
        if isinstance(message, HumanMessage):
            text = _content_text(message.content)
            if text:
                return text
    return None


def _conversation_input(state: Any) -> list[dict[str, str]]:
    """Convert agent state messages into Engram conversation input."""
    messages = (
        state.get("messages")
        if isinstance(state, dict)
        else getattr(state, "messages", None)
    )
    if not messages:
        return []
    conversation: list[dict[str, str]] = []
    for message in convert_to_openai_messages(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = _content_text(message.get("content"))
        if role in {"user", "assistant", "system"} and content:
            conversation.append({"role": role, "content": content})
    return conversation


def _content_text(content: Any) -> str:
    """Flatten LangChain/OpenAI message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _context_raw(context: Any, key: str) -> Any:
    """Read a raw key from a runtime context that may be a mapping or object."""
    if context is None:
        return None
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _context_value(context: Any, key: str) -> str | None:
    """Read a key from a runtime context and coerce it to a non-empty string."""
    value = _context_raw(context, key)
    return str(value) if value else None
