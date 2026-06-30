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
        group: str | None = None,
        properties: dict[str, str] | None = None,
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
            group: Optional Engram group scope applied to reads and writes.
            properties: Optional Engram scope properties applied to reads and
                writes (for example `{"app": "support"}`).
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
        self._group = group
        self._properties = properties
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
                    retrieval_config=self._retrieval_config,
                    properties=self._properties,
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
                    retrieval_config=self._retrieval_config,
                    properties=self._properties,
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
                    properties=self._properties,
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
                    properties=self._properties,
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


def _context_value(context: Any, key: str) -> str | None:
    """Read a key from a runtime context that may be a mapping or object."""
    if context is None:
        return None
    if isinstance(context, dict):
        value = context.get(key)
    else:
        value = getattr(context, key, None)
    return str(value) if value else None
