"""A LangGraph `BaseStore` backed by Weaviate Engram.

`EngramStore` lets existing LangGraph store-based code — including
`create_agent(store=...)` and LangMem — target Engram:

```python
from langchain.agents import create_agent
from langchain_engram import EngramStore

agent = create_agent("anthropic:claude-sonnet-4-6", store=EngramStore())
```

!!! warning "Experimental"
    Engram is a memory-*extraction* service, not an exact key-value store. A
    `put` is handed to Engram's asynchronous pipeline, which extracts, dedupes,
    and assigns its own ids — so the `key` you pass to `put` is not the key you
    read back. Use `search` to retrieve memories; the `key` on each returned
    item is the Engram memory id, which you can then pass to `get` or `delete`.
    `list_namespaces` is not supported and returns an empty list.

Namespace mapping: `namespace[0]` is the Engram `user_id`; an optional
`namespace[1]` is the Engram `group`. Deeper namespaces are rejected.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from engram.errors import EngramError
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from langchain_engram._client import build_async_client, build_client

if TYPE_CHECKING:
    from collections.abc import Iterable

    from engram import AsyncEngramClient, EngramClient
    from engram._models import Memory

_TEXT_KEYS = ("content", "text", "data", "value")


class EngramStore(BaseStore):
    """A LangGraph `BaseStore` implementation backed by Engram. Experimental."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        retrieval_config: Any | None = None,
        properties: dict[str, str] | None = None,
        conversation_id: str | None = None,
        conversation_property: str = "conversation_id",
        scope_search_to_conversation: bool = False,
        topics: list[str] | None = None,
        client: EngramClient | None = None,
        async_client: AsyncEngramClient | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            api_key: Engram API key. Falls back to `ENGRAM_API_KEY`.
            base_url: Override the Engram API base URL.
            timeout: Per-request timeout in seconds.
            retrieval_config: Default Engram retrieval configuration for searches
                (named type or retrieval model). When omitted, vector retrieval
                using the search `limit` is used.
            properties: Default Engram scope properties applied to every call.
            conversation_id: Ties every written memory to a conversation, stored
                under the Engram scope property named `conversation_property`.
            conversation_property: Engram scope property name under which the
                conversation id is stored. The actual name is defined when you
                create your Engram project/topic (commonly `conversation_id` or
                `session_id`); set this to match. Defaults to `conversation_id`.
            scope_search_to_conversation: When `True`, `search` is filtered to the
                bound `conversation_id`. When `False` (default), the conversation
                id only tags writes.
            topics: Default Engram topics to restrict `search` to.
            client: A pre-built synchronous client to reuse.
            async_client: A pre-built asynchronous client to reuse.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._retrieval_config = retrieval_config
        self._properties = properties
        self._conversation_id = conversation_id
        self._conversation_property = conversation_property
        self._scope_search_to_conversation = scope_search_to_conversation
        self._topics: list[Any] | None = topics
        self._client = client
        self._async_client = async_client

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

    # -- BaseStore API ------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of store operations synchronously."""
        client = self._sync_client()
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._do_get(client, op))
            elif isinstance(op, PutOp):
                self._do_put(client, op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._do_search(client, op))
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:  # pragma: no cover - defensive
                msg = f"Unsupported store operation: {type(op).__name__}"
                raise NotImplementedError(msg)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of store operations asynchronously."""
        client = self._get_async_client()
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self._ado_get(client, op))
            elif isinstance(op, PutOp):
                await self._ado_put(client, op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(await self._ado_search(client, op))
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:  # pragma: no cover - defensive
                msg = f"Unsupported store operation: {type(op).__name__}"
                raise NotImplementedError(msg)
        return results

    # -- sync handlers ------------------------------------------------------

    def _do_get(self, client: EngramClient, op: GetOp) -> Item | None:
        user_id, group = _scope(op.namespace)
        try:
            memory = client.memories.get(op.key, user_id=user_id, group=group)
        except EngramError:
            return None
        return _memory_to_item(memory, op.namespace)

    def _do_put(self, client: EngramClient, op: PutOp) -> None:
        user_id, group = _scope(op.namespace)
        if op.value is None:
            with suppress(EngramError):
                client.memories.delete(op.key, user_id=user_id, group=group)
            return
        client.memories.add(
            _extract_text(op.value),
            user_id=user_id,
            group=group,
            properties=self._write_properties(),
        )

    def _do_search(self, client: EngramClient, op: SearchOp) -> list[SearchItem]:
        user_id, group = _scope(op.namespace_prefix)
        results = client.memories.search(
            query=op.query or "",
            user_id=user_id,
            group=group,
            topics=self._topics,
            retrieval_config=self._retrieval(op.limit),
            properties=self._search_properties(op.filter),
        )
        return [_memory_to_search_item(m, op.namespace_prefix) for m in results]

    # -- async handlers -----------------------------------------------------

    async def _ado_get(self, client: AsyncEngramClient, op: GetOp) -> Item | None:
        user_id, group = _scope(op.namespace)
        try:
            memory = await client.memories.get(op.key, user_id=user_id, group=group)
        except EngramError:
            return None
        return _memory_to_item(memory, op.namespace)

    async def _ado_put(self, client: AsyncEngramClient, op: PutOp) -> None:
        user_id, group = _scope(op.namespace)
        if op.value is None:
            with suppress(EngramError):
                await client.memories.delete(op.key, user_id=user_id, group=group)
            return
        await client.memories.add(
            _extract_text(op.value),
            user_id=user_id,
            group=group,
            properties=self._write_properties(),
        )

    async def _ado_search(
        self, client: AsyncEngramClient, op: SearchOp
    ) -> list[SearchItem]:
        user_id, group = _scope(op.namespace_prefix)
        results = await client.memories.search(
            query=op.query or "",
            user_id=user_id,
            group=group,
            topics=self._topics,
            retrieval_config=self._retrieval(op.limit),
            properties=self._search_properties(op.filter),
        )
        return [_memory_to_search_item(m, op.namespace_prefix) for m in results]

    # -- helpers ------------------------------------------------------------

    def _retrieval(self, limit: int | None) -> Any:
        if self._retrieval_config is not None:
            return self._retrieval_config
        from engram import VectorRetrieval

        return VectorRetrieval(limit=limit)

    def _write_properties(self) -> dict[str, str] | None:
        merged: dict[str, str] = dict(self._properties or {})
        if self._conversation_id:
            merged[self._conversation_property] = self._conversation_id
        return merged or None

    def _search_properties(
        self, op_filter: dict[str, Any] | None
    ) -> dict[str, str] | None:
        merged: dict[str, str] = dict(self._properties or {})
        if self._scope_search_to_conversation and self._conversation_id:
            merged[self._conversation_property] = self._conversation_id
        for key, value in (op_filter or {}).items():
            if isinstance(value, (str, int, float, bool)):
                merged[key] = str(value)
        return merged or None


def _scope(namespace: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Map a namespace to an Engram (user_id, group) pair."""
    if not namespace:
        return None, None
    if len(namespace) == 1:
        return namespace[0], None
    if len(namespace) == 2:
        return namespace[0], namespace[1]
    msg = (
        "EngramStore namespaces map to (user_id,) or (user_id, group); "
        f"got {len(namespace)} levels: {namespace!r}"
    )
    raise ValueError(msg)


def _extract_text(value: dict[str, Any]) -> str:
    """Pull the memory text out of a stored value dict."""
    for key in _TEXT_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return json.dumps(value, sort_keys=True, default=str)


def _memory_value(memory: Memory) -> dict[str, Any]:
    return {
        "content": memory.content,
        "topic": memory.topic,
        "tags": memory.tags or [],
        "properties": memory.properties or {},
    }


def _memory_to_item(memory: Memory, namespace: tuple[str, ...]) -> Item:
    return Item(
        value=_memory_value(memory),
        key=memory.id,
        namespace=namespace,
        created_at=memory.created_at,  # type: ignore[arg-type]
        updated_at=memory.updated_at,  # type: ignore[arg-type]
    )


def _memory_to_search_item(memory: Memory, namespace: tuple[str, ...]) -> SearchItem:
    return SearchItem(
        namespace=namespace,
        key=memory.id,
        value=_memory_value(memory),
        created_at=memory.created_at,  # type: ignore[arg-type]
        updated_at=memory.updated_at,  # type: ignore[arg-type]
        score=memory.score,
    )
