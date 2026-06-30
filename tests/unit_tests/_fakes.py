"""In-memory fakes for the Engram SDK used by unit tests (no network)."""

from __future__ import annotations

from typing import Any

from engram._models import Memory, Run, SearchResults
from engram.errors import EngramError

_TS = "2026-06-30T00:00:00+00:00"


def make_memory(
    content: str,
    *,
    memory_id: str = "mem-1",
    score: float | None = None,
    user_id: str | None = "alice",
) -> Memory:
    """Build a `Memory` with sensible test defaults."""
    return Memory(
        id=memory_id,
        project_id="proj-1",
        content=content,
        topic="general",
        group="default",
        created_at=_TS,
        updated_at=_TS,
        user_id=user_id,
        tags=[],
        score=score,
        properties={},
    )


class _Recorder:
    """Records calls and serves canned search/get responses."""

    def __init__(
        self,
        search_results: list[Memory] | None,
        get_results: dict[str, Memory] | None,
        raise_on_search: bool,
    ) -> None:
        self._search_results = search_results or []
        self._get_results = get_results or {}
        self._raise_on_search = raise_on_search
        self.add_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    def record_add(self, input_data: Any, **kwargs: Any) -> Run:
        self.add_calls.append({"input": input_data, **kwargs})
        return Run(run_id="run-1", status="pending")

    def record_search(self, **kwargs: Any) -> SearchResults:
        if self._raise_on_search:
            msg = "boom"
            raise EngramError(msg)
        self.search_calls.append(kwargs)
        return SearchResults(
            list(self._search_results), total=len(self._search_results)
        )

    def record_get(self, memory_id: str, **kwargs: Any) -> Memory:
        self.get_calls.append({"memory_id": memory_id, **kwargs})
        if memory_id not in self._get_results:
            msg = f"memory {memory_id} not found"
            raise EngramError(msg)
        return self._get_results[memory_id]

    def record_delete(self, memory_id: str, **kwargs: Any) -> None:
        self.delete_calls.append({"memory_id": memory_id, **kwargs})


class FakeMemories:
    """Synchronous fake of `client.memories`."""

    def __init__(self, recorder: _Recorder) -> None:
        self._r = recorder

    def add(self, input_data: Any, **kwargs: Any) -> Run:
        return self._r.record_add(input_data, **kwargs)

    def search(self, **kwargs: Any) -> SearchResults:
        return self._r.record_search(**kwargs)

    def get(self, memory_id: str, **kwargs: Any) -> Memory:
        return self._r.record_get(memory_id, **kwargs)

    def delete(self, memory_id: str, **kwargs: Any) -> None:
        self._r.record_delete(memory_id, **kwargs)


class AsyncFakeMemories:
    """Asynchronous fake of `client.memories`."""

    def __init__(self, recorder: _Recorder) -> None:
        self._r = recorder

    async def add(self, input_data: Any, **kwargs: Any) -> Run:
        return self._r.record_add(input_data, **kwargs)

    async def search(self, **kwargs: Any) -> SearchResults:
        return self._r.record_search(**kwargs)

    async def get(self, memory_id: str, **kwargs: Any) -> Memory:
        return self._r.record_get(memory_id, **kwargs)

    async def delete(self, memory_id: str, **kwargs: Any) -> None:
        self._r.record_delete(memory_id, **kwargs)


class FakeEngramClient:
    """Stands in for `EngramClient`/`AsyncEngramClient` in unit tests."""

    def __init__(
        self,
        *,
        search_results: list[Memory] | None = None,
        get_results: dict[str, Memory] | None = None,
        raise_on_search: bool = False,
        is_async: bool = False,
    ) -> None:
        self.recorder = _Recorder(search_results, get_results, raise_on_search)
        self.memories: FakeMemories | AsyncFakeMemories = (
            AsyncFakeMemories(self.recorder)
            if is_async
            else FakeMemories(self.recorder)
        )
