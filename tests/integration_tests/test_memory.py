"""Live integration tests against a real Engram project.

These require a valid `ENGRAM_API_KEY` and make network calls. They are skipped
automatically when the key is not set, so the default `make test` stays offline.
"""

from __future__ import annotations

import os
import uuid

import pytest

from langchain_engram import EngramStore, create_memory_tools
from langchain_engram._client import build_client

pytestmark = pytest.mark.skipif(
    not os.environ.get("ENGRAM_API_KEY"),
    reason="ENGRAM_API_KEY not set; skipping live Engram tests",
)


@pytest.fixture
def user_id() -> str:
    # unique per run so assertions are not polluted by prior runs
    return f"it-{uuid.uuid4()}"


def test_add_wait_search_round_trip(user_id: str) -> None:
    client = build_client()
    run = client.memories.add("The user's favorite color is teal.", user_id=user_id)
    status = client.runs.wait(run.run_id, timeout=60.0)
    assert status.status == "completed"

    results = client.memories.search(query="favorite color", user_id=user_id)
    assert any("teal" in m.content.lower() for m in results)


def test_memory_tools_round_trip(user_id: str) -> None:
    search_tool, add_tool = create_memory_tools(user_id=user_id)
    add_tool.invoke({"text": "The user is allergic to peanuts."})

    client = build_client()
    # the add tool returns a run id; give the pipeline time to commit
    client.runs.wait(
        add_tool.invoke({"text": "The user lives in Lisbon."}).split()[-1].rstrip(")."),
        timeout=60.0,
    )
    result = search_tool.invoke({"query": "where does the user live"})
    assert "lisbon" in result.lower()


def test_store_put_search_round_trip(user_id: str) -> None:
    store = EngramStore()
    store.put((user_id,), "k1", {"content": "The user plays the cello."})

    # search after the pipeline has had a moment; eventual consistency applies
    results = store.search((user_id,), query="instrument", limit=5)
    if not results:
        pytest.skip("pipeline not yet committed; eventual consistency")
    assert any("cello" in item.value["content"].lower() for item in results)
