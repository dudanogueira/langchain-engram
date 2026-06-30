"""Unit tests for the memory tools (mocked Engram client, no network)."""

from __future__ import annotations

import pytest

from langchain_engram import create_memory_tools

from ._fakes import FakeEngramClient, make_memory


def test_tools_names_and_scope_safety() -> None:
    fake = FakeEngramClient()
    search_tool, add_tool = create_memory_tools(user_id="alice", client=fake)

    assert search_tool.name == "search_memories"
    assert add_tool.name == "add_memories"
    # the model controls only the query/text — never the user scope
    assert set(search_tool.args) == {"query"}
    assert set(add_tool.args) == {"text"}


def test_search_returns_formatted_memories() -> None:
    fake = FakeEngramClient(
        search_results=[make_memory("Prefers email."), make_memory("Lives in Berlin.")]
    )
    search_tool, _ = create_memory_tools(user_id="alice", client=fake)

    result = search_tool.invoke({"query": "contact preferences"})

    assert "Prefers email." in result
    assert "Lives in Berlin." in result
    assert fake.recorder.search_calls[0]["user_id"] == "alice"


def test_search_handles_no_results() -> None:
    fake = FakeEngramClient(search_results=[])
    search_tool, _ = create_memory_tools(user_id="alice", client=fake)
    assert search_tool.invoke({"query": "anything"}) == "No relevant memories found."


def test_add_stores_memory() -> None:
    fake = FakeEngramClient()
    _, add_tool = create_memory_tools(user_id="alice", client=fake)

    result = add_tool.invoke({"text": "I'm vegetarian."})

    assert "run-1" in result
    assert fake.recorder.add_calls[0]["input"] == "I'm vegetarian."
    assert fake.recorder.add_calls[0]["user_id"] == "alice"


def test_scope_properties_and_group_applied() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    search_tool, _ = create_memory_tools(
        user_id="alice", group="support", properties={"app": "web"}, client=fake
    )
    search_tool.invoke({"query": "q"})
    call = fake.recorder.search_calls[0]
    assert call["group"] == "support"
    assert call["properties"] == {"app": "web"}


def test_missing_user_id_raises() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    search_tool, _ = create_memory_tools(client=fake)  # no user_id, no runtime
    with pytest.raises(ValueError, match="Could not resolve an Engram user id"):
        search_tool.invoke({"query": "q"})


def test_conversation_id_tags_add() -> None:
    fake = FakeEngramClient()
    _, add_tool = create_memory_tools(
        user_id="alice", conversation_id="conv-7", client=fake
    )
    add_tool.invoke({"text": "Likes aisle seats."})
    assert fake.recorder.add_calls[0]["properties"] == {"conversation_id": "conv-7"}


def test_search_not_conversation_scoped_by_default() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    search_tool, _ = create_memory_tools(
        user_id="alice", conversation_id="conv-7", client=fake
    )
    search_tool.invoke({"query": "q"})
    assert fake.recorder.search_calls[0]["properties"] is None


def test_topics_passed_to_search_tool() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    search_tool, _ = create_memory_tools(
        user_id="alice", topics=["profile"], client=fake
    )
    search_tool.invoke({"query": "q"})
    assert fake.recorder.search_calls[0]["topics"] == ["profile"]
