"""Unit tests for `EngramStore` (mocked Engram client, no network)."""

from __future__ import annotations

import pytest

from langchain_engram import EngramStore

from ._fakes import FakeEngramClient, make_memory


def test_put_adds_memory_text() -> None:
    fake = FakeEngramClient()
    store = EngramStore(client=fake)

    store.put(("alice",), "ignored-key", {"content": "Likes jazz."})

    call = fake.recorder.add_calls[0]
    assert call["input"] == "Likes jazz."
    assert call["user_id"] == "alice"
    assert call["group"] is None


def test_put_with_group_namespace() -> None:
    fake = FakeEngramClient()
    store = EngramStore(client=fake)
    store.put(("alice", "support"), "k", {"text": "Opened a ticket."})
    call = fake.recorder.add_calls[0]
    assert call["user_id"] == "alice"
    assert call["group"] == "support"
    assert call["input"] == "Opened a ticket."


def test_put_falls_back_to_json_for_unknown_shape() -> None:
    fake = FakeEngramClient()
    store = EngramStore(client=fake)
    store.put(("alice",), "k", {"foo": "bar"})
    assert fake.recorder.add_calls[0]["input"] == '{"foo": "bar"}'


def test_search_maps_results_to_search_items() -> None:
    fake = FakeEngramClient(
        search_results=[make_memory("Drinks decaf.", memory_id="m9", score=0.87)]
    )
    store = EngramStore(client=fake)

    items = store.search(("alice",), query="coffee", limit=3)

    assert len(items) == 1
    item = items[0]
    assert item.key == "m9"
    assert item.value["content"] == "Drinks decaf."
    assert item.score == 0.87
    assert item.namespace == ("alice",)
    # limit flows into a vector retrieval config by default
    assert fake.recorder.search_calls[0]["retrieval_config"].limit == 3


def test_search_filter_becomes_properties() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    store = EngramStore(client=fake, properties={"app": "web"})
    store.search(("alice",), query="q", filter={"channel": "email"})
    props = fake.recorder.search_calls[0]["properties"]
    assert props == {"app": "web", "channel": "email"}


def test_get_returns_item_for_known_id() -> None:
    fake = FakeEngramClient(get_results={"m1": make_memory("Hi.", memory_id="m1")})
    store = EngramStore(client=fake)
    item = store.get(("alice",), "m1")
    assert item is not None
    assert item.key == "m1"
    assert item.value["content"] == "Hi."


def test_get_missing_returns_none() -> None:
    fake = FakeEngramClient(get_results={})
    store = EngramStore(client=fake)
    assert store.get(("alice",), "nope") is None


def test_delete_calls_sdk() -> None:
    fake = FakeEngramClient()
    store = EngramStore(client=fake)
    store.delete(("alice",), "m1")
    assert fake.recorder.delete_calls[0]["memory_id"] == "m1"
    assert fake.recorder.delete_calls[0]["user_id"] == "alice"


def test_deep_namespace_rejected() -> None:
    store = EngramStore(client=FakeEngramClient())
    with pytest.raises(ValueError, match="namespaces map to"):
        store.put(("a", "b", "c"), "k", {"content": "x"})


def test_list_namespaces_unsupported_returns_empty() -> None:
    store = EngramStore(client=FakeEngramClient())
    assert store.list_namespaces() == []


def test_conversation_id_tags_put() -> None:
    fake = FakeEngramClient()
    store = EngramStore(client=fake, conversation_id="conv-3")
    store.put(("alice",), "k", {"content": "Plays cello."})
    assert fake.recorder.add_calls[0]["properties"] == {"conversation_id": "conv-3"}


def test_conversation_property_is_configurable() -> None:
    # the scope property name is defined per Engram project; allow any name
    fake = FakeEngramClient()
    store = EngramStore(
        client=fake, conversation_id="conv-3", conversation_property="session_id"
    )
    store.put(("alice",), "k", {"content": "x"})
    assert fake.recorder.add_calls[0]["properties"] == {"session_id": "conv-3"}


def test_search_conversation_scope_opt_in() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    store = EngramStore(
        client=fake, conversation_id="conv-3", scope_search_to_conversation=True
    )
    store.search(("alice",), query="q")
    assert fake.recorder.search_calls[0]["properties"] == {"conversation_id": "conv-3"}


def test_store_topics_passed_to_search() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    store = EngramStore(client=fake, topics=["profile"])
    store.search(("alice",), query="q")
    assert fake.recorder.search_calls[0]["topics"] == ["profile"]


async def test_async_put_and_search() -> None:
    fake = FakeEngramClient(
        search_results=[make_memory("Async fact.", memory_id="a1")], is_async=True
    )
    store = EngramStore(async_client=fake)

    await store.aput(("alice",), "k", {"content": "Async fact."})
    assert fake.recorder.add_calls[0]["input"] == "Async fact."

    items = await store.asearch(("alice",), query="fact")
    assert items[0].key == "a1"
