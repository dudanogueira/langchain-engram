"""Unit tests for `EngramMiddleware` (mocked Engram client, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_engram import EngramMiddleware

from ._fakes import FakeEngramClient, make_memory


def _request(
    messages: list[Any], *, system: str | None = None, context: Any = None
) -> ModelRequest:
    return ModelRequest(
        model=SimpleNamespace(),  # never used; the handler is supplied by the test
        messages=messages,
        system_message=SystemMessage(content=system) if system else None,
        tools=[],
        state={"messages": messages},
        runtime=SimpleNamespace(context=context),
    )


def _capturing_handler() -> tuple[dict[str, Any], Any]:
    captured: dict[str, Any] = {}

    def handler(request: ModelRequest) -> str:
        captured["request"] = request
        return "RESPONSE"

    return captured, handler


def test_recall_injects_memories_into_system_message() -> None:
    fake = FakeEngramClient(search_results=[make_memory("The user only drinks decaf.")])
    mw = EngramMiddleware(user_id="alice")
    mw._client = fake

    captured, handler = _capturing_handler()
    out = mw.wrap_model_call(_request([HumanMessage("recommend a coffee")]), handler)

    assert out == "RESPONSE"
    # search was scoped to the bound user and used the latest human turn as query
    assert fake.recorder.search_calls[0]["user_id"] == "alice"
    assert fake.recorder.search_calls[0]["query"] == "recommend a coffee"
    # memory content reached the model via the system message
    assert "only drinks decaf" in captured["request"].system_message.content


def test_recall_preserves_existing_system_prompt() -> None:
    fake = FakeEngramClient(search_results=[make_memory("Likes Python.")])
    mw = EngramMiddleware(user_id="alice")
    mw._client = fake

    captured, handler = _capturing_handler()
    mw.wrap_model_call(
        _request([HumanMessage("hi")], system="You are helpful."), handler
    )

    content = captured["request"].system_message.content
    assert "You are helpful." in content
    assert "Likes Python." in content


def test_recall_noop_when_no_memories() -> None:
    fake = FakeEngramClient(search_results=[])
    mw = EngramMiddleware(user_id="alice")
    mw._client = fake

    captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)

    assert captured["request"].system_message is None


def test_recall_swallows_errors() -> None:
    fake = FakeEngramClient(raise_on_search=True)
    mw = EngramMiddleware(user_id="alice")
    mw._client = fake

    captured, handler = _capturing_handler()
    # must not raise — the agent keeps running without memory
    out = mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    assert out == "RESPONSE"
    assert captured["request"].system_message is None


def test_recall_disabled() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    mw = EngramMiddleware(user_id="alice", recall=False)
    mw._client = fake

    _captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    assert fake.recorder.search_calls == []


def test_write_persists_conversation_on_after_agent() -> None:
    fake = FakeEngramClient()
    mw = EngramMiddleware(user_id="alice")
    mw._client = fake

    state = {"messages": [HumanMessage("I love hiking."), AIMessage("Noted!")]}
    mw.after_agent(state, SimpleNamespace(context=None))

    assert len(fake.recorder.add_calls) == 1
    call = fake.recorder.add_calls[0]
    assert call["user_id"] == "alice"
    roles = [m["role"] for m in call["input"]]
    assert roles == ["user", "assistant"]
    assert call["input"][0]["content"] == "I love hiking."


def test_write_disabled() -> None:
    fake = FakeEngramClient()
    mw = EngramMiddleware(user_id="alice", write=False)
    mw._client = fake
    mw.after_agent({"messages": [HumanMessage("hi")]}, SimpleNamespace(context=None))
    assert fake.recorder.add_calls == []


def test_user_id_resolved_from_runtime_context() -> None:
    fake = FakeEngramClient(search_results=[make_memory("y")])
    mw = EngramMiddleware()  # no static user_id
    mw._client = fake

    _captured, handler = _capturing_handler()
    mw.wrap_model_call(
        _request([HumanMessage("hi")], context={"user_id": "bob"}), handler
    )
    assert fake.recorder.search_calls[0]["user_id"] == "bob"


def test_missing_user_id_is_swallowed_during_recall() -> None:
    fake = FakeEngramClient(search_results=[make_memory("y")])
    mw = EngramMiddleware()
    mw._client = fake

    captured, handler = _capturing_handler()
    # user id cannot be resolved -> error swallowed, model still called
    out = mw.wrap_model_call(_request([HumanMessage("hi")], context=None), handler)
    assert out == "RESPONSE"
    assert fake.recorder.search_calls == []


async def test_async_recall_and_write() -> None:
    fake = FakeEngramClient(
        search_results=[make_memory("The user is vegetarian.")], is_async=True
    )
    mw = EngramMiddleware(user_id="alice")
    mw._async_client = fake

    captured: dict[str, Any] = {}

    async def handler(request: ModelRequest) -> str:
        captured["request"] = request
        return "RESPONSE"

    out = await mw.awrap_model_call(_request([HumanMessage("dinner ideas?")]), handler)
    assert out == "RESPONSE"
    assert "vegetarian" in captured["request"].system_message.content

    await mw.aafter_agent(
        {"messages": [HumanMessage("hi"), AIMessage("hello")]},
        SimpleNamespace(context=None),
    )
    assert len(fake.recorder.add_calls) == 1


def test_constructor_requires_no_network() -> None:
    # building the middleware must not touch the network or require a key
    mw = EngramMiddleware(user_id="alice")
    assert mw._client is None
    assert mw._async_client is None


@pytest.mark.parametrize("retrieval", ["vector", "hybrid"])
def test_retrieval_config_passed_through(retrieval: str) -> None:
    fake = FakeEngramClient(search_results=[make_memory("z")])
    mw = EngramMiddleware(user_id="alice", retrieval_config=retrieval)
    mw._client = fake
    _captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    assert fake.recorder.search_calls[0]["retrieval_config"] == retrieval


# -- conversation_id --------------------------------------------------------


def test_conversation_id_tags_writes_but_not_recall_by_default() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    mw = EngramMiddleware(user_id="alice", conversation_id="conv-1")
    mw._client = fake

    _captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    mw.after_agent(
        {"messages": [HumanMessage("hi"), AIMessage("yo")]},
        SimpleNamespace(context=None),
    )

    # write is tagged with the conversation under the default `conversation_id` scope
    assert fake.recorder.add_calls[0]["properties"] == {"conversation_id": "conv-1"}
    # recall spans all of the user's memories (not filtered to the conversation)
    assert fake.recorder.search_calls[0]["properties"] is None


def test_conversation_id_scopes_recall_when_requested() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    mw = EngramMiddleware(
        user_id="alice",
        conversation_id="conv-1",
        scope_recall_to_conversation=True,
        properties={"app": "web"},
    )
    mw._client = fake

    _captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    assert fake.recorder.search_calls[0]["properties"] == {
        "app": "web",
        "conversation_id": "conv-1",
    }


def test_conversation_id_resolved_from_runtime_context() -> None:
    fake = FakeEngramClient()
    mw = EngramMiddleware(user_id="alice")  # conversation_id read from context
    mw._client = fake

    mw.after_agent(
        {"messages": [HumanMessage("hi"), AIMessage("yo")]},
        SimpleNamespace(context={"conversation_id": "ctx-conv"}),
    )
    assert fake.recorder.add_calls[0]["properties"] == {"conversation_id": "ctx-conv"}


def test_custom_conversation_property_name() -> None:
    fake = FakeEngramClient()
    mw = EngramMiddleware(
        user_id="alice", conversation_id="c9", conversation_property="thread"
    )
    mw._client = fake
    mw.after_agent(
        {"messages": [HumanMessage("hi"), AIMessage("yo")]},
        SimpleNamespace(context=None),
    )
    assert fake.recorder.add_calls[0]["properties"] == {"thread": "c9"}


# -- topics -----------------------------------------------------------------


def test_topics_default_passed_to_search() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    mw = EngramMiddleware(user_id="alice", topics=["preferences", "profile"])
    mw._client = fake
    _captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    assert fake.recorder.search_calls[0]["topics"] == ["preferences", "profile"]


def test_topics_overridden_at_runtime() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    mw = EngramMiddleware(user_id="alice", topics=["preferences"])
    mw._client = fake
    _captured, handler = _capturing_handler()
    mw.wrap_model_call(
        _request([HumanMessage("hi")], context={"topics": ["travel"]}), handler
    )
    # runtime context overrides the instantiation default
    assert fake.recorder.search_calls[0]["topics"] == ["travel"]


def test_topics_none_when_unset() -> None:
    fake = FakeEngramClient(search_results=[make_memory("x")])
    mw = EngramMiddleware(user_id="alice")
    mw._client = fake
    _captured, handler = _capturing_handler()
    mw.wrap_model_call(_request([HumanMessage("hi")]), handler)
    assert fake.recorder.search_calls[0]["topics"] is None
