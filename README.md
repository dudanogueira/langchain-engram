# langchain-engram

LangChain integration for [Weaviate Engram](https://docs.weaviate.io/engram) — managed,
production-grade memory for AI agents.

Engram runs an asynchronous pipeline that extracts facts from conversations, reconciles and
dedupes them, and stores them scoped per project, user, and custom property. This package lets
LangChain agents read and write that memory with no glue code.

## Installation

```bash
pip install langchain-engram
```

Set your API key (create one at <https://console.weaviate.cloud/engram>):

```bash
export ENGRAM_API_KEY="..."
```

## Surfaces

This package exposes three ways to use Engram, from most to least automatic.

### 1. `EngramMiddleware` — drop-in agent memory

Auto-recalls relevant memories before each model call and persists the conversation when the
agent finishes. The simplest option.

```python
from langchain.agents import create_agent
from langchain_engram import EngramMiddleware

agent = create_agent(
    "anthropic:claude-sonnet-4-6",
    middleware=[EngramMiddleware(user_id="alice@example.com")],
)

# Turn 1 — teach it something
agent.invoke({"messages": [{"role": "user", "content": "I only drink decaf."}]})

# Turn 2 (even a brand new thread) — it remembers
agent.invoke({"messages": [{"role": "user", "content": "Recommend me a coffee."}]})
```

Resolve `user_id` dynamically from the agent's runtime context instead of binding it:

```python
EngramMiddleware()  # reads context["user_id"]
EngramMiddleware(user_id=lambda runtime: runtime.context["account_id"])
```

### 2. Memory tools — explicit agent control

Give the model `search_memories` and `add_memories` tools it can call deliberately. The
`user_id` scope is fixed by you, never by the model.

```python
from langchain.agents import create_agent
from langchain_engram import create_memory_tools

tools = create_memory_tools(user_id="alice@example.com")
agent = create_agent("anthropic:claude-sonnet-4-6", tools=tools)
```

### 3. `EngramStore` — a LangGraph `BaseStore` (experimental)

Point existing LangGraph store-based code (including LangMem) at Engram:

```python
from langchain.agents import create_agent
from langchain_engram import EngramStore

agent = create_agent("anthropic:claude-sonnet-4-6", store=EngramStore())
```

> **Note:** Engram is a memory-*extraction* service, not an exact key-value store. `put` is
> processed asynchronously and Engram assigns its own deduped ids, so the key you write is not
> the key you read back — use `search` to retrieve, then `get`/`delete` by the returned key.
> `list_namespaces` is not supported. Namespaces map as `(user_id,)` or `(user_id, group)`.

## Examples

Runnable Jupyter notebooks that install this package from source and exercise every
surface against a live Engram project live in [`notebooks/`](notebooks/) — a ready-made
testing playground. A plain script version is in [`examples/`](examples/).

## Scoping

All surfaces accept Engram's scoping knobs: `user_id` (the memory owner), an optional `group`,
and arbitrary `properties` (for example `{"app": "support"}`). Searches additionally accept a
`retrieval_config` — a named type (`"vector"`, `"bm25"`, `"hybrid"`, `"fetch"`) or a retrieval
model such as `HybridRetrieval(limit=5)`.

> **Required scope properties.** If your Engram group has **scoped topics**, every write must
> include the properties that group requires, or the API rejects it:
>
> ```
> APIError: group 'default': insufficient scope: missing required scope properties
> [custom_scope_1 some_other] to write memories
> ```
>
> Supply them through `properties=` on whichever surface you use — they are sent on both `add`
> and `search`:
>
> ```python
> EngramMiddleware(user_id="alice", properties={"custom_scope_1": "web", "some_other": "v1"})
> create_memory_tools(user_id="alice", properties={"custom_scope_1": "web", "some_other": "v1"})
> EngramStore(properties={"custom_scope_1": "web", "some_other": "v1"})
> ```
>
> A `conversation_id` (stored under `conversation_property`) is one such scope property and is
> merged in automatically.

> **`user_id` is only honored on user-scoped topics.** Engram attaches `user_id` (and lets you
> filter searches by it) **only when the target topic is configured as user-scoped**. If a write
> requires custom scope properties but *not* `user_id` (e.g. it asks for `[custom_scope_1
> some_other]` only), that topic is not user-scoped: the stored memory's `user_id` is `None` and
> a search filtered by `user_id` won't narrow to it. The package always sends `user_id`; to
> actually tie memories to users, add `user_id` to the topic's scope in the Engram console.

### Tying memories to a conversation

> **Naming — the scope property is defined by *your* Engram project.** When you create a project
> you choose the scope property name; it is commonly `conversation_id` or `session_id` (the same
> concept LangGraph calls `thread_id`). `conversation_id` is **not** a reserved LangChain key — this
> package uses it only as a friendly parameter name for the *value*, and stores it under the
> property named by `conversation_property` (default `conversation_id`). **Set
> `conversation_property` to match whatever your project's scope property is** (e.g. `session_id`).

> **Prerequisite:** your Engram project must have the **conversation scope available** — that is, a
> topic configured as `scoped by <your scope property>` (for example a `ConversationSummary` topic)
> in your group's pipeline (Engram console). Without it, the value is only a tag/filter and
> reconciliation stays user-level (see the note below).

Pass a `conversation_id` to attach the scope property to every written memory (and, when scoped
recall is on, to the search filter):

```python
# Default property name -> stored as properties={"conversation_id": "abc123"}
EngramMiddleware(user_id="alice", conversation_id="abc123")

# Match a project whose scope property is `session_id`
EngramMiddleware(user_id="alice", conversation_id="abc123", conversation_property="session_id")

# Resolved per invocation from runtime context
agent = create_agent("anthropic:claude-sonnet-4-6", middleware=[EngramMiddleware(user_id="alice")])
agent.invoke({"messages": [...]}, context={"conversation_id": "abc123"})
```

You can also change the context key with `conversation_id_context_key="thread_id"`.

> **Important — how much isolation you get depends on your Engram topic config.**
> In Engram, only `user_id` and project are absolute isolation boundaries. The scope property is a
> *filter*, not a hard boundary, **unless the target topic is configured to be `scoped by` that
> property** in your group's pipeline (set this in the Engram console).
>
> - **Topic *not* scoped by the property (default):** the value is attribution + filtering only.
>   Engram's reconciliation runs at the user level, so contradictory facts from different
>   conversations get merged into one **unscoped** memory. A property-filtered search returns that
>   conversation's memories **plus** unscoped memories (the merged ones surface in every
>   conversation); only *other* property values are excluded.
> - **Topic *scoped by* the property:** reconciliation stays within each conversation and a
>   filtered search returns only that conversation's memories — true per-conversation isolation.
>
> This package sends `properties={<conversation_property>: ...}` on both `add` and `search`, which
> is exactly what a scoped topic needs — so once your `conversation_property` matches the topic's
> scope, scoping "just works." `scope_recall_to_conversation=True` (or
> `scope_search_to_conversation=True` on the tools/store) adds the property to the search filter.
> For hard isolation without touching topic config, use a distinct `user_id` per conversation.

### Restricting search to topics

Pass `topics` to limit search to specific Engram topics. The default set is configurable at
construction and overridable per invocation through the runtime context (`topics_context_key`,
default `"topics"`):

```python
# Default topics for every search
EngramMiddleware(user_id="alice", topics=["preferences", "profile"])

# Override for one invocation
agent.invoke({"messages": [...]}, context={"user_id": "alice", "topics": ["travel"]})
```

## Configuration

| Variable           | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `ENGRAM_API_KEY`   | API key (required unless passed in code) |
| `ENGRAM_BASE_URL`  | Override the API base URL (optional)     |

## License

BSD-3-Clause license
