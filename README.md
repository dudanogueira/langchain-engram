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

## Configuration

| Variable           | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `ENGRAM_API_KEY`   | API key (required unless passed in code) |
| `ENGRAM_BASE_URL`  | Override the API base URL (optional)     |

## License

BSD-3-Clause license
