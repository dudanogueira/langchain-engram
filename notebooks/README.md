# langchain-engram playground notebooks

Interactive notebooks for trying the integration against a **live** Engram project.
They install the package **from source** (editable), so any edit you make to
`langchain_engram/` is picked up after a kernel restart — a handy testing playground.

## Prerequisites

- An Engram API key — create one at <https://console.weaviate.cloud/engram>.
- For the agent notebooks (1, 2, and the last cell of 3): an `ANTHROPIC_API_KEY`.

The notebooks prompt for any missing key with `getpass`, so nothing is hard-coded.

## Running

From the repository root:

```bash
uv sync --all-groups          # set up the environment
uv run --with jupyter jupyter lab notebooks/
```

or with an existing Jupyter:

```bash
pip install -e ".[test]" jupyterlab   # editable install from source
jupyter lab notebooks/
```

### Loading the package — two options

Every notebook's setup section offers two interchangeable options (run **one**):

- **Option A — editable install:** `%pip install -q -e ".."` installs `langchain-engram`
  *and* its dependencies from source into the kernel.
- **Option B — local import (no install):** adds the repo root to `sys.path` so
  `import langchain_engram` resolves straight to the source tree. Nothing is installed,
  so launch Jupyter from an environment that already has the dependencies
  (`uv run --with jupyter jupyter lab`). Best for a tight edit/run loop on the source.

Both pick up your local edits after a kernel restart.

## Notebooks

| Notebook | What it shows | Needs LLM key |
| --- | --- | --- |
| `0_setup_and_smoke_test.ipynb` | Install, set keys, add → wait → search round trip | no |
| `1_middleware_agent_memory.ipynb` | `EngramMiddleware`: automatic recall + write, dynamic `user_id` | yes |
| `2_memory_tools.ipynb` | `create_memory_tools`: agent-controlled `search_memories` / `add_memories` | yes |
| `3_engram_store.ipynb` | `EngramStore`: the LangGraph `BaseStore` adapter | last cell only |

## A note on timing

Engram's write pipeline is **asynchronous** — a memory isn't searchable the instant
you add it. The notebooks handle this two ways, both reusable in your own tests:

- `seed_memory(...)` adds a memory and blocks on `client.runs.wait(...)` until the
  pipeline commits (deterministic).
- `wait_until_searchable(...)` polls `search` until an expected memory appears
  (for fire-and-forget writes such as the middleware's auto-write).
