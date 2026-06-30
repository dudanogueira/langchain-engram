"""End-to-end example: an agent that remembers across separate threads.

Prerequisites:
    pip install langchain-engram langchain-anthropic
    export ENGRAM_API_KEY=...
    export ANTHROPIC_API_KEY=...

Run:
    python examples/agent_with_memory.py
"""

from __future__ import annotations

from langchain.agents import create_agent

from langchain_engram import EngramMiddleware

USER_ID = "demo-user@example.com"


def main() -> None:
    agent = create_agent(
        "anthropic:claude-sonnet-4-6",
        middleware=[EngramMiddleware(user_id=USER_ID)],
    )

    # Turn 1: teach the agent a durable preference.
    agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Remember that I only drink decaf coffee."}
            ]
        }
    )

    # Turn 2: a brand-new conversation (no shared message history). The middleware
    # recalls the stored preference from Engram before the model answers.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Recommend a coffee for me."}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
