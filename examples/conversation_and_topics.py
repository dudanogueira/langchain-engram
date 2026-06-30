"""Example: scope Engram memory by conversation and by topic.

Demonstrates two scoping knobs shared by every surface:

- `conversation_id` — tags each written memory with its conversation. Recall spans
  all of the user's memories by default; `scope_recall_to_conversation=True` makes
  it conversation-local.
- `topics` — restricts search to specific Engram topics.

Prerequisites:
    pip install langchain-engram langchain-anthropic
    export ENGRAM_API_KEY=...
    export ANTHROPIC_API_KEY=...

Run:
    python examples/conversation_and_topics.py
"""

from __future__ import annotations

from langchain.agents import create_agent

from langchain_engram import EngramMiddleware
from langchain_engram._client import build_client

USER_ID = "conversation-demo@example.com"

# The scope property name is defined when you create your Engram project — commonly
# "conversation_id" or "session_id". Set this to match your project's configuration.
SCOPE_PROPERTY = "conversation_id"

# The topic recall reads from. Must exist in your project (e.g. a conversation summary).
RECALL_TOPIC = "ConversationSummary"


def main() -> None:
    client = build_client()

    # Seed two memories for the same user in two different conversations and wait
    # for Engram's pipeline to commit them.
    for text, conversation in [
        ("The user wants a window seat.", "trip-tokyo"),
        ("The user wants an aisle seat.", "trip-lisbon"),
    ]:
        run = client.memories.add(
            text, user_id=USER_ID, properties={SCOPE_PROPERTY: conversation}
        )
        client.runs.wait(run.run_id, timeout=60.0)

    # A property filter retrieves only the matching conversation.
    tokyo = client.memories.search(
        query="seat", user_id=USER_ID, properties={SCOPE_PROPERTY: "trip-tokyo"}
    )
    print("Tokyo-only memories:", [m.content for m in tokyo])

    # The middleware applies the same scoping automatically. We bind recall to the
    # ConversationSummary topic only, tag writes with the conversation under the
    # project's scope property, and add the conversation to the recall filter.
    agent = create_agent(
        "anthropic:claude-sonnet-4-6",
        middleware=[
            EngramMiddleware(
                user_id=USER_ID,
                conversation_id="trip-tokyo",
                conversation_property=SCOPE_PROPERTY,
                topics=[RECALL_TOPIC],
                scope_recall_to_conversation=True,
            )
        ],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Which seat do I want?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
