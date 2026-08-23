"""Read completed turns out of an agent's message history."""

from __future__ import annotations

from langchain.messages import AIMessage, HumanMessage


def latest_user_text(messages: object) -> str | None:
    """Return the most recent non-empty human message."""
    if not isinstance(messages, (list, tuple)):
        return None

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.text.strip()
            if content:
                return content
    return None


def latest_completed_turn(
    messages: object,
) -> tuple[HumanMessage, AIMessage] | None:
    """Return the last assistant response and its preceding user message.

    Tool calls and tool results between the two are skipped: this is the
    user-visible turn, not the full agent trace.
    """
    if not isinstance(messages, (list, tuple)):
        return None

    assistant_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], AIMessage):
            assistant_index = index
            break

    if assistant_index is None:
        return None

    for index in range(assistant_index - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index], messages[assistant_index]

    return None
