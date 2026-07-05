"""Schemas used by context component builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from application.llm.schemas import ChatMessage


@dataclass(frozen=True, slots=True)
class ConversationContext:
    """Conversation state needed for the current assistant turn."""

    conversation_id: str
    turn_id: str
    current_user_message: ChatMessage
    recent_messages: Sequence[ChatMessage] = ()
    summary: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
