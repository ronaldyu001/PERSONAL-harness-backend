"""Conversation domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Mapping, Self


ConversationMessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """One immutable message in a conversation timeline."""

    conversation_id: str
    role: ConversationMessageRole
    content: str
    message_id: str | None = None
    user_id: str | None = None
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required message invariants."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id is required")
        if not self.content.strip():
            raise ValueError("content is required")

    @property
    def occurred_at(self) -> datetime:
        """Return the persisted timestamp or a current UTC fallback."""
        return self.created_at or datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Conversation:
    """Aggregate root for a persisted multi-turn conversation."""

    conversation_id: str
    user_id: str | None = None
    title: str | None = None
    messages: tuple[ConversationMessage, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate conversation identity and message ownership."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id is required")

        for message in self.messages:
            if message.conversation_id != self.conversation_id:
                raise ValueError("message conversation_id must match conversation_id")

    @property
    def has_messages(self) -> bool:
        """Return whether the conversation has any persisted messages."""
        return bool(self.messages)

    @property
    def started_at(self) -> datetime | None:
        """Return the earliest known conversation timestamp."""
        if self.created_at is not None:
            return self.created_at
        if not self.messages:
            return None
        return min(message.occurred_at for message in self.messages)

    @property
    def last_activity_at(self) -> datetime | None:
        """Return the most recent known conversation timestamp."""
        if self.updated_at is not None:
            return self.updated_at
        if not self.messages:
            return self.created_at
        return max(message.occurred_at for message in self.messages)

    def append(self, message: ConversationMessage) -> Self:
        """Return a new conversation with one message appended."""
        if message.conversation_id != self.conversation_id:
            raise ValueError("message conversation_id must match conversation_id")

        return type(self)(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            title=self.title,
            messages=(*self.messages, message),
            created_at=self.created_at,
            updated_at=message.occurred_at,
            metadata=self.metadata,
        )

    def recent_messages(self, limit: int) -> tuple[ConversationMessage, ...]:
        """Return the most recent messages without mutating the conversation."""
        if limit <= 0:
            return ()
        return self.messages[-limit:]
