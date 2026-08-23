"""Application schemas for conversation persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from domain.entities import Conversation, ConversationMessage


@dataclass(frozen=True, slots=True)
class ConversationWriteRequest:
    """Request to append one message to a conversation."""

    message: ConversationMessage


@dataclass(frozen=True, slots=True)
class ConversationWriteResult:
    """Result of appending one message to a conversation."""

    message_id: str
    conversation_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversationInfo:
    """One conversation without its messages, for listing."""

    conversation_id: str
    title: str
    created_at: datetime
    last_updated: datetime


@dataclass(frozen=True, slots=True)
class ConversationListRequest:
    """Request the most recently active conversations for one user.

    ``limit`` of ``None`` means the caller did not ask for a count; the use
    case resolves it from configured policy.
    """

    user_id: str
    limit: int | None = None

    def __post_init__(self) -> None:
        """Validate the requested listing window."""
        if not self.user_id.strip():
            raise ValueError("user_id is required")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")


@dataclass(frozen=True, slots=True)
class ConversationListResult:
    """Conversations ordered by most recent activity first."""

    conversations: tuple[ConversationInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class ConversationReadRequest:
    """Request one conversation and its messages."""

    conversation_id: str
    user_id: str

    def __post_init__(self) -> None:
        """Validate the identifiers a read is scoped to."""
        if not self.conversation_id.strip():
            raise ValueError("conversation_id is required")
        if not self.user_id.strip():
            raise ValueError("user_id is required")


@dataclass(frozen=True, slots=True)
class ConversationReadResult:
    """One conversation, or nothing when the user does not own it."""

    conversation: Conversation | None = None
