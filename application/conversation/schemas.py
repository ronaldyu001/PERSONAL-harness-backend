"""Application schemas for conversation persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from domain.entities.conversation import ConversationMessage


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
