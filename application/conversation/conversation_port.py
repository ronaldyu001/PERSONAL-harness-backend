"""Application-facing protocol for conversation persistence."""

from __future__ import annotations

from typing import Protocol

from application.conversation.schemas import (
    ConversationWriteRequest,
    ConversationWriteResult,
)


class ConversationPort(Protocol):
    """Application boundary implemented by concrete conversation stores."""

    async def write(self, request: ConversationWriteRequest) -> ConversationWriteResult:
        """Append one message to the conversation timeline."""
        ...
