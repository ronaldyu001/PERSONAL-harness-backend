"""Application-facing protocol for conversation persistence."""

from __future__ import annotations

from typing import Protocol

from application.conversation.schemas import (
    ConversationListRequest,
    ConversationListResult,
    ConversationReadRequest,
    ConversationReadResult,
    ConversationWriteRequest,
    ConversationWriteResult,
)


class ConversationPort(Protocol):
    """Application boundary implemented by concrete conversation stores."""

    async def write(self, request: ConversationWriteRequest) -> ConversationWriteResult:
        """Append one message to the conversation timeline."""
        ...

    async def list_conversations(
        self,
        request: ConversationListRequest,
    ) -> ConversationListResult:
        """List a user's conversations, most recently active first."""
        ...

    async def get_conversation(
        self,
        request: ConversationReadRequest,
    ) -> ConversationReadResult:
        """Read one conversation the user owns, with its messages."""
        ...
