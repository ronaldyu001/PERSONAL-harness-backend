"""Conversation history read use case."""

from __future__ import annotations

from dataclasses import replace

from application.conversation import (
    ConversationListRequest,
    ConversationListResult,
    PortConversation,
    ConversationReadRequest,
    ConversationReadResult,
)


class UseCaseReadConversationHistory:
    """Reads persisted conversations back for one user.

    Only the read half of ``PortConversation`` is exercised here; the write
    half belongs to ``MiddlewareConversation``.
    """

    def __init__(
        self,
        conversations: PortConversation,
        *,
        default_list_size: int,
        max_list_size: int,
    ) -> None:
        """Create the use case with a store and its configured paging policy."""
        if default_list_size <= 0:
            raise ValueError("default_list_size must be positive")
        if max_list_size <= 0:
            raise ValueError("max_list_size must be positive")
        if default_list_size > max_list_size:
            raise ValueError("default_list_size must not exceed max_list_size")

        self._conversations = conversations
        self._default_list_size = default_list_size
        self._max_list_size = max_list_size

    async def list(self, request: ConversationListRequest) -> ConversationListResult:
        """List a user's conversations, most recently active first."""
        # Paging policy is resolved here, so stores always see a concrete count.
        requested = request.limit or self._default_list_size
        return await self._conversations.list_conversations(
            replace(request, limit=min(requested, self._max_list_size))
        )

    async def get(self, request: ConversationReadRequest) -> ConversationReadResult:
        """Read one conversation the user owns, with its messages."""
        return await self._conversations.get_conversation(request)
