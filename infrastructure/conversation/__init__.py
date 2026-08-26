"""Conversation persistence adapters."""

from infrastructure.conversation.adapter_postgres import (
    PostgresConversationAdapter,
)


__all__ = ("PostgresConversationAdapter",)
