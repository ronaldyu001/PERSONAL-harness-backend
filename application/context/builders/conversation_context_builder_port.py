"""Conversation context builder contract."""

from __future__ import annotations

from typing import Protocol

from application.context.builders.schemas import ConversationContext
from application.context.schemas import ContextBlock


class ConversationContextBuilder(Protocol):
    """Builds one conversation context block."""

    async def build(self, *args: object, **kwargs: object) -> ContextBlock[ConversationContext]:
        """Build conversation context for one assistant turn."""
        ...
