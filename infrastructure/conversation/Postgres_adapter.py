"""SQLAlchemy implementation of the application conversation port."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import uuid4

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from application.conversation.schemas import (
    ConversationWriteRequest,
    ConversationWriteResult,
)
from database.engines.Maia import create_session_factory
from database.models.Maia import Conversation, ConversationMessage
from domain.entities.conversation import (
    ConversationMessage as ConversationMessageEntity,
)


# Matches the frontend's title rule so a stored title reads like the one the
# UI already shows (PERSONAL-harness-frontend/src/App.tsx).
TITLE_MAX_LENGTH = 44
UNTITLED_CONVERSATION = "New conversation"

_TRAILING_PARTIAL_WORD = re.compile(r"\s+\S*$")


class PostgresConversationAdapter:
    """Append conversation messages to Postgres through SQLAlchemy."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> PostgresConversationAdapter:
        """Build the adapter from an engine owned by the composition root."""
        return cls(create_session_factory(engine))

    async def write(
        self,
        request: ConversationWriteRequest,
    ) -> ConversationWriteResult:
        """Append one message, creating its conversation on first write."""
        message = request.message
        message_id = message.message_id or str(uuid4())
        occurred_at = message.occurred_at

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    self._conversation_upsert(message, occurred_at)
                )
                await session.execute(
                    self._message_insert(message, message_id, occurred_at)
                )

        return ConversationWriteResult(
            message_id=message_id,
            conversation_id=message.conversation_id,
        )

    @staticmethod
    def _conversation_upsert(
        message: ConversationMessageEntity,
        occurred_at: datetime,
    ):
        """Create the conversation row, or bump its activity timestamp.

        The title is written once, by whichever message arrives first, and is
        never overwritten by a later turn.
        """
        statement = insert(Conversation).values(
            conversation_id=message.conversation_id,
            user_id=message.user_id or "",
            title=derive_title(message),
            created_at=occurred_at,
            last_updated=occurred_at,
        )

        return statement.on_conflict_do_update(
            index_elements=[Conversation.conversation_id],
            set_={"last_updated": statement.excluded.last_updated},
        )

    @staticmethod
    def _message_insert(
        message: ConversationMessageEntity,
        message_id: str,
        occurred_at: datetime,
    ):
        """Insert the message, ignoring a turn that was already recorded."""
        metadata = dict(message.metadata)
        statement = insert(ConversationMessage).values(
            message_id=message_id,
            conversation_id=message.conversation_id,
            user_id=message.user_id,
            role=message.role,
            content=message.content,
            created_at=occurred_at,
            metadata_=metadata or None,
        )

        return statement.on_conflict_do_nothing(
            index_elements=[ConversationMessage.message_id]
        )


def derive_title(message: ConversationMessageEntity) -> str:
    """Derive a conversation title from the message that opened it."""
    if message.role != "user":
        return UNTITLED_CONVERSATION
    return truncate_title(message.content)


def truncate_title(text: str) -> str:
    """Shorten a title to one line, cutting on a word boundary."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return UNTITLED_CONVERSATION
    if len(collapsed) <= TITLE_MAX_LENGTH:
        return collapsed

    head = collapsed[:TITLE_MAX_LENGTH]
    # A single very long word leaves nothing to cut back to.
    trimmed = _TRAILING_PARTIAL_WORD.sub("", head) or head
    return f"{trimmed}…"
