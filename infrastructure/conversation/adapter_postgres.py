"""SQLAlchemy implementation of the application conversation port."""

from __future__ import annotations

import re
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from application.conversation import (
    ConversationListRequest,
    ConversationListResult,
    ConversationReadRequest,
    ConversationReadResult,
    ConversationInfo,
    ConversationWriteRequest,
    ConversationWriteResult,
)
from database.engines.engine_maia import create_session_factory
from database.models.model_maia import Conversation, ConversationMessage
from domain.entities import (
    Conversation as ConversationEntity,
    ConversationMessage as ConversationMessageEntity,
)


# Matches the frontend's title rule so a stored title reads like the one the
# UI already shows (PERSONAL-harness-frontend/src/App.tsx).
TITLE_MAX_LENGTH = 44
UNTITLED_CONVERSATION = "New conversation"

_TRAILING_PARTIAL_WORD = re.compile(r"\s+\S*$")


class AdapterPostgresConversation:
    """Append conversation messages to Postgres through SQLAlchemy."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> AdapterPostgresConversation:
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

    async def list_conversations(
        self,
        request: ConversationListRequest,
    ) -> ConversationListResult:
        """List a user's conversations, most recently active first."""
        statement = (
            select(
                Conversation.conversation_id,
                Conversation.title,
                Conversation.created_at,
                Conversation.last_updated,
            )
            .where(Conversation.user_id == request.user_id)
            .order_by(Conversation.last_updated.desc())
            .limit(request.limit)
        )

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()

        return ConversationListResult(
            conversations=tuple(
                ConversationInfo(
                    conversation_id=row.conversation_id,
                    title=row.title,
                    created_at=row.created_at,
                    last_updated=row.last_updated,
                )
                for row in rows
            )
        )

    async def get_conversation(
        self,
        request: ConversationReadRequest,
    ) -> ConversationReadResult:
        """Read one conversation the user owns, with its messages."""
        conversation_statement = select(Conversation).where(
            Conversation.conversation_id == request.conversation_id,
            # Ownership scoping: another user's conversation reads as absent.
            Conversation.user_id == request.user_id,
        )
        message_statement = (
            select(ConversationMessage)
            .where(ConversationMessage.conversation_id == request.conversation_id)
            .order_by(
                ConversationMessage.created_at.asc(),
                ConversationMessage.message_id.asc(),
            )
        )

        async with self._session_factory() as session:
            conversation = (
                await session.execute(conversation_statement)
            ).scalar_one_or_none()
            if conversation is None:
                return ConversationReadResult()

            messages = (await session.execute(message_statement)).scalars().all()

        return ConversationReadResult(
            conversation=ConversationEntity(
                conversation_id=conversation.conversation_id,
                user_id=conversation.user_id,
                title=conversation.title,
                messages=tuple(self._to_entity(message) for message in messages),
                created_at=conversation.created_at,
                updated_at=conversation.last_updated,
                metadata=conversation.metadata_ or {},
            )
        )

    @staticmethod
    def _to_entity(row: ConversationMessage) -> ConversationMessageEntity:
        """Map one stored message row onto the domain entity."""
        return ConversationMessageEntity(
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            message_id=row.message_id,
            user_id=row.user_id,
            created_at=row.created_at,
            metadata=row.metadata_ or {},
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
