"""SQLAlchemy models for Maia's persisted conversations.

The declarative attribute for each ``metadata`` column is named ``metadata_``
because Declarative reserves ``metadata`` on the class itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by every Maia table."""


class Conversation(Base):
    """One conversation timeline, keyed by the agent's session id."""

    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(
        String(255), 
        primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        String(255), 
        nullable=False
    )
    title: Mapped[str] = mapped_column(
        Text, 
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    summary: Mapped[str | None] = mapped_column(
        Text, 
        nullable=True
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )


class ConversationMessage(Base):
    """One immutable message belonging to a conversation."""

    __tablename__ = "conversation_messages"

    # The agent's message id, so re-running a turn cannot duplicate a row.
    message_id: Mapped[str] = mapped_column(
        String(255), 
        primary_key=True
    )
    conversation_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Optional because not every message originates from the user.
    user_id: Mapped[str | None] = mapped_column(
        String(255), 
        nullable=True
    )
    role: Mapped[str] = mapped_column(
        String(32), 
        nullable=False
    )
    content: Mapped[str] = mapped_column(
        Text, 
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=True,
    )

    __table_args__ = (
        # Reading a transcript means ordering one conversation by time.
        Index(
            "ix_conversation_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )
