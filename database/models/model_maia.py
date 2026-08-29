"""SQLAlchemy models for everything Maia keeps in Postgres.

Conversations are the transcript; the two event tables are what the agent
recorded about the turns that produced it.

The declarative attribute for each ``metadata`` column is named ``metadata_``
because Declarative reserves ``metadata`` on the class itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
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


class ModelContextEvent(Base):
    """One effective model request and its completion metadata.

    ``session_id`` is nullable because a temporary turn writes no conversation
    row to reference; every other turn writes one before its first model call.
    """

    __tablename__ = "model_context_events"

    # The log line has no id of its own; one is minted per row so a reader can
    # key a list on it and hold a selection across a refresh.
    event_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    invocation_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=True,
    )
    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )
    model: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    # Plain text, not a PG enum: there are no migrations here, and extending an
    # enum needs ALTER TYPE, which create_all will never run.
    mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False
    )
    model_call: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    system_message: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    messages: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False
    )
    tools: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False
    )
    status: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True
    )
    usage: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )

    __table_args__ = (
        # The grain of the table, and idempotency for a retried write.
        UniqueConstraint(
            "invocation_id",
            "model_call",
            name="uq_model_context_events_invocation_call",
        ),
        # The bench opened on one conversation, newest first. Ascending is
        # enough; Postgres scans a btree index backwards.
        Index(
            "ix_model_context_events_session_occurred",
            "session_id",
            "occurred_at",
        ),
        Index(
            "ix_model_context_events_user_occurred",
            "user_id",
            "occurred_at",
        ),
    )


class ResponseGateEvent(Base):
    """One response-gate evaluation and routing decision."""

    __tablename__ = "response_gate_events"

    event_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    invocation_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=True,
    )
    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )
    model: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False
    )
    evaluation_call: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    repair_attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    decision: Mapped[str] = mapped_column(
        String(32),
        nullable=False
    )
    # Tri-state and must stay so: NULL is the gate erroring, not a missing value.
    passed: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True
    )
    violations: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False
    )
    feedback: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    candidate_message_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    candidate_characters: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )
    candidate: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )
    available_tools: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False
    )
    tools_used: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False
    )
    # The evaluator's view of the turn. Nullable: structure mode records no
    # text, and rows written before the gate kept its context have none.
    gate_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    usage: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True
    )
    error_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    __table_args__ = (
        UniqueConstraint(
            "invocation_id",
            "evaluation_call",
            name="uq_response_gate_events_invocation_call",
        ),
        Index(
            "ix_response_gate_events_session_occurred",
            "session_id",
            "occurred_at",
        ),
        Index(
            "ix_response_gate_events_user_occurred",
            "user_id",
            "occurred_at",
        ),
    )
