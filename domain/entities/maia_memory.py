"""Maia memory domain entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping, Self


MemoryKind = Literal["fact", "preference", "summary", "instruction", "other"]


@dataclass(frozen=True, slots=True)
class Memory:
    """Durable fact or preference retained for future context building."""

    content: str
    user_id: str
    kind: MemoryKind = "other"
    memory_id: str | None = None
    conversation_id: str | None = None
    source_message_ids: tuple[str, ...] = ()
    confidence: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required memory invariants."""
        if not self.content.strip():
            raise ValueError("content is required")
        if not self.user_id.strip():
            raise ValueError("user_id is required")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def with_identity(self, memory_id: str) -> Self:
        """Return a copy with a persistent storage identity assigned."""
        if not memory_id.strip():
            raise ValueError("memory_id is required")

        return type(self)(
            content=self.content,
            user_id=self.user_id,
            kind=self.kind,
            memory_id=memory_id,
            conversation_id=self.conversation_id,
            source_message_ids=self.source_message_ids,
            confidence=self.confidence,
            created_at=self.created_at,
            updated_at=self.updated_at,
            metadata=self.metadata,
        )

    def retrieval_attributes(self) -> dict[str, Any]:
        """Return flat attributes commonly used to scope memory retrieval."""
        attributes = {
            "kind": self.kind,
            "conversation_id": self.conversation_id,
            "source_message_ids": list(self.source_message_ids),
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            **self.metadata,
        }
        return {key: value for key, value in attributes.items() if value is not None}
