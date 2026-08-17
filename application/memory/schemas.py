"""Application schemas for durable memory storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from application.llm.schemas import ChatMessage, ChatResponse
from domain.entities.memory import Memory, MemoryKind


@dataclass(frozen=True, slots=True)
class MemorySaveRequest:
    """Completed turn submitted for provider-managed memory inference."""

    user_message: ChatMessage
    assistant_response: ChatResponse
    user_id: str
    conversation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySaveResult:
    """Memories inferred and persisted from a completed turn."""

    memories: tuple[Memory, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def saved_count(self) -> int:
        """Return the number of memories inferred from the turn."""
        return len(self.memories)


@dataclass(frozen=True, slots=True)
class MemoryRetrieveRequest:
    """Request to retrieve memories relevant to a query."""

    query: str
    user_id: str | None = None
    conversation_id: str | None = None
    kinds: tuple[MemoryKind, ...] = ()
    limit: int = 10
    min_score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    """A memory returned by retrieval with optional relevance score."""

    memory: Memory
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryRetrieveResult:
    """Result of retrieving memories."""

    memories: tuple[RetrievedMemory, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
