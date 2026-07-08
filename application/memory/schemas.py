"""Application schemas for durable memory storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from domain.entities.memory import Memory, MemoryKind


@dataclass(frozen=True, slots=True)
class MemorySaveRequest:
    """Request to persist a memory item."""

    memory: Memory
    upsert: bool = True


@dataclass(frozen=True, slots=True)
class MemorySaveResult:
    """Result of persisting a memory item."""

    memory_id: str
    created: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
