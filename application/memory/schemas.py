"""Provider-agnostic memory schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping


MemoryKind = Literal["fact", "preference", "summary", "instruction", "other"]


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """Durable memory item stored for future context building."""

    content: str
    user_id: str | None = None
    memory_id: str | None = None
    kind: MemoryKind = "other"
    conversation_id: str | None = None
    source_message_ids: tuple[str, ...] = ()
    confidence: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySaveRequest:
    """Request to persist a memory item."""

    memory: MemoryRecord
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

    memory: MemoryRecord
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryRetrieveResult:
    """Result of retrieving memories."""

    memories: tuple[RetrievedMemory, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
