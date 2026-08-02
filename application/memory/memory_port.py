"""Application-facing protocol for durable memory storage."""

from __future__ import annotations

from typing import Protocol

from application.memory.schemas import (
    MemoryRetrieveRequest,
    MemoryRetrieveResult,
    MemorySaveRequest,
    MemorySaveResult,
)


class MemoryPort(Protocol):
    """Application boundary implemented by concrete memory stores."""

    async def save(self, request: MemorySaveRequest) -> MemorySaveResult:
        """Infer and persist durable memories from one completed turn."""
        ...

    async def retrieve(self, request: MemoryRetrieveRequest) -> MemoryRetrieveResult:
        """Retrieve memories relevant to a query or conversation."""
        ...
