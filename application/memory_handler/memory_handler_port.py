"""Application-facing protocol for post-response memory handling."""

from __future__ import annotations

from typing import Protocol

from application.memory_handler.schemas import MemoryDigestRequest, MemoryDigestResult


class MemoryHandlerPort(Protocol):
    """Application boundary for deciding whether a completed turn becomes memory."""

    async def digest(self, request: MemoryDigestRequest) -> MemoryDigestResult:
        """Inspect a completed turn and return memory candidates, if any."""
        ...
