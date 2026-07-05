"""Schemas for post-response memory handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from application.context.schemas import ApplicationContext
from application.llm.schemas import ChatMessage, ChatResponse
from application.memory.schemas import MemoryRecord


MemoryDigestDecision = Literal["save", "skip"]


@dataclass(frozen=True, slots=True)
class MemoryDigestRequest:
    """Completed turn data used to decide whether memory should be updated."""

    user_message: ChatMessage
    assistant_response: ChatResponse
    context: ApplicationContext | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """Candidate memory produced by digesting a completed turn."""

    memory: MemoryRecord
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryDigestResult:
    """Result of deciding what, if anything, should be saved as memory."""

    decision: MemoryDigestDecision
    candidates: tuple[MemoryCandidate, ...] = ()
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def should_save(self) -> bool:
        """Return whether digest produced memory candidates to save."""
        return self.decision == "save" and bool(self.candidates)
