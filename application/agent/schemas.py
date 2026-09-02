"""Provider-independent schemas owned by the conversational agent boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


AgentRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class AgentMessage:
    """One message supplied to or returned from a conversational agent."""

    role: AgentRole
    content: str


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Provider-independent request for one agent invocation."""

    model: str
    messages: Sequence[AgentMessage]
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Provider-independent response returned by an agent implementation."""

    content: str
    usage: Mapping[str, Any] | None = None
    finish_reason: str | None = None
