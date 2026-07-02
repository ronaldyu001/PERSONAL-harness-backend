"""Provider-agnostic chat request and response models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


ChatRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat conversation.

    The role identifies who produced the message, and content contains the text
    sent to or returned from the model.
    """

    role: ChatRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Provider-agnostic request for generating a chat response."""

    model: str
    messages: Sequence[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """Provider-agnostic response returned by an LLM chat implementation."""

    content: str
    usage: Mapping[str, Any] | None = None
