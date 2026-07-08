"""Application-facing protocol for LLM chat providers."""

from __future__ import annotations

from typing import Protocol

from application.llm.schemas import ChatRequest, ChatResponse


class LLMPort(Protocol):
    """Application boundary implemented by concrete LLM providers."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat response for the supplied request."""
        ...
