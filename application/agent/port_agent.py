"""Application-facing conversational agent contract."""

from __future__ import annotations

from typing import Protocol

from application.llm import ChatRequest, ChatResponse


class AgentPort(Protocol):
    """Runs a chat request through an agent implementation."""

    async def chat(
        self,
        request: ChatRequest,
        *,
        session_id: str,
        user_id: str,
        temporary: bool = False,
    ) -> ChatResponse:
        """Generate a response within an application chat session."""
        ...
