"""Application-facing conversational agent contract."""

from __future__ import annotations

from typing import Protocol

from application.agent.schemas import AgentRequest, AgentResponse


class AgentPort(Protocol):
    """Runs a chat request through an agent implementation."""

    async def chat(
        self,
        request: AgentRequest,
        *,
        session_id: str,
        user_id: str,
        temporary: bool = False,
    ) -> AgentResponse:
        """Generate a response within an application chat session."""
        ...
