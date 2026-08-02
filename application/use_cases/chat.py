"""Chat use case orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

from application.agent import AgentPort, EmptyAgentResponseError
from application.llm.schemas import ChatMessage, ChatRequest


@dataclass(frozen=True, slots=True)
class ChatCommand:
    """Input needed to generate one assistant response."""

    message: str
    model: str
    user_id: str
    session_id: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = 1024


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Assistant response returned by the chat use case."""

    content: str
    session_id: str
    usage: Mapping[str, Any] | None = None
    finish_reason: str | None = None


class ChatUseCase:
    """Coordinates a user message through a conversational agent."""

    def __init__(self, agent: AgentPort) -> None:
        """Create the use case with an agent port implementation."""
        self._agent = agent

    async def execute(self, command: ChatCommand) -> ChatResult:
        """Generate an assistant response for a single user message."""
        session_id = command.session_id or str(uuid4())

        # Build the provider-agnostic request expected by the agent port.
        request = ChatRequest(
            model=command.model,
            messages=(ChatMessage(role="user", content=command.message),),
            temperature=command.temperature,
            max_tokens=command.max_tokens,
        )

        # Delegate agent execution to the infrastructure adapter.
        response = await self._agent.chat(
            request,
            session_id=session_id,
            user_id=command.user_id,
        )
        if not response.content.strip():
            raise EmptyAgentResponseError(
                "Agent completed without user-visible content."
            )

        return ChatResult(
            content=response.content,
            session_id=session_id,
            usage=response.usage,
            finish_reason=response.finish_reason,
        )
