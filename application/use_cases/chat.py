"""Chat use case orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from application.llm.llm_port import LLMPort
from application.llm.schemas import ChatMessage, ChatRequest


@dataclass(frozen=True, slots=True)
class ChatCommand:
    """Input needed to generate one assistant response."""

    message: str
    model: str
    temperature: float = 0.7
    max_tokens: int | None = 256


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Assistant response returned by the chat use case."""

    content: str
    usage: Mapping[str, Any] | None = None


class ChatUseCase:
    """Coordinates a simple user-message-to-LLM chat flow."""

    def __init__(self, llm: LLMPort) -> None:
        """Create the use case with an LLM port implementation."""
        self._llm = llm

    async def execute(self, command: ChatCommand) -> ChatResult:
        """Generate an assistant response for a single user message."""
        # Build the provider-agnostic LLM request expected by the LLM port.
        request = ChatRequest(
            model=command.model,
            messages=(ChatMessage(role="user", content=command.message),),
            temperature=command.temperature,
            max_tokens=command.max_tokens,
        )

        # Delegate provider-specific execution to the infrastructure adapter.
        response = await self._llm.chat(request)

        return ChatResult(content=response.content, usage=response.usage)
