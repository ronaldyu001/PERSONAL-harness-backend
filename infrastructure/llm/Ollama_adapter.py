"""Ollama LLM adapter."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from ollama import AsyncClient

from Harness.application.llm.schemas import ChatRequest, ChatResponse


# Load local .env values at module import so engine configuration is defined in
# one top-level block. Production should usually inject these as real env vars.
load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))


class OllamaAdapter:
    """Calls an Ollama chat endpoint using the application LLM port types."""

    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        """Create a reusable async Ollama client."""
        # The Ollama SDK owns HTTP details; this class only maps application
        # models to and from the SDK request and response shapes.
        self._client = AsyncClient(
            host=self._normalize_host(host),
            timeout=timeout,
        )

    @classmethod
    def from_env(cls) -> OllamaAdapter:
        """Alternate constructor to build the adapter from environment variables.

        Expected variables:
            OLLAMA_BASE_URL: Optional Ollama host, defaults to localhost:11434.
            OLLAMA_TIMEOUT: Optional request timeout in seconds.
        """
        return cls(
            host=OLLAMA_BASE_URL,
            timeout=OLLAMA_TIMEOUT,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Ollama and map the response to the port type."""
        # Convert application-level dataclasses into the message dictionaries
        # expected by the Ollama SDK.
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ]

        # Ollama model parameters live under options. num_predict is Ollama's
        # equivalent of a max output token limit.
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        response = await self._client.chat(
            model=request.model,
            messages=messages,
            options=options,
            stream=False,
        )

        # Keep Ollama SDK objects inside the infrastructure layer and return the
        # project-owned response dataclass to the application layer.
        return ChatResponse(
            content=response["message"]["content"],
            usage=self._usage_from_response(response),
        )

    @staticmethod
    def _normalize_host(host: str) -> str:
        """Return the host URL expected by the Ollama SDK."""
        normalized = host.rstrip("/")

        # If someone passes an API route, reduce it to the server host because
        # the SDK adds the route itself.
        for suffix in ("/api/chat", "/api/generate"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]

        return normalized

    @staticmethod
    def _usage_from_response(response: Any) -> dict[str, int] | None:
        """Extract token usage fields from an Ollama chat response."""
        usage_fields = (
            "prompt_eval_count",
            "eval_count",
            "prompt_eval_duration",
            "eval_duration",
            "total_duration",
            "load_duration",
        )

        usage = {
            field: response[field]
            for field in usage_fields
            if field in response and response[field] is not None
        }

        return usage or None
