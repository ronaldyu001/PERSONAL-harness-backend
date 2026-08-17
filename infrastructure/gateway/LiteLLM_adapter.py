"""OpenAI-compatible LiteLLM gateway implementation."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from application.llm.schemas import ChatRequest, ChatResponse
from infrastructure.settings import GatewayConfig


class LiteLLMAdapter:
    """Calls a LiteLLM proxy endpoint using the application LLM port types."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        """Create a reusable async OpenAI SDK client for LiteLLM."""
        # LiteLLM proxy exposes an OpenAI-compatible API, so the OpenAI SDK can
        # target the container endpoint by changing only the base URL.
        self._client = AsyncOpenAI(
            base_url=self._normalize_base_url(base_url),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_config(cls, config: GatewayConfig) -> LiteLLMAdapter:
        """Build the adapter from its resolved configuration section."""
        return cls(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to LiteLLM and map the response to the port type."""
        # Convert application-level dataclasses into the dictionary payload the
        # OpenAI SDK expects for chat completions.
        create_args: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
        }

        # Only include optional arguments when present so LiteLLM/provider
        # defaults can still apply.
        if request.max_tokens is not None:
            create_args["max_tokens"] = request.max_tokens

        # The SDK appends /chat/completions to the configured /v1 base URL.
        response = await self._client.chat.completions.create(**create_args)
        message = response.choices[0].message

        # Keep OpenAI SDK objects inside the infrastructure layer and return the
        # project-owned response dataclass to the application layer.
        return ChatResponse(
            content=message.content or "",
            usage=response.usage.model_dump() if response.usage else None,
        )

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Return the /v1 base URL expected by the OpenAI SDK."""
        normalized = base_url.rstrip("/")
        chat_completions_suffix = "/chat/completions"

        # If a full chat completions endpoint was provided, reduce it to the API
        # base because the SDK adds the endpoint path itself.
        if normalized.endswith(chat_completions_suffix):
            normalized = normalized[: -len(chat_completions_suffix)]

        # Accept host-only container URLs while still giving the SDK the
        # OpenAI-compatible API root.
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"

        return normalized
