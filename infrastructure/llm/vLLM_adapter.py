"""OpenAI-compatible vLLM LLM adapter."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from Harness.application.llm.schemas import ChatRequest, ChatResponse


# Load local .env values at module import so engine configuration is defined in
# one top-level block. Production should usually inject these as real env vars.
load_dotenv()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "60"))
VLLM_MAX_RETRIES = int(os.getenv("VLLM_MAX_RETRIES", "2"))


class VLLMAdapter:
    """Calls a vLLM OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        """Create a reusable async OpenAI SDK client for vLLM."""
        # vLLM exposes an OpenAI-compatible API, so the OpenAI SDK can target it
        # by swapping the base URL away from the hosted OpenAI API.
        self._client = AsyncOpenAI(
            base_url=self._normalize_base_url(base_url),
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(cls) -> VLLMAdapter:
        """Alternate constructor to build the adapter from environment variables.

        Expected variables:
            VLLM_BASE_URL: Base vLLM URL, ideally ending in /v1.
            VLLM_API_KEY: Optional API key if the vLLM server requires one.
            VLLM_TIMEOUT: Optional request timeout in seconds.
            VLLM_MAX_RETRIES: Optional OpenAI SDK retry count.
        """
        # Fail fast on the required endpoint instead of surfacing a vague SDK
        # error after the first request.
        if not VLLM_BASE_URL:
            raise RuntimeError("VLLM_BASE_URL must be set.")

        # Local vLLM deployments commonly accept any API key, so EMPTY is a
        # practical default unless the server has auth configured.
        return cls(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            timeout=VLLM_TIMEOUT,
            max_retries=VLLM_MAX_RETRIES,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to vLLM and map the SDK response to the port type."""
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

        # Only include optional arguments when present so SDK defaults and vLLM
        # server defaults can still apply.
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

        # Accept host-only URLs for convenience while still giving the SDK the
        # OpenAI-compatible API root.
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"

        return normalized
