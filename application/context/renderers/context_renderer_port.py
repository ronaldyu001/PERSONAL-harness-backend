"""Application-facing protocol for rendering context into LLM requests."""

from __future__ import annotations

from typing import Protocol

from application.context.renderers.schemas import ContextRenderOptions
from application.context.schemas import ApplicationContext
from application.llm.schemas import ChatRequest


class ContextRendererPort(Protocol):
    """Application boundary implemented by concrete context renderers."""

    async def render(
        self,
        context: ApplicationContext,
        options: ContextRenderOptions,
    ) -> ChatRequest:
        """Render structured context into a provider-agnostic chat request."""
        ...
