"""Application-facing protocol for building structured assistant context."""

from __future__ import annotations

from typing import Protocol

from application.context.schemas import ApplicationContext


class ContextBuilderPort(Protocol):
    """Application boundary implemented by concrete context builders."""

    async def build(self, *args: object, **kwargs: object) -> ApplicationContext:
        """Build structured context for one assistant turn."""
        ...
