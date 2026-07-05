"""Schemas used by context renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ContextRenderOptions:
    """Provider-agnostic options for rendering context into a chat request."""

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
