"""Application-facing model catalog contract."""

from __future__ import annotations

from typing import Protocol

from application.models.schemas import ModelListResult


class PortListModels(Protocol):
    """Lists models available through an interchangeable model gateway."""

    async def list_models(self) -> ModelListResult:
        """Return the model names currently advertised by the gateway."""
        ...
