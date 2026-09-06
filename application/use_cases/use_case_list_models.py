"""Available model listing use case."""

from __future__ import annotations

from application.models import ModelListResult, PortListModels


class UseCaseListModels:
    """Read the model catalog through its provider-neutral port."""

    def __init__(self, models: PortListModels) -> None:
        """Create the use case with a model catalog implementation."""
        self._models = models

    async def execute(self) -> ModelListResult:
        """Return the models currently available for chat."""
        return await self._models.list_models()
