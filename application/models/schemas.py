"""Provider-neutral model catalog data contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelListResult:
    """Model names currently advertised by the configured gateway."""

    models: tuple[str, ...] = ()
