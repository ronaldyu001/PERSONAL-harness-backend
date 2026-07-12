"""Mem0 reranker configuration builder."""

from __future__ import annotations

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings


def build_reranker_config(settings: Mem0Settings) -> None:
    """Build Mem0's optional reranker config."""
    return None
