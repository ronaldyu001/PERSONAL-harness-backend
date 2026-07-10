"""Mem0 history database configuration builder."""

from __future__ import annotations

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings


def build_history_db_path(settings: Mem0Settings) -> str:
    """Return the SQLite history database path used by Mem0."""
    return settings.history_db_path
