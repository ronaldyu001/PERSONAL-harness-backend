"""Shared configuration helpers for local agent logs."""

from __future__ import annotations

from pathlib import Path


_OFF_VALUES = {"", "0", "false", "no", "off"}
_FULL_VALUES = {"1", "true", "yes", "on", "full"}
_VALID_MODES = {"off", "structure", "full"}


def normalize_log_mode(value: str, *, env_name: str) -> str:
    """Normalize a logging mode or raise an environment-specific error."""
    normalized = value.strip().lower()
    if normalized in _OFF_VALUES:
        return "off"
    if normalized in _FULL_VALUES:
        return "full"
    if normalized not in _VALID_MODES:
        valid = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"{env_name} must be one of: {valid}")
    return normalized


def resolve_log_dir(value: str | Path | None) -> Path:
    """Resolve an explicit log directory or use the backend's `.logs` folder."""
    if value is not None and str(value).strip():
        return Path(value).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[3]
    return project_root / ".logs"
