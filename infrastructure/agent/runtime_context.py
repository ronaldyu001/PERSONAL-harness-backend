"""Typed runtime context passed to LangChain agent middleware."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """Immutable identity available during one agent invocation."""

    user_id: str
    session_id: str
