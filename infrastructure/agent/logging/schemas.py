"""Structured schemas for Maia's local JSONL logs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelContextLogEvent(BaseModel):
    """One effective model request and its completion metadata."""

    event: Literal["model_context"] = "model_context"
    timestamp: str
    invocation_id: str
    session_id: str | None = None
    model: str
    mode: Literal["structure", "full"]
    model_call: int = Field(ge=1)
    system_message: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["success", "error"] | None = None
    usage: dict[str, Any] | None = None


class ResponseGateLogEvent(BaseModel):
    """One response-gate evaluation and routing decision."""

    event: Literal["response_gate"] = "response_gate"
    timestamp: str
    invocation_id: str
    session_id: str | None = None
    model: str
    mode: Literal["structure", "full"]
    evaluation_call: int = Field(ge=1)
    repair_attempt: int = Field(ge=0)
    decision: Literal["allow", "retry", "fallback", "allow_on_error"]
    passed: bool | None
    violations: list[str] = Field(default_factory=list)
    feedback: str | None = None
    candidate_message_id: str | None = None
    candidate_characters: int = Field(ge=0)
    candidate: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
