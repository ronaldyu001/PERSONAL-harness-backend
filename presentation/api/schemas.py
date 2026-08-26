"""Request and response bodies for the HTTP API.

These are the wire contract: snake_case field names the frontend adapter maps
to camelCase. Application schemas are converted to and from these in
``presentation.api.routes``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field


class ChatRequestBody(BaseModel):
    """Request body for a single chat turn."""

    message: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024, gt=0)


class ChatResponseBody(BaseModel):
    """Response body for a single chat turn."""

    content: str
    session_id: str
    usage: Mapping[str, Any] | None = None
    finish_reason: str | None = None


class HealthResponseBody(BaseModel):
    """Response body for a health check."""

    status: str


class ConversationInfoBody(BaseModel):
    """One conversation without its messages."""

    conversation_id: str
    title: str
    created_at: datetime
    last_updated: datetime


class ConversationMessageBody(BaseModel):
    """One stored message in a conversation timeline."""

    message_id: str | None = None
    role: str
    content: str
    created_at: datetime | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ConversationBody(BaseModel):
    """One conversation with its messages."""

    conversation_id: str
    title: str | None = None
    created_at: datetime | None = None
    last_updated: datetime | None = None
    messages: list[ConversationMessageBody] = Field(default_factory=list)


class ModelContextEventBody(BaseModel):
    """One model-context trace, as the agent recorded it."""

    event: Literal["model_context"] = "model_context"
    timestamp: datetime
    invocation_id: str
    session_id: str | None = None
    user_id: str | None = None
    model: str
    mode: Literal["structure", "full"]
    model_call: int
    system_message: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["success", "error"] | None = None
    usage: dict[str, Any] | None = None


class ResponseGateEventBody(BaseModel):
    """One response-gate trace, as the agent recorded it."""

    event: Literal["response_gate"] = "response_gate"
    timestamp: datetime
    invocation_id: str
    session_id: str | None = None
    user_id: str | None = None
    model: str
    mode: Literal["structure", "full"]
    evaluation_call: int
    repair_attempt: int
    decision: Literal["allow", "retry", "fallback", "allow_on_error"]
    passed: bool | None = None
    violations: list[str] = Field(default_factory=list)
    feedback: str | None = None
    candidate_message_id: str | None = None
    candidate_characters: int
    candidate: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


class TraceRecordBody(BaseModel):
    """One trace and where it came from."""

    id: str
    # The API only ever serves recorded lines; seeded ones were fixtures.
    origin: Literal["captured"] = "captured"
    event: ModelContextEventBody | ResponseGateEventBody = Field(
        discriminator="event"
    )


class TraceStreamBody(BaseModel):
    """One stream of traces, most recent first."""

    stream: Literal["model-context", "response-gate"]
    # Where the records were read from, so a reader can go find them.
    source: str
    records: list[TraceRecordBody] = Field(default_factory=list)
    captured_at: datetime | None = None
