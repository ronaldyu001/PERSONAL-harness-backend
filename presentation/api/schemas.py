"""Request and response bodies for the HTTP API.

These are the wire contract: snake_case field names the frontend adapter maps
to camelCase. Application schemas are converted to and from these in
``presentation.api.routes``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

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
