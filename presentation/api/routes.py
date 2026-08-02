"""HTTP routes for the API."""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from application.use_cases.chat import ChatCommand, ChatResult, ChatUseCase


router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequestBody(BaseModel):
    """Request body for a single chat turn."""

    message: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=256, gt=0)


class ChatResponseBody(BaseModel):
    """Response body for a single chat turn."""

    content: str
    session_id: str
    usage: Mapping[str, Any] | None = None


class HealthResponseBody(BaseModel):
    """Response body for a health check."""

    status: str


def _chat_use_case(request: Request) -> ChatUseCase:
    """Return the app-scoped chat use case."""
    return request.app.state.chat_use_case


@router.get("/health", response_model=HealthResponseBody)
async def health() -> HealthResponseBody:
    """Return API health status."""
    return HealthResponseBody(status="ok")


@router.post("/chat", response_model=ChatResponseBody)
async def chat(body: ChatRequestBody, request: Request) -> ChatResponseBody:
    """Generate an assistant response for one user message."""
    # Convert HTTP input into the application use-case command.
    command = ChatCommand(
        message=body.message,
        model=body.model,
        user_id=body.user_id,
        session_id=body.session_id,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )

    # Run the application use case and map the result back to HTTP.
    result: ChatResult = await _chat_use_case(request).execute(command)

    return ChatResponseBody(
        content=result.content,
        session_id=result.session_id,
        usage=result.usage,
    )
