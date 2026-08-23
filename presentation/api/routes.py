"""HTTP routes for the API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request, status

from application.agent import EmptyAgentResponseError
from application.conversation import (
    ConversationListRequest,
    ConversationReadRequest,
)
from application.use_cases import (
    ChatCommand,
    ChatResult,
    ChatUseCase,
    ReadConversationHistoryUseCase,
)
from presentation.api.schemas import (
    ChatRequestBody,
    ChatResponseBody,
    ConversationBody,
    ConversationInfoBody,
    ConversationMessageBody,
    HealthResponseBody,
)


router = APIRouter(prefix="/api", tags=["chat"])


def _chat_use_case(request: Request) -> ChatUseCase:
    """Return the app-scoped chat use case."""
    return request.app.state.chat_use_case


def _read_history_use_case(request: Request) -> ReadConversationHistoryUseCase | None:
    """Return the history use case, or nothing when persistence is off."""
    return getattr(request.app.state, "read_conversation_history_use_case", None)


@router.get("/health", response_model=HealthResponseBody)
async def health() -> HealthResponseBody:
    """Return API health status."""
    return HealthResponseBody(status="ok")


@router.post("/chat", response_model=ChatResponseBody)
async def chat(body: ChatRequestBody, request: Request) -> ChatResponseBody:
    """Generate an assistant response for one user message."""
    return await _run_chat(body, request, temporary=False)


@router.post("/temp-chat", response_model=ChatResponseBody)
async def temp_chat(body: ChatRequestBody, request: Request) -> ChatResponseBody:
    """Answer one message without persisting the turn or learning from it."""
    return await _run_chat(body, request, temporary=True)


async def _run_chat(
    body: ChatRequestBody,
    request: Request,
    *,
    temporary: bool,
) -> ChatResponseBody:
    """Run one chat turn; the route decides whether it is remembered."""
    # Convert HTTP input into the application use-case command.
    command = ChatCommand(
        message=body.message,
        model=body.model,
        user_id=body.user_id,
        session_id=body.session_id,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        temporary=temporary,
    )

    # Run the application use case and map the result back to HTTP.
    try:
        result: ChatResult = await _chat_use_case(request).execute(command)
    except EmptyAgentResponseError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The agent returned an empty response.",
        ) from error

    return ChatResponseBody(
        content=result.content,
        session_id=result.session_id,
        usage=result.usage,
        finish_reason=result.finish_reason,
    )


@router.get("/conversations", response_model=list[ConversationInfoBody])
async def list_conversations(
    request: Request,
    user_id: str = Query(..., min_length=1),
    # Omitted means "use the configured default"; the use case also applies
    # the configured ceiling.
    limit: int | None = Query(None, gt=0),
) -> list[ConversationInfoBody]:
    """List a user's conversations, most recently active first."""
    use_case = _read_history_use_case(request)
    # An unwired history reads as empty rather than as a failure.
    if use_case is None:
        return []

    result = await use_case.list(
        ConversationListRequest(user_id=user_id, limit=limit)
    )

    return [
        ConversationInfoBody(
            conversation_id=info.conversation_id,
            title=info.title,
            created_at=info.created_at,
            last_updated=info.last_updated,
        )
        for info in result.conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationBody)
async def get_conversation(
    conversation_id: str,
    request: Request,
    user_id: str = Query(..., min_length=1),
) -> ConversationBody:
    """Return one conversation the user owns, with its messages."""
    use_case = _read_history_use_case(request)
    if use_case is not None:
        result = await use_case.get(
            ConversationReadRequest(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        )
        conversation = result.conversation
    else:
        conversation = None

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationBody(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        created_at=conversation.created_at,
        last_updated=conversation.last_activity_at,
        messages=[
            ConversationMessageBody(
                message_id=message.message_id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
                metadata=dict(message.metadata),
            )
            for message in conversation.messages
        ],
    )
