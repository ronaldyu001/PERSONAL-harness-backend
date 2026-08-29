"""HTTP routes for the API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status

from application.agent import EmptyAgentResponseError
from application.conversation import (
    ConversationListRequest,
    ConversationReadRequest,
)
from application.observability import (
    ModelContextTrace,
    Trace,
    TraceReadRequest,
)
from application.use_cases import (
    ChatCommand,
    ChatResult,
    ChatUseCase,
    ReadConversationHistoryUseCase,
    ReadTracesUseCase,
)
from presentation.api.schemas import (
    ChatRequestBody,
    ChatResponseBody,
    ConversationBody,
    ConversationInfoBody,
    ConversationMessageBody,
    HealthResponseBody,
    ModelContextEventBody,
    ResponseGateEventBody,
    TraceRecordBody,
    TraceStreamBody,
)


router = APIRouter(prefix="/api", tags=["chat"])

TraceStreamName = Literal["model-context", "response-gate"]

# Named so a reader can go find the records themselves.
_TRACE_SOURCES: dict[TraceStreamName, str] = {
    "model-context": "model_context_events",
    "response-gate": "response_gate_events",
}


def _chat_use_case(request: Request) -> ChatUseCase:
    """Return the app-scoped chat use case."""
    return request.app.state.chat_use_case


def _read_history_use_case(request: Request) -> ReadConversationHistoryUseCase | None:
    """Return the history use case, or nothing when persistence is off."""
    return getattr(request.app.state, "read_conversation_history_use_case", None)


def _read_traces_use_case(request: Request) -> ReadTracesUseCase | None:
    """Return the trace read use case, or nothing when no sink is wired."""
    return getattr(request.app.state, "read_traces_use_case", None)


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


@router.get("/traces", response_model=TraceStreamBody, tags=["traces"])
async def read_traces(
    request: Request,
    stream: TraceStreamName,
    user_id: str = Query(..., min_length=1),
    session_id: str | None = Query(None, min_length=1),
    invocation_id: str | None = Query(None, min_length=1),
    # Omitted means "use the configured default"; the use case also applies
    # the configured ceiling.
    limit: int | None = Query(None, gt=0),
) -> TraceStreamBody:
    """Read one of Maia's trace streams, most recent first."""
    use_case = _read_traces_use_case(request)
    # An unwired sink reads as empty rather than as a failure: nothing is
    # broken, there is just nothing recorded to read.
    if use_case is None:
        return TraceStreamBody(stream=stream, source=_TRACE_SOURCES[stream])

    result = await use_case.read(
        TraceReadRequest(
            stream=stream,
            user_id=user_id,
            session_id=session_id,
            invocation_id=invocation_id,
            limit=limit,
        )
    )

    return TraceStreamBody(
        stream=stream,
        source=_TRACE_SOURCES[stream],
        records=[_to_trace_record(record) for record in result.records],
        # When this response was produced, which is a fact about the response.
        captured_at=datetime.now(UTC),
    )


def _to_trace_record(trace: Trace) -> TraceRecordBody:
    """Map one recorded trace onto its wire body."""
    if isinstance(trace, ModelContextTrace):
        event = ModelContextEventBody(
            timestamp=trace.occurred_at,
            invocation_id=trace.invocation_id,
            session_id=trace.session_id,
            user_id=trace.user_id,
            model=trace.model,
            mode=trace.mode,
            model_call=trace.model_call,
            system_message=(
                dict(trace.system_message)
                if trace.system_message is not None
                else None
            ),
            messages=[dict(message) for message in trace.messages],
            tools=[dict(tool) for tool in trace.tools],
            status=trace.status,
            usage=dict(trace.usage) if trace.usage is not None else None,
        )
    else:
        event = ResponseGateEventBody(
            timestamp=trace.occurred_at,
            invocation_id=trace.invocation_id,
            session_id=trace.session_id,
            user_id=trace.user_id,
            model=trace.model,
            mode=trace.mode,
            evaluation_call=trace.evaluation_call,
            repair_attempt=trace.repair_attempt,
            decision=trace.decision,
            passed=trace.passed,
            violations=list(trace.violations),
            feedback=trace.feedback,
            candidate_message_id=trace.candidate_message_id,
            candidate_characters=trace.candidate_characters,
            candidate=trace.candidate,
            available_tools=list(trace.available_tools),
            tools_used=list(trace.tools_used),
            gate_context=(
                dict(trace.gate_context)
                if trace.gate_context is not None
                else None
            ),
            usage=dict(trace.usage) if trace.usage is not None else None,
            error_type=trace.error_type,
            error_message=trace.error_message,
        )

    # The sink assigns the id; a trace that reached a reader always has one.
    return TraceRecordBody(id=trace.event_id or "", event=event)
