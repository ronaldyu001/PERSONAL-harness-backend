"""Application schemas for Maia's agent traces.

A trace is one line the agent recorded about a turn. The two streams are kept
apart rather than unioned: they share an invocation and little else, and a
reader asks for one or the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping


TraceStream = Literal["model-context", "response-gate"]

# A recorded trace is never "off"; that mode means nothing was recorded.
TraceMode = Literal["structure", "full"]

GateDecision = Literal["allow", "retry", "fallback", "allow_on_error"]


def _as_utc(value: datetime, *, name: str) -> datetime:
    """Return an aware instant in UTC, or reject a naive one."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ModelContextTrace:
    """One effective model request and its completion metadata."""

    invocation_id: str
    occurred_at: datetime
    model: str
    mode: TraceMode
    model_call: int
    # Absent on a temporary turn, which writes no conversation to belong to.
    session_id: str | None = None
    user_id: str | None = None
    system_message: Mapping[str, Any] | None = None
    messages: tuple[Mapping[str, Any], ...] = ()
    tools: tuple[Mapping[str, Any], ...] = ()
    status: Literal["success", "error"] | None = None
    usage: Mapping[str, Any] | None = None
    # Assigned by the sink; absent on the way in.
    event_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the identity a reader groups and orders records by."""
        if not self.invocation_id.strip():
            raise ValueError("invocation_id is required")
        if self.model_call < 1:
            raise ValueError("model_call must be positive")
        object.__setattr__(
            self,
            "occurred_at",
            _as_utc(self.occurred_at, name="occurred_at"),
        )


@dataclass(frozen=True, slots=True)
class ResponseGateTrace:
    """One response-gate evaluation and routing decision."""

    invocation_id: str
    occurred_at: datetime
    model: str
    mode: TraceMode
    evaluation_call: int
    repair_attempt: int
    decision: GateDecision
    candidate_characters: int
    # Tri-state: ``None`` is the gate erroring, not a missing value.
    passed: bool | None = None
    session_id: str | None = None
    user_id: str | None = None
    violations: tuple[str, ...] = ()
    feedback: str | None = None
    candidate_message_id: str | None = None
    candidate: str | None = None
    available_tools: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    # What the evaluator was actually given about the turn: the rubric it
    # judged by, the system prompt and memories in force, the conversation
    # window, and the tool evidence as budgeted for it. Absent in structure
    # mode, which keeps the decision and drops the text.
    gate_context: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    # Assigned by the sink; absent on the way in.
    event_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the identity a reader groups and orders records by."""
        if not self.invocation_id.strip():
            raise ValueError("invocation_id is required")
        if self.evaluation_call < 1:
            raise ValueError("evaluation_call must be positive")
        if self.repair_attempt < 0:
            raise ValueError("repair_attempt must not be negative")
        if self.candidate_characters < 0:
            raise ValueError("candidate_characters must not be negative")
        object.__setattr__(
            self,
            "occurred_at",
            _as_utc(self.occurred_at, name="occurred_at"),
        )


Trace = ModelContextTrace | ResponseGateTrace


@dataclass(frozen=True, slots=True)
class ModelContextWriteRequest:
    """Request to record one model-context trace."""

    trace: ModelContextTrace


@dataclass(frozen=True, slots=True)
class ResponseGateWriteRequest:
    """Request to record one response-gate trace."""

    trace: ResponseGateTrace


@dataclass(frozen=True, slots=True)
class TraceWriteResult:
    """Result of recording one trace."""

    stream: TraceStream
    event_id: str


@dataclass(frozen=True, slots=True)
class TraceReadRequest:
    """Request one stream of traces for a user.

    ``limit`` of ``None`` means the caller did not ask for a count; the use
    case resolves it from configured policy.
    """

    stream: TraceStream
    user_id: str
    session_id: str | None = None
    invocation_id: str | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        """Validate the window and the identifiers a read is scoped to."""
        if not self.user_id.strip():
            raise ValueError("user_id is required")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("session_id must not be blank")
        if self.invocation_id is not None and not self.invocation_id.strip():
            raise ValueError("invocation_id must not be blank")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")


@dataclass(frozen=True, slots=True)
class TraceReadResult:
    """Traces from one stream, most recent first."""

    stream: TraceStream
    records: tuple[Trace, ...] = ()
