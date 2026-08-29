"""JSON Lines implementation of the application observability port.

The fallback sink, used when no database is configured. It reads as well as it
writes, so the endpoint serves the same shape whichever sink is wired.

The private Pydantic models in this module belong solely to this adapter. The
application observability schemas remain the canonical port contract; these
models convert that contract to a stable JSON-safe format and validate records
when they are read back from disk.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, Field

from application.observability import (
    ModelContextTrace,
    ModelContextWriteRequest,
    ResponseGateTrace,
    ResponseGateWriteRequest,
    Trace,
    TraceReadRequest,
    TraceReadResult,
    TraceStream,
    TraceWriteResult,
)
from infrastructure.settings import LoggingConfig


logger = logging.getLogger(__name__)


class _ModelContextLogEvent(BaseModel):
    """JSONL representation of one effective model request."""

    event: Literal["model_context"] = "model_context"
    timestamp: str
    invocation_id: str
    session_id: str | None = None
    # Absent from lines written before traces were scoped to an owner.
    user_id: str | None = None
    model: str
    mode: Literal["structure", "full"]
    model_call: int = Field(ge=1)
    system_message: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["success", "error"] | None = None
    usage: dict[str, Any] | None = None


class _ResponseGateLogEvent(BaseModel):
    """JSONL representation of one response-gate evaluation."""

    event: Literal["response_gate"] = "response_gate"
    timestamp: str
    invocation_id: str
    session_id: str | None = None
    # Absent from lines written before traces were scoped to an owner.
    user_id: str | None = None
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
    # Absent from lines written before the gate kept what it read.
    gate_context: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

CONTEXT_LOG_FILENAME = "agent-context.jsonl"
RESPONSE_GATE_LOG_FILENAME = "response-gate.jsonl"

# One lock for both files: appends are short and contention is not the problem
# being solved, interleaved writes are.
_FILE_LOCK = Lock()


def resolve_log_dir(value: str | Path | None) -> Path:
    """Resolve an explicit log directory or use the backend's `.logs` folder."""
    if value is not None and str(value).strip():
        return Path(value).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".logs"


class JsonlObservabilityAdapter:
    """Append agent traces to JSON Lines files, and read them back."""

    def __init__(
        self,
        *,
        context_dir: str | Path | None = None,
        response_gate_dir: str | Path | None = None,
    ) -> None:
        self._paths: dict[TraceStream, Path] = {
            "model-context": resolve_log_dir(context_dir) / CONTEXT_LOG_FILENAME,
            "response-gate": (
                resolve_log_dir(response_gate_dir) / RESPONSE_GATE_LOG_FILENAME
            ),
        }

    @classmethod
    def from_config(cls, config: LoggingConfig) -> JsonlObservabilityAdapter:
        """Build the file sink from its resolved config section."""
        return cls(
            context_dir=config.context_dir,
            response_gate_dir=config.response_gate_dir,
        )

    def log_path(self, stream: TraceStream) -> Path:
        """Return the file one stream is written to."""
        return self._paths[stream]

    async def record_model_context(
        self,
        request: ModelContextWriteRequest,
    ) -> TraceWriteResult:
        """Record one effective model request and its completion metadata."""
        trace = request.trace
        event = _ModelContextLogEvent(
            timestamp=trace.occurred_at.isoformat(),
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
        return await self._append("model-context", event, trace.model_call)

    async def record_response_gate(
        self,
        request: ResponseGateWriteRequest,
    ) -> TraceWriteResult:
        """Record one gate evaluation and routing decision."""
        trace = request.trace
        event = _ResponseGateLogEvent(
            timestamp=trace.occurred_at.isoformat(),
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
        return await self._append(
            "response-gate", event, trace.evaluation_call
        )

    async def read_traces(self, request: TraceReadRequest) -> TraceReadResult:
        """Read one stream of traces back, most recent first."""
        records = await asyncio.to_thread(self._read_matching, request)
        return TraceReadResult(stream=request.stream, records=records)

    async def _append(
        self,
        stream: TraceStream,
        event: _ModelContextLogEvent | _ResponseGateLogEvent,
        sequence: int,
    ) -> TraceWriteResult:
        """Append one validated line and report the id it can be found by."""
        await asyncio.to_thread(self._append_line, stream, event)
        return TraceWriteResult(
            stream=stream,
            event_id=self._event_id(stream, event.invocation_id, sequence),
        )

    def _append_line(
        self,
        stream: TraceStream,
        event: _ModelContextLogEvent | _ResponseGateLogEvent,
    ) -> None:
        """Write one line, whole, without interleaving with another."""
        path = self._paths[stream]
        line = event.model_dump_json()

        with _FILE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{line}\n")

    def _read_matching(self, request: TraceReadRequest) -> tuple[Trace, ...]:
        """Collect the newest matching traces, filtering before the limit."""
        path = self._paths[request.stream]
        if not path.exists():
            # A stream nothing has written yet is empty, not broken.
            return ()

        with _FILE_LOCK:
            lines = path.read_text(encoding="utf-8").splitlines()

        matches: list[Trace] = []
        # The file is append-only, so the newest line is the last one.
        for number in range(len(lines), 0, -1):
            if request.limit is not None and len(matches) >= request.limit:
                break

            trace = self._parse(request.stream, lines[number - 1], number)
            if trace is None or not self._matches(trace, request):
                continue
            matches.append(trace)

        return tuple(matches)

    def _parse(
        self,
        stream: TraceStream,
        line: str,
        line_number: int,
    ) -> Trace | None:
        """Parse one stored line, or skip it when it cannot be read.

        A half-written final line is normal for a file that is appended to
        while it is being read, so a bad line is not a failed read.
        """
        if not line.strip():
            return None

        try:
            if stream == "model-context":
                event = _ModelContextLogEvent.model_validate_json(line)
                return self._to_model_context(
                    event,
                    self._event_id(stream, event.invocation_id, event.model_call),
                )

            gate_event = _ResponseGateLogEvent.model_validate_json(line)
            return self._to_response_gate(
                gate_event,
                self._event_id(
                    stream,
                    gate_event.invocation_id,
                    gate_event.evaluation_call,
                ),
            )
        except (ValueError, TypeError):
            logger.warning(
                "Skipped an unreadable line %d in %s",
                line_number,
                self._paths[stream],
            )
            return None

    @staticmethod
    def _matches(trace: Trace, request: TraceReadRequest) -> bool:
        """Return whether one trace belongs in this read."""
        if trace.user_id != request.user_id:
            return False
        if request.session_id is not None and trace.session_id != request.session_id:
            return False
        if (
            request.invocation_id is not None
            and trace.invocation_id != request.invocation_id
        ):
            return False
        return True

    @staticmethod
    def _event_id(stream: TraceStream, invocation_id: str, sequence: int) -> str:
        """Return a record id that survives a refresh.

        Derived from the record rather than its position, so a reader keeps its
        selection no matter what has been appended since, and so a write and a
        later read agree on what to call the same line. The grain matches the
        uniqueness the database sink enforces.
        """
        return f"{stream}:{invocation_id}:{sequence}"

    @staticmethod
    def _to_model_context(
        event: _ModelContextLogEvent,
        event_id: str,
    ) -> ModelContextTrace:
        """Map one stored model-context line onto its application schema."""
        return ModelContextTrace(
            invocation_id=event.invocation_id,
            occurred_at=datetime.fromisoformat(event.timestamp),
            model=event.model,
            mode=event.mode,
            model_call=event.model_call,
            session_id=event.session_id,
            user_id=event.user_id,
            system_message=event.system_message,
            messages=tuple(event.messages),
            tools=tuple(event.tools),
            status=event.status,
            usage=event.usage,
            event_id=event_id,
        )

    @staticmethod
    def _to_response_gate(
        event: _ResponseGateLogEvent,
        event_id: str,
    ) -> ResponseGateTrace:
        """Map one stored response-gate line onto its application schema."""
        return ResponseGateTrace(
            invocation_id=event.invocation_id,
            occurred_at=datetime.fromisoformat(event.timestamp),
            model=event.model,
            mode=event.mode,
            evaluation_call=event.evaluation_call,
            repair_attempt=event.repair_attempt,
            decision=event.decision,
            candidate_characters=event.candidate_characters,
            passed=event.passed,
            session_id=event.session_id,
            user_id=event.user_id,
            violations=tuple(event.violations),
            feedback=event.feedback,
            candidate_message_id=event.candidate_message_id,
            candidate=event.candidate,
            available_tools=tuple(event.available_tools),
            tools_used=tuple(event.tools_used),
            gate_context=event.gate_context,
            usage=event.usage,
            error_type=event.error_type,
            error_message=event.error_message,
            event_id=event_id,
        )
