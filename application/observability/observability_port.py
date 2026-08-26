"""Application-facing protocol for recording and reading agent traces."""

from __future__ import annotations

from typing import Protocol

from application.observability.schemas import (
    ModelContextWriteRequest,
    ResponseGateWriteRequest,
    TraceReadRequest,
    TraceReadResult,
    TraceWriteResult,
)


class ObservabilityPort(Protocol):
    """Application boundary implemented by concrete trace sinks.

    Both halves belong to one protocol because a sink that records traces is
    the sink that can read them back; keeping them apart would let a store be
    wired for writing and silently be unreadable.
    """

    async def record_model_context(
        self,
        request: ModelContextWriteRequest,
    ) -> TraceWriteResult:
        """Record one effective model request and its completion metadata."""
        ...

    async def record_response_gate(
        self,
        request: ResponseGateWriteRequest,
    ) -> TraceWriteResult:
        """Record one gate evaluation and routing decision."""
        ...

    async def read_traces(self, request: TraceReadRequest) -> TraceReadResult:
        """Read one stream of traces back, most recent first."""
        ...
