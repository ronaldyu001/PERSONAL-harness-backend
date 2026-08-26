"""Observability port and schemas for Maia's agent traces.

The agent's logging middleware drives the write side of this port and
``ReadTracesUseCase`` drives the read side.
"""

from application.observability.port_observability import ObservabilityPort
from application.observability.schemas import (
    GateDecision,
    ModelContextTrace,
    ModelContextWriteRequest,
    ResponseGateTrace,
    ResponseGateWriteRequest,
    Trace,
    TraceMode,
    TraceReadRequest,
    TraceReadResult,
    TraceStream,
    TraceWriteResult,
)


__all__ = (
    "GateDecision",
    "ModelContextTrace",
    "ModelContextWriteRequest",
    "ObservabilityPort",
    "ResponseGateTrace",
    "ResponseGateWriteRequest",
    "Trace",
    "TraceMode",
    "TraceReadRequest",
    "TraceReadResult",
    "TraceStream",
    "TraceWriteResult",
)
