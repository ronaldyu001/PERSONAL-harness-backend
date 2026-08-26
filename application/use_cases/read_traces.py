"""Agent trace read use case."""

from __future__ import annotations

from dataclasses import replace

from application.observability import (
    ObservabilityPort,
    TraceReadRequest,
    TraceReadResult,
)


class ReadTracesUseCase:
    """Reads one of Maia's trace streams back for inspection.

    Only the read half of ``ObservabilityPort`` is exercised here; the write
    half belongs to the agent's logging middleware.
    """

    def __init__(
        self,
        traces: ObservabilityPort,
        *,
        default_page_size: int,
        max_page_size: int,
    ) -> None:
        """Create the use case with a sink and its configured paging policy."""
        if default_page_size <= 0:
            raise ValueError("default_page_size must be positive")
        if max_page_size <= 0:
            raise ValueError("max_page_size must be positive")
        if default_page_size > max_page_size:
            raise ValueError("default_page_size must not exceed max_page_size")

        self._traces = traces
        self._default_page_size = default_page_size
        self._max_page_size = max_page_size

    async def read(self, request: TraceReadRequest) -> TraceReadResult:
        """Read one stream, most recent first."""
        # Paging policy is resolved here, so sinks always see a concrete count.
        requested = request.limit or self._default_page_size
        return await self._traces.read_traces(
            replace(request, limit=min(requested, self._max_page_size))
        )
