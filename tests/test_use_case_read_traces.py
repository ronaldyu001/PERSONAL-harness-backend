"""Tests for reading agent traces back."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from application.observability import (
    ModelContextTrace,
    TraceReadRequest,
    TraceReadResult,
)
from application.use_cases import ReadTracesUseCase


class RecordingTraces:
    """ObservabilityPort read double that records what it was asked."""

    def __init__(self, records: tuple[object, ...] = ()) -> None:
        self.records = records
        self.requests: list[TraceReadRequest] = []

    async def record_model_context(self, request):
        raise NotImplementedError

    async def record_response_gate(self, request):
        raise NotImplementedError

    async def read_traces(self, request: TraceReadRequest) -> TraceReadResult:
        self.requests.append(request)
        return TraceReadResult(stream=request.stream, records=self.records)


def _use_case(
    traces: RecordingTraces,
    *,
    default_page_size: int = 100,
    max_page_size: int = 500,
) -> ReadTracesUseCase:
    return ReadTracesUseCase(
        traces,
        default_page_size=default_page_size,
        max_page_size=max_page_size,
    )


def _trace() -> ModelContextTrace:
    return ModelContextTrace(
        invocation_id="invocation-1",
        occurred_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
        model="qwen",
        mode="full",
        model_call=1,
        session_id="session-1",
        user_id="user-1",
    )


class ReadTracesPagingTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_omitted_limit_uses_the_configured_default(self) -> None:
        traces = RecordingTraces()

        await _use_case(traces).read(
            TraceReadRequest(stream="model-context", user_id="user-1")
        )

        # Sinks always see a concrete count, never None.
        self.assertEqual(traces.requests[0].limit, 100)

    async def test_an_oversized_limit_is_clamped_to_the_ceiling(self) -> None:
        traces = RecordingTraces()

        await _use_case(traces).read(
            TraceReadRequest(
                stream="model-context", user_id="user-1", limit=5_000
            )
        )

        self.assertEqual(traces.requests[0].limit, 500)

    async def test_a_limit_under_the_ceiling_is_honored(self) -> None:
        traces = RecordingTraces()

        await _use_case(traces).read(
            TraceReadRequest(stream="model-context", user_id="user-1", limit=5)
        )

        self.assertEqual(traces.requests[0].limit, 5)

    async def test_the_filters_reach_the_sink_unchanged(self) -> None:
        traces = RecordingTraces((_trace(),))

        result = await _use_case(traces).read(
            TraceReadRequest(
                stream="response-gate",
                user_id="user-1",
                session_id="session-1",
                invocation_id="invocation-1",
            )
        )

        asked = traces.requests[0]
        self.assertEqual(asked.stream, "response-gate")
        self.assertEqual(asked.user_id, "user-1")
        self.assertEqual(asked.session_id, "session-1")
        self.assertEqual(asked.invocation_id, "invocation-1")
        self.assertEqual(len(result.records), 1)


class ReadTracesPolicyTests(unittest.TestCase):
    def test_a_default_above_the_ceiling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _use_case(RecordingTraces(), default_page_size=600, max_page_size=500)

    def test_a_non_positive_default_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _use_case(RecordingTraces(), default_page_size=0)

    def test_a_non_positive_ceiling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _use_case(RecordingTraces(), max_page_size=0)


class TraceReadRequestValidationTests(unittest.TestCase):
    def test_an_owner_is_required(self) -> None:
        with self.assertRaises(ValueError):
            TraceReadRequest(stream="model-context", user_id="   ")

    def test_a_blank_session_is_rejected(self) -> None:
        # Blank is not the same as omitted, and would silently match nothing.
        with self.assertRaises(ValueError):
            TraceReadRequest(
                stream="model-context", user_id="user-1", session_id="  "
            )

    def test_a_blank_invocation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TraceReadRequest(
                stream="model-context", user_id="user-1", invocation_id="  "
            )

    def test_a_non_positive_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TraceReadRequest(
                stream="model-context", user_id="user-1", limit=0
            )


if __name__ == "__main__":
    unittest.main()
