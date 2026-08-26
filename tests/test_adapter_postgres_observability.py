"""Tests for the Postgres trace sink, without a Postgres.

The statements are compiled rather than executed: that is enough to catch a
misspelled column or a lost filter, which is how a mapping layer fails.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from application.observability import (
    ModelContextTrace,
    ModelContextWriteRequest,
    ResponseGateTrace,
    ResponseGateWriteRequest,
    TraceReadRequest,
)
from infrastructure.observability import PostgresObservabilityAdapter


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Result:
    def __init__(self, rows: tuple[object, ...]) -> None:
        self._rows = rows
        self.rowcount = len(rows)

    def scalars(self):
        return self

    def all(self):
        return self._rows


class RecordingSession:
    """AsyncSession double that keeps the statements it was given."""

    def __init__(self, rows: tuple[object, ...] = ()) -> None:
        self.rows = rows
        self.statements: list[object] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _Transaction()

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.rows)


def _adapter(session: RecordingSession) -> PostgresObservabilityAdapter:
    return PostgresObservabilityAdapter(lambda: session)


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def _context_row() -> SimpleNamespace:
    return SimpleNamespace(
        event_id="event-1",
        occurred_at=NOW,
        invocation_id="invocation-1",
        session_id="session-1",
        user_id="user-1",
        model="qwen",
        mode="full",
        model_call=2,
        system_message={"type": "system"},
        messages=[{"type": "human", "content_characters": 5}],
        tools=[{"name": "search_web"}],
        status="success",
        usage={"total_tokens": 92},
    )


def _gate_row() -> SimpleNamespace:
    return SimpleNamespace(
        event_id="event-2",
        occurred_at=NOW,
        invocation_id="invocation-1",
        session_id=None,
        user_id="user-1",
        model="qwen",
        mode="structure",
        evaluation_call=1,
        repair_attempt=0,
        decision="allow_on_error",
        passed=None,
        violations=["Offered an unavailable web search."],
        feedback=None,
        candidate_message_id="candidate-1",
        candidate_characters=13,
        candidate=None,
        available_tools=["search_web"],
        tools_used=[],
        usage=None,
        error_type="RuntimeError",
        error_message="the evaluator is unavailable",
    )


def _context_trace() -> ModelContextTrace:
    return ModelContextTrace(
        invocation_id="invocation-1",
        occurred_at=NOW,
        model="qwen",
        mode="full",
        model_call=1,
        session_id="session-1",
        user_id="user-1",
        messages=({"type": "human"},),
    )


def _gate_trace() -> ResponseGateTrace:
    return ResponseGateTrace(
        invocation_id="invocation-1",
        occurred_at=NOW,
        model="qwen",
        mode="full",
        evaluation_call=1,
        repair_attempt=0,
        decision="allow",
        candidate_characters=13,
        passed=True,
        session_id="session-1",
        user_id="user-1",
    )


class PostgresWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_repeated_model_call_is_ignored(self) -> None:
        # The grain of the table is the invocation and the call within it.
        session = RecordingSession()

        result = await _adapter(session).record_model_context(
            ModelContextWriteRequest(trace=_context_trace())
        )

        sql = _sql(session.statements[0])
        self.assertIn("INSERT INTO model_context_events", sql)
        self.assertIn("ON CONFLICT (invocation_id, model_call) DO NOTHING", sql)
        self.assertEqual(result.stream, "model-context")
        self.assertTrue(result.event_id)

    async def test_a_repeated_gate_evaluation_is_ignored(self) -> None:
        session = RecordingSession()

        result = await _adapter(session).record_response_gate(
            ResponseGateWriteRequest(trace=_gate_trace())
        )

        sql = _sql(session.statements[0])
        self.assertIn("INSERT INTO response_gate_events", sql)
        self.assertIn(
            "ON CONFLICT (invocation_id, evaluation_call) DO NOTHING", sql
        )
        self.assertEqual(result.stream, "response-gate")

    async def test_a_supplied_event_id_is_kept(self) -> None:
        session = RecordingSession()
        trace = ModelContextTrace(
            invocation_id="invocation-1",
            occurred_at=NOW,
            model="qwen",
            mode="full",
            model_call=1,
            event_id="chosen-id",
        )

        result = await _adapter(session).record_model_context(
            ModelContextWriteRequest(trace=trace)
        )

        self.assertEqual(result.event_id, "chosen-id")


class PostgresReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_owner_scopes_every_read(self) -> None:
        session = RecordingSession()

        await _adapter(session).read_traces(
            TraceReadRequest(
                stream="model-context", user_id="user-1", limit=10
            )
        )

        sql = _sql(session.statements[0])
        self.assertIn("FROM model_context_events", sql)
        self.assertIn("model_context_events.user_id = ", sql)
        # Newest first, with the call number breaking a shared timestamp.
        self.assertIn(
            "ORDER BY model_context_events.occurred_at DESC, "
            "model_context_events.model_call DESC",
            sql,
        )
        self.assertIn("LIMIT", sql)

    async def test_the_focus_filters_reach_the_query(self) -> None:
        session = RecordingSession()

        await _adapter(session).read_traces(
            TraceReadRequest(
                stream="response-gate",
                user_id="user-1",
                session_id="session-1",
                invocation_id="invocation-1",
            )
        )

        sql = _sql(session.statements[0])
        self.assertIn("FROM response_gate_events", sql)
        self.assertIn("response_gate_events.session_id = ", sql)
        self.assertIn("response_gate_events.invocation_id = ", sql)
        self.assertIn(
            "ORDER BY response_gate_events.occurred_at DESC, "
            "response_gate_events.evaluation_call DESC",
            sql,
        )

    async def test_an_omitted_filter_is_not_a_null_comparison(self) -> None:
        # A missing session means every session, not the ones with no session.
        session = RecordingSession()

        await _adapter(session).read_traces(
            TraceReadRequest(stream="model-context", user_id="user-1")
        )

        self.assertNotIn("session_id IS NULL", _sql(session.statements[0]))

    async def test_a_stored_model_context_row_maps_onto_its_schema(self) -> None:
        session = RecordingSession(rows=(_context_row(),))

        result = await _adapter(session).read_traces(
            TraceReadRequest(stream="model-context", user_id="user-1")
        )

        trace = result.records[0]
        self.assertEqual(result.stream, "model-context")
        self.assertEqual(trace.event_id, "event-1")
        self.assertEqual(trace.invocation_id, "invocation-1")
        self.assertEqual(trace.session_id, "session-1")
        self.assertEqual(trace.model_call, 2)
        self.assertEqual(trace.messages, ({"type": "human", "content_characters": 5},))
        self.assertEqual(trace.tools, ({"name": "search_web"},))
        self.assertEqual(trace.status, "success")
        self.assertEqual(trace.usage, {"total_tokens": 92})

    async def test_a_stored_gate_row_maps_onto_its_schema(self) -> None:
        session = RecordingSession(rows=(_gate_row(),))

        result = await _adapter(session).read_traces(
            TraceReadRequest(stream="response-gate", user_id="user-1")
        )

        trace = result.records[0]
        self.assertEqual(trace.event_id, "event-2")
        self.assertEqual(trace.decision, "allow_on_error")
        # Tri-state: None is the gate erroring, not a missing value.
        self.assertIsNone(trace.passed)
        # A temporary turn references no conversation.
        self.assertIsNone(trace.session_id)
        self.assertEqual(trace.violations, ("Offered an unavailable web search.",))
        self.assertEqual(trace.available_tools, ("search_web",))
        self.assertEqual(trace.tools_used, ())
        self.assertEqual(trace.error_type, "RuntimeError")


class PostgresRetentionTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_streams_are_pruned_together(self) -> None:
        session = RecordingSession()

        await _adapter(session).purge_before(NOW - timedelta(days=30))

        statements = [_sql(statement) for statement in session.statements]
        self.assertEqual(len(statements), 2)
        self.assertIn("DELETE FROM model_context_events", statements[0])
        self.assertIn("occurred_at < ", statements[0])
        self.assertIn("DELETE FROM response_gate_events", statements[1])


if __name__ == "__main__":
    unittest.main()
