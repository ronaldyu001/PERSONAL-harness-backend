"""Tests for the JSON Lines trace sink."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from application.observability import (
    ModelContextTrace,
    ModelContextWriteRequest,
    ResponseGateTrace,
    ResponseGateWriteRequest,
    TraceReadRequest,
)
from infrastructure.observability import AdapterJsonlObservability
from infrastructure.settings import load_infrastructure_settings


NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


def _context_trace(
    *,
    model_call: int = 1,
    invocation_id: str = "invocation-1",
    session_id: str | None = "session-1",
    user_id: str | None = "user-1",
    occurred_at: datetime | None = None,
    mode: str = "full",
) -> ModelContextTrace:
    return ModelContextTrace(
        invocation_id=invocation_id,
        occurred_at=occurred_at or NOW,
        model="qwen",
        mode=mode,
        model_call=model_call,
        session_id=session_id,
        user_id=user_id,
        system_message={"type": "system", "content_characters": 12},
        messages=({"type": "human", "content": "Hello", "content_characters": 5},),
        tools=({"name": "search_web"},),
        status="success",
        usage={"total_tokens": 92},
    )


def _gate_trace(
    *,
    evaluation_call: int = 1,
    decision: str = "allow",
    passed: bool | None = True,
    user_id: str | None = "user-1",
) -> ResponseGateTrace:
    return ResponseGateTrace(
        invocation_id="invocation-1",
        occurred_at=NOW,
        model="qwen",
        mode="full",
        evaluation_call=evaluation_call,
        repair_attempt=0,
        decision=decision,
        candidate_characters=13,
        passed=passed,
        session_id="session-1",
        user_id=user_id,
        violations=("Offered an unavailable web search.",),
        feedback="Do not offer to browse.",
        candidate_message_id="candidate-1",
        candidate="Play Pokemon.",
        available_tools=("search_web",),
        tools_used=(),
        usage={"total_tokens": 92},
    )


class JsonlRoundTripTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_model_context_trace_survives_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AdapterJsonlObservability(
                context_dir=temp_dir,
                response_gate_dir=temp_dir,
            )
            written = _context_trace()

            await adapter.record_model_context(
                ModelContextWriteRequest(trace=written)
            )
            result = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="user-1")
            )

        read = result.records[0]
        self.assertEqual(result.stream, "model-context")
        self.assertEqual(read.invocation_id, written.invocation_id)
        self.assertEqual(read.occurred_at, written.occurred_at)
        self.assertEqual(read.model_call, written.model_call)
        self.assertEqual(read.session_id, written.session_id)
        self.assertEqual(read.messages, written.messages)
        self.assertEqual(read.tools, written.tools)
        self.assertEqual(read.status, "success")
        self.assertEqual(read.usage, {"total_tokens": 92})

    async def test_a_response_gate_trace_survives_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AdapterJsonlObservability(
                context_dir=temp_dir,
                response_gate_dir=temp_dir,
            )
            written = _gate_trace(decision="allow_on_error", passed=None)

            await adapter.record_response_gate(
                ResponseGateWriteRequest(trace=written)
            )
            result = await adapter.read_traces(
                TraceReadRequest(stream="response-gate", user_id="user-1")
            )

        read = result.records[0]
        self.assertEqual(read.decision, "allow_on_error")
        # Tri-state: None is the gate erroring, not a missing value.
        self.assertIsNone(read.passed)
        self.assertEqual(read.violations, written.violations)
        self.assertEqual(read.candidate, "Play Pokemon.")
        self.assertEqual(read.available_tools, ("search_web",))
        self.assertEqual(read.tools_used, ())

    async def test_the_write_and_the_read_agree_on_the_record_id(self) -> None:
        # The reader keys its list on this, so the two have to match.
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AdapterJsonlObservability(
                context_dir=temp_dir,
                response_gate_dir=temp_dir,
            )

            written = await adapter.record_model_context(
                ModelContextWriteRequest(trace=_context_trace())
            )
            first = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="user-1")
            )
            # Appending more must not renumber what is already there.
            await adapter.record_model_context(
                ModelContextWriteRequest(trace=_context_trace(model_call=2))
            )
            second = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="user-1")
            )

        self.assertEqual(first.records[0].event_id, written.event_id)
        self.assertEqual(second.records[-1].event_id, written.event_id)


class JsonlStreamSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_stream_has_its_own_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AdapterJsonlObservability(
                context_dir=temp_dir,
                response_gate_dir=temp_dir,
            )

            await adapter.record_response_gate(
                ResponseGateWriteRequest(trace=_gate_trace())
            )

            self.assertEqual(
                adapter.log_path("response-gate").name,
                "response-gate.jsonl",
            )
            self.assertFalse(
                (Path(temp_dir) / "agent-context.jsonl").exists()
            )
            # A stream nothing has written yet reads empty, not broken.
            context = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="user-1")
            )

        self.assertEqual(context.records, ())

    def test_the_sink_follows_the_resolved_logging_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = load_infrastructure_settings(environ={
                "LITELLM_BASE_URL": "http://litellm.test",
                "AGENT_CONTEXT_LOGGING": "structure",
                "AGENT_CONTEXT_LOG_DIR": temp_dir,
            })

            adapter = AdapterJsonlObservability.from_config(settings.logging)

            self.assertEqual(
                adapter.log_path("model-context").parent, Path(temp_dir)
            )
            self.assertEqual(
                adapter.log_path("response-gate").parent, Path(temp_dir)
            )


class JsonlFilterTests(unittest.IsolatedAsyncioTestCase):
    async def _adapter(self, temp_dir: str) -> AdapterJsonlObservability:
        adapter = AdapterJsonlObservability(
            context_dir=temp_dir,
            response_gate_dir=temp_dir,
        )
        # Three turns, oldest first, so the newest is last in the file.
        for index, (invocation, session, user) in enumerate(
            (
                ("invocation-1", "session-1", "user-1"),
                ("invocation-2", "session-2", "user-1"),
                ("invocation-3", "session-3", "user-2"),
            )
        ):
            await adapter.record_model_context(
                ModelContextWriteRequest(
                    trace=_context_trace(
                        invocation_id=invocation,
                        session_id=session,
                        user_id=user,
                        occurred_at=NOW + timedelta(minutes=index),
                    )
                )
            )
        return adapter

    async def test_records_come_back_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = await self._adapter(temp_dir)
            result = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="user-1")
            )

        self.assertEqual(
            [record.invocation_id for record in result.records],
            ["invocation-2", "invocation-1"],
        )

    async def test_another_user_sees_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = await self._adapter(temp_dir)
            result = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="nobody")
            )

        self.assertEqual(result.records, ())

    async def test_a_session_narrows_to_one_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = await self._adapter(temp_dir)
            result = await adapter.read_traces(
                TraceReadRequest(
                    stream="model-context",
                    user_id="user-1",
                    session_id="session-1",
                )
            )

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].session_id, "session-1")

    async def test_an_invocation_narrows_to_one_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = await self._adapter(temp_dir)
            result = await adapter.read_traces(
                TraceReadRequest(
                    stream="model-context",
                    user_id="user-1",
                    invocation_id="invocation-1",
                )
            )

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].invocation_id, "invocation-1")

    async def test_filters_apply_before_the_limit(self) -> None:
        # Taking the window first would hide the older session entirely, which
        # is what a client-side filter over a paged read gets wrong.
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = await self._adapter(temp_dir)
            result = await adapter.read_traces(
                TraceReadRequest(
                    stream="model-context",
                    user_id="user-1",
                    session_id="session-1",
                    limit=1,
                )
            )

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].session_id, "session-1")

    async def test_a_limit_takes_the_newest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = await self._adapter(temp_dir)
            result = await adapter.read_traces(
                TraceReadRequest(
                    stream="model-context", user_id="user-1", limit=1
                )
            )

        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].invocation_id, "invocation-2")


class JsonlDamagedFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_an_unreadable_line_is_skipped_rather_than_fatal(self) -> None:
        # A half-written final line is normal for a file being appended to.
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AdapterJsonlObservability(
                context_dir=temp_dir,
                response_gate_dir=temp_dir,
            )
            await adapter.record_model_context(
                ModelContextWriteRequest(trace=_context_trace())
            )
            with adapter.log_path("model-context").open(
                "a", encoding="utf-8"
            ) as log_file:
                log_file.write('{"event": "model_context", "timesta')

            with self.assertLogs(
                "infrastructure.observability.adapter_jsonl",
                level="WARNING",
            ):
                result = await adapter.read_traces(
                    TraceReadRequest(stream="model-context", user_id="user-1")
                )

        self.assertEqual(len(result.records), 1)

    async def test_a_line_written_before_owners_is_not_served(self) -> None:
        # Older lines carry no user_id, so a scoped read cannot claim them.
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = AdapterJsonlObservability(
                context_dir=temp_dir,
                response_gate_dir=temp_dir,
            )
            path = adapter.log_path("model-context")
            path.parent.mkdir(parents=True, exist_ok=True)
            legacy = {
                "event": "model_context",
                "timestamp": NOW.isoformat(),
                "invocation_id": "invocation-old",
                "session_id": "session-1",
                "model": "qwen",
                "mode": "full",
                "model_call": 1,
            }
            path.write_text(f"{json.dumps(legacy)}\n", encoding="utf-8")

            result = await adapter.read_traces(
                TraceReadRequest(stream="model-context", user_id="user-1")
            )

        self.assertEqual(result.records, ())


if __name__ == "__main__":
    unittest.main()
