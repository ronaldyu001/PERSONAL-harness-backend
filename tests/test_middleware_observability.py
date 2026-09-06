"""Tests for what Maia's agent records about a turn."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.runtime import Runtime

from application.observability import (
    ModelContextTrace,
    ResponseGateTrace,
    TraceWriteResult,
)
from infrastructure.agent.context import ContextRuntime
from infrastructure.agent.middleware import (
    MiddlewareModelContext,
    MiddlewareModelResponseGate,
    ResponseEvaluation,
)
from infrastructure.settings import load_infrastructure_settings


class RecordingObservability:
    """PortObservability double that keeps what it was handed."""

    def __init__(self) -> None:
        self.model_context: list[ModelContextTrace] = []
        self.response_gate: list[ResponseGateTrace] = []

    async def record_model_context(self, request) -> TraceWriteResult:
        self.model_context.append(request.trace)
        return TraceWriteResult(stream="model-context", event_id="recorded")

    async def record_response_gate(self, request) -> TraceWriteResult:
        self.response_gate.append(request.trace)
        return TraceWriteResult(stream="response-gate", event_id="recorded")

    async def read_traces(self, request):
        raise NotImplementedError


class FailingObservability:
    """PortObservability double standing in for a sink outage."""

    async def record_model_context(self, request) -> TraceWriteResult:
        raise RuntimeError("the trace sink is unreachable")

    async def record_response_gate(self, request) -> TraceWriteResult:
        raise RuntimeError("the trace sink is unreachable")

    async def read_traces(self, request):
        raise NotImplementedError


class QueueEvaluator:
    """Return predefined structured gate decisions."""

    def __init__(self, *results: object) -> None:
        self._results = list(results)

    async def ainvoke(self, request):
        return self._results.pop(0)


def _gate_config():
    return load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm:4000",
    }).agent.response_gate


def _model() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="unused")])


def _gate(observability, *, mode: str = "full"):
    return MiddlewareModelResponseGate.from_config(
        _gate_config(),
        model=_model(),
        system_prompt="Answer directly.",
        observability=observability,
        mode=mode,
        evaluator=QueueEvaluator(
            ResponseEvaluation(
                passed=False,
                violations=["Offered an unavailable web search."],
                feedback="Do not offer to browse.",
            )
        ),
    )


def _gate_state() -> dict[str, list[object]]:
    return {
        "messages": [
            HumanMessage(content="Which game should I play?"),
            AIMessage(content="I can check that online.", id="candidate-1"),
        ]
    }


def _request(runtime, *, messages=None, tools=()) -> ModelRequest:
    return ModelRequest(
        model=_model(),
        system_message=SystemMessage(content="You are Maia."),
        messages=(
            messages
            if messages is not None
            else [HumanMessage(content="Which game should I play?")]
        ),
        tools=list(tools),
        runtime=runtime,
    )


async def _handler(request):
    return AIMessage(content="Play Pokemon.", id="candidate-1")


class InvocationIdTests(unittest.IsolatedAsyncioTestCase):
    """The two streams have to be joinable for one turn."""

    async def test_both_streams_share_the_runtime_invocation_id(self) -> None:
        # One turn means one runtime context, and both writers read the id off
        # it. Before this, each writer minted its own uuid and the ledger could
        # not stack a turn's context and gate records together.
        context = ContextRuntime(user_id="user-1", session_id="session-1")
        runtime = Runtime(context=context)
        observability = RecordingObservability()

        await MiddlewareModelContext(
            mode="structure", observability=observability
        ).awrap_model_call(_request(runtime), _handler)
        await _gate(observability).aafter_model(_gate_state(), runtime)

        self.assertEqual(
            observability.model_context[0].invocation_id,
            context.invocation_id,
        )
        self.assertEqual(
            observability.response_gate[0].invocation_id,
            context.invocation_id,
        )

    async def test_a_missing_runtime_context_still_records_an_id(self) -> None:
        # Readers group by this id, so a null would collapse unrelated turns.
        observability = RecordingObservability()

        await MiddlewareModelContext(
            mode="structure", observability=observability
        ).awrap_model_call(_request(None), _handler)

        trace = observability.model_context[0]
        self.assertTrue(trace.invocation_id)
        self.assertIsNone(trace.session_id)


class TemporaryTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_neither_stream_references_a_conversation(self) -> None:
        # A temporary turn writes no conversation row, so a trace that named
        # one would point at nothing.
        runtime = Runtime(
            context=ContextRuntime(
                user_id="user-1",
                session_id="session-1",
                temporary=True,
            )
        )
        observability = RecordingObservability()

        await MiddlewareModelContext(
            mode="structure", observability=observability
        ).awrap_model_call(_request(runtime), _handler)
        await _gate(observability).aafter_model(_gate_state(), runtime)

        self.assertIsNone(observability.model_context[0].session_id)
        self.assertIsNone(observability.response_gate[0].session_id)
        # The owner still scopes the read, temporary or not.
        self.assertEqual(observability.model_context[0].user_id, "user-1")


class RecordingModeTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self) -> Runtime:
        return Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )

    async def test_off_records_nothing(self) -> None:
        observability = RecordingObservability()
        middleware = MiddlewareModelContext(
            mode="off", observability=observability
        )

        self.assertFalse(middleware.enabled)
        await middleware.awrap_model_call(_request(self._runtime()), _handler)

        self.assertEqual(observability.model_context, [])

    async def test_without_a_sink_nothing_is_recorded(self) -> None:
        middleware = MiddlewareModelContext(mode="full", observability=None)

        self.assertFalse(middleware.enabled)
        response = await middleware.awrap_model_call(
            _request(self._runtime()), _handler
        )

        self.assertEqual(response.text, "Play Pokemon.")

    async def test_structure_keeps_the_shape_and_drops_the_text(self) -> None:
        observability = RecordingObservability()

        await MiddlewareModelContext(
            mode="structure", observability=observability
        ).awrap_model_call(_request(self._runtime()), _handler)

        message = observability.model_context[0].messages[0]
        self.assertNotIn("content", message)
        self.assertEqual(message["content_characters"], 25)

    async def test_full_keeps_the_text(self) -> None:
        observability = RecordingObservability()

        await MiddlewareModelContext(
            mode="full", observability=observability
        ).awrap_model_call(_request(self._runtime()), _handler)

        message = observability.model_context[0].messages[0]
        self.assertEqual(message["content"], "Which game should I play?")

    async def test_structure_drops_gate_feedback_and_candidate(self) -> None:
        observability = RecordingObservability()

        await _gate(observability, mode="structure").aafter_model(
            _gate_state(), self._runtime()
        )

        trace = observability.response_gate[0]
        self.assertIsNone(trace.feedback)
        self.assertIsNone(trace.candidate)
        # The decision and its size survive; only the words go.
        self.assertEqual(trace.decision, "retry")
        self.assertEqual(trace.candidate_characters, 24)
        self.assertEqual(trace.violations, ("Offered an unavailable web search.",))

    async def test_full_keeps_gate_feedback_and_candidate(self) -> None:
        observability = RecordingObservability()

        await _gate(observability, mode="full").aafter_model(
            _gate_state(), self._runtime()
        )

        trace = observability.response_gate[0]
        self.assertEqual(trace.feedback, "Do not offer to browse.")
        self.assertEqual(trace.candidate, "I can check that online.")


class RecordingFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_failing_sink_does_not_break_the_turn(self) -> None:
        runtime = Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )
        middleware = MiddlewareModelContext(
            mode="full", observability=FailingObservability()
        )

        with self.assertLogs(
            "infrastructure.agent.middleware.middleware_model_context",
            level="ERROR",
        ):
            response = await middleware.awrap_model_call(
                _request(runtime), _handler
            )

        self.assertEqual(response.text, "Play Pokemon.")

    async def test_a_failing_sink_does_not_break_the_gate(self) -> None:
        runtime = Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )

        with self.assertLogs(
            "infrastructure.agent.middleware.middleware_model_response_gate",
            level="ERROR",
        ):
            await _gate(FailingObservability()).aafter_model(
                _gate_state(), runtime
            )


class ModelCallNumberingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_call_number_climbs_across_one_turn(self) -> None:
        # The number is half of what makes a record unique within a turn.
        runtime = Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )
        observability = RecordingObservability()
        middleware = MiddlewareModelContext(
            mode="structure", observability=observability
        )

        for _ in range(3):
            await middleware.awrap_model_call(_request(runtime), _handler)

        self.assertEqual(
            [trace.model_call for trace in observability.model_context],
            [1, 2, 3],
        )
        self.assertEqual(
            {trace.invocation_id for trace in observability.model_context},
            {runtime.context.invocation_id},
        )

    async def test_a_failed_call_is_recorded_as_an_error(self) -> None:
        runtime = Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )
        observability = RecordingObservability()

        async def failing_handler(request):
            raise RuntimeError("the provider is down")

        with self.assertRaises(RuntimeError):
            await MiddlewareModelContext(
                mode="structure", observability=observability
            ).awrap_model_call(_request(runtime), failing_handler)

        trace = observability.model_context[0]
        self.assertEqual(trace.status, "error")
        self.assertIsNone(trace.usage)


class JsonSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_content_json_cannot_hold_is_flattened(self) -> None:
        # A JSONB column raises on these where a file sink used to coerce them,
        # so the middleware flattens before either sink sees the record.
        runtime = Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )
        observability = RecordingObservability()
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "recorded_at": datetime(2026, 8, 24, tzinfo=timezone.utc),
                }
            ]
        )

        await MiddlewareModelContext(
            mode="full", observability=observability
        ).awrap_model_call(
            _request(runtime, messages=[message]), _handler
        )

        content = observability.model_context[0].messages[0]["content"]
        self.assertEqual(content[0]["recorded_at"], "2026-08-24 00:00:00+00:00")

    async def test_a_tool_artifact_never_reaches_the_sink(self) -> None:
        runtime = Runtime(
            context=ContextRuntime(user_id="user-1", session_id="session-1")
        )
        observability = RecordingObservability()
        messages = [
            HumanMessage(content="What happened today?"),
            ToolMessage(
                content="[1] Bounded evidence",
                tool_call_id="call-123",
                name="web_search",
                artifact={"raw_response": "must not be recorded"},
            ),
        ]

        await MiddlewareModelContext(
            mode="full", observability=observability
        ).awrap_model_call(_request(runtime, messages=messages), _handler)

        tool_message = observability.model_context[0].messages[1]
        self.assertTrue(tool_message["artifact_excluded"])
        self.assertNotIn("artifact", tool_message)


if __name__ == "__main__":
    unittest.main()
