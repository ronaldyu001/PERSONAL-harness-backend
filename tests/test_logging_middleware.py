"""Tests for local model-context logging middleware."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.runtime import Runtime

from infrastructure.agent.logging import (
    ModelContextLogEvent,
    ResponseGateLogEvent,
    ResponseGateLogWriter,
)
from infrastructure.agent.middleware.logging_middleware import ContextLoggingMiddleware
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.settings import load_infrastructure_settings


class ContextLoggingMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def test_loggers_follow_the_resolved_logging_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = load_infrastructure_settings(environ={
                "LITELLM_BASE_URL": "http://litellm.test",
                "AGENT_CONTEXT_LOGGING": "structure",
                "AGENT_CONTEXT_LOG_DIR": temp_dir,
            })

            context_logger = ContextLoggingMiddleware.from_config(
                settings.logging
            )
            gate_logger = ResponseGateLogWriter.from_config(
                settings.logging
            )

            self.assertTrue(context_logger.enabled)
            self.assertTrue(gate_logger.enabled)
            self.assertEqual(context_logger.log_path.parent, Path(temp_dir))
            self.assertEqual(gate_logger.log_path.parent, Path(temp_dir))

    async def test_logs_effective_context_without_tool_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            middleware = ContextLoggingMiddleware(
                mode="full",
                log_dir=temp_dir,
            )
            model = FakeMessagesListChatModel(
                responses=[AIMessage(content="unused")]
            )
            request = ModelRequest(
                model=model,
                system_message=SystemMessage(content="You are Maia."),
                messages=[
                    HumanMessage(content="What happened today?"),
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "web_search",
                            "args": {"query": "important events today"},
                            "id": "call-123",
                            "type": "tool_call",
                        }],
                    ),
                    ToolMessage(
                        content="[1] Bounded evidence",
                        tool_call_id="call-123",
                        name="web_search",
                        artifact={"raw_response": "must not be logged"},
                    ),
                ],
                tools=[],
                runtime=Runtime(
                    context=AgentRuntimeContext(
                        user_id="user-1",
                        session_id="session-1",
                    )
                ),
            )

            async def handler(received_request):
                self.assertIs(received_request, request)
                return AIMessage(
                    content="Final response",
                    usage_metadata={
                        "input_tokens": 137,
                        "output_tokens": 12,
                        "total_tokens": 149,
                    },
                )

            response = await middleware.awrap_model_call(request, handler)

            self.assertEqual(response.text, "Final response")
            events = Path(middleware.log_path).read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(events), 1)

            event = json.loads(events[0])
            ModelContextLogEvent.model_validate(event)
            self.assertEqual(event["session_id"], "session-1")
            self.assertEqual(event["model_call"], 1)
            self.assertEqual(event["status"], "success")
            self.assertEqual(
                event["usage"],
                {
                    "input_tokens": 137,
                    "output_tokens": 12,
                    "total_tokens": 149,
                },
            )
            self.assertEqual(
                event["system_message"]["content"],
                "You are Maia.",
            )
            self.assertEqual(
                event["messages"][1]["tool_calls"][0],
                {
                    "name": "web_search",
                    "id": "call-123",
                    "args": {"query": "important events today"},
                },
            )
            self.assertEqual(
                event["messages"][2]["content"],
                "[1] Bounded evidence",
            )
            self.assertTrue(event["messages"][2]["artifact_excluded"])
            self.assertNotIn("raw_response", events[0])

    async def test_logs_failed_model_call_without_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            middleware = ContextLoggingMiddleware(
                mode="structure",
                log_dir=temp_dir,
            )
            request = ModelRequest(
                model=FakeMessagesListChatModel(
                    responses=[AIMessage(content="unused")]
                ),
                messages=[HumanMessage(content="Hello")],
            )

            async def handler(_request):
                raise RuntimeError("provider failed")

            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await middleware.awrap_model_call(request, handler)

            events = Path(middleware.log_path).read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(events), 1)
            event = json.loads(events[0])
            self.assertEqual(event["status"], "error")
            self.assertIsNone(event["usage"])

    async def test_off_mode_does_not_create_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "disabled"
            middleware = ContextLoggingMiddleware(
                mode="off",
                log_dir=log_dir,
            )
            model = FakeMessagesListChatModel(
                responses=[AIMessage(content="unused")]
            )
            request = ModelRequest(
                model=model,
                messages=[HumanMessage(content="Hello")],
            )

            async def handler(_request):
                return AIMessage(content="Hello")

            await middleware.awrap_model_call(request, handler)

            self.assertFalse(log_dir.exists())

    async def test_gate_structure_log_uses_a_separate_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gate_writer = ResponseGateLogWriter(
                mode="structure",
                log_dir=temp_dir,
            )

            await gate_writer.log_evaluation(
                session_id="session-1",
                model="qwen",
                evaluation_call=1,
                repair_attempt=1,
                decision="retry",
                passed=False,
                violations=["Offered an unavailable web search."],
                feedback="Do not offer to browse.",
                candidate_message_id="candidate-1",
                candidate="I can check that online.",
                available_tools=[],
                tools_used=[],
                usage={"total_tokens": 92},
            )

            self.assertEqual(gate_writer.log_path.name, "response-gate.jsonl")
            self.assertFalse((Path(temp_dir) / "agent-context.jsonl").exists())
            event = ResponseGateLogEvent.model_validate_json(
                gate_writer.log_path.read_text(encoding="utf-8")
            )
            self.assertEqual(event.decision, "retry")
            self.assertEqual(event.usage, {"total_tokens": 92})
            self.assertIsNone(event.candidate)
            self.assertIsNone(event.feedback)

if __name__ == "__main__":
    unittest.main()
