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

from infrastructure.agent.middleware.logging_middleware import (
    ContextLoggingMiddleware,
)
from infrastructure.agent.runtime_context import AgentRuntimeContext


class ContextLoggingMiddlewareTests(unittest.IsolatedAsyncioTestCase):
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
                return AIMessage(content="Final response")

            response = await middleware.awrap_model_call(request, handler)

            self.assertEqual(response.text, "Final response")
            events = Path(middleware.log_path).read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(events), 1)

            event = json.loads(events[0])
            self.assertEqual(event["session_id"], "session-1")
            self.assertEqual(event["model_call"], 1)
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


if __name__ == "__main__":
    unittest.main()
