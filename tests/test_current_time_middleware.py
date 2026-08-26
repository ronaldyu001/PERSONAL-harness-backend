"""Tests for transient current-time model context."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.runtime import Runtime

from infrastructure.agent.middleware.middleware_current_time import (
    CurrentTimeMiddleware,
)
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.settings import load_infrastructure_settings


class CurrentTimeMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_injects_local_time_only_into_effective_model_request(self) -> None:
        config = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://litellm:4000",
        }).agent.time_context
        middleware = CurrentTimeMiddleware.from_config(config)
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="unused")]
        )
        messages = [HumanMessage(content="What time is it?")]
        request = ModelRequest(
            model=model,
            system_message=SystemMessage(content="You are Maia."),
            messages=messages,
            tools=[],
            runtime=Runtime(
                context=AgentRuntimeContext(
                    user_id="user-1",
                    session_id="session-1",
                    invocation_time_utc=datetime(
                        2026, 1, 15, 18, 30, tzinfo=timezone.utc
                    ),
                    timezone="America/Denver",
                )
            ),
        )
        received: list[ModelRequest] = []

        async def handler(effective_request: ModelRequest) -> AIMessage:
            received.append(effective_request)
            return AIMessage(content="It is 11:30 AM.")

        result = await middleware.awrap_model_call(request, handler)

        self.assertEqual(result.text, "It is 11:30 AM.")
        self.assertEqual(len(received), 1)
        effective_system = received[0].system_message
        assert effective_system is not None
        self.assertIn(
            "Current time: 2026-01-15T11:30:00-07:00 (America/Denver).",
            effective_system.text,
        )
        self.assertEqual(request.system_message.text, "You are Maia.")
        self.assertIs(request.messages, messages)
        self.assertEqual([message.text for message in messages], [
            "What time is it?"
        ])

    def test_runtime_context_uses_daylight_saving_offset(self) -> None:
        context = AgentRuntimeContext(
            user_id="user-1",
            session_id="session-1",
            invocation_time_utc=datetime(
                2026, 8, 18, 1, 30, tzinfo=timezone.utc
            ),
            timezone="America/Denver",
        )

        self.assertEqual(context.current_time_iso, "2026-08-17T19:30:00-06:00")

    def test_runtime_context_rejects_naive_datetime(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            AgentRuntimeContext(
                user_id="user-1",
                session_id="session-1",
                invocation_time_utc=datetime(2026, 8, 17, 12, 0),
                timezone="America/Denver",
            )


if __name__ == "__main__":
    unittest.main()
