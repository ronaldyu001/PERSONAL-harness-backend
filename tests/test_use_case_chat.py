"""Tests for the application chat use case."""

from __future__ import annotations

import unittest
from uuid import UUID

from application.agent import (
    AgentRequest,
    AgentResponse,
    EmptyAgentResponseError,
)
from application.use_cases import ChatCommand, UseCaseChat


class RecordingAgent:
    """Small PortAgent test double."""

    def __init__(self, content: str = "agent reply") -> None:
        self.content = content
        self.request: AgentRequest | None = None
        self.session_id: str | None = None
        self.user_id: str | None = None
        self.temporary: bool | None = None

    async def chat(
        self,
        request: AgentRequest,
        *,
        session_id: str,
        user_id: str,
        temporary: bool = False,
    ) -> AgentResponse:
        self.request = request
        self.session_id = session_id
        self.user_id = user_id
        self.temporary = temporary
        return AgentResponse(
            content=self.content,
            usage={"total_tokens": 3},
            finish_reason="stop",
        )


class UseCaseChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_delegates_to_agent_port(self) -> None:
        agent = RecordingAgent()
        use_case = UseCaseChat(agent)

        result = await use_case.execute(
            ChatCommand(
                message="Hello",
                model="qwen",
                user_id="user-1",
                temperature=0.2,
                max_tokens=128,
            )
        )

        self.assertEqual(result.content, "agent reply")
        self.assertEqual(result.usage, {"total_tokens": 3})
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.session_id, agent.session_id)
        self.assertEqual(agent.user_id, "user-1")
        UUID(result.session_id)
        self.assertIsNotNone(agent.request)
        assert agent.request is not None
        self.assertEqual(agent.request.model, "qwen")
        self.assertEqual(agent.request.messages[0].content, "Hello")
        self.assertEqual(agent.request.temperature, 0.2)
        self.assertEqual(agent.request.max_tokens, 128)

    async def test_execute_reuses_supplied_session_id(self) -> None:
        agent = RecordingAgent()
        use_case = UseCaseChat(agent)

        result = await use_case.execute(
            ChatCommand(
                message="Continue",
                model="qwen",
                user_id="user-1",
                session_id="existing-session",
            )
        )

        self.assertEqual(result.session_id, "existing-session")
        self.assertEqual(agent.session_id, "existing-session")

    async def test_execute_rejects_empty_agent_response(self) -> None:
        use_case = UseCaseChat(RecordingAgent(content="  "))

        with self.assertRaises(EmptyAgentResponseError):
            await use_case.execute(
                ChatCommand(
                    message="Hello",
                    model="qwen",
                    user_id="user-1",
                )
            )


if __name__ == "__main__":
    unittest.main()
