"""Tests for the application chat flow and LangChain agent adapter."""

from __future__ import annotations

import unittest
from uuid import UUID

from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.checkpoint.memory import InMemorySaver

from application.agent import EmptyAgentResponseError
from application.llm import ChatMessage, ChatRequest, ChatResponse
from application.use_cases import ChatCommand, ChatUseCase
from infrastructure.agent import LangChainAdapter
from infrastructure.settings import load_infrastructure_settings


def infrastructure_settings(*, response_gate_enabled: bool = True):
    settings = load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm:4000",
    })
    gate = settings.agent.response_gate.model_copy(
        update={"enabled": response_gate_enabled}
    )
    agent = settings.agent.model_copy(update={"response_gate": gate})
    return settings.model_copy(update={"agent": agent})


class RecordingAgent:
    """Small AgentPort test double."""

    def __init__(self, content: str = "agent reply") -> None:
        self.content = content
        self.request: ChatRequest | None = None
        self.session_id: str | None = None
        self.user_id: str | None = None
        self.temporary: bool | None = None

    async def chat(
        self,
        request: ChatRequest,
        *,
        session_id: str,
        user_id: str,
        temporary: bool = False,
    ) -> ChatResponse:
        self.request = request
        self.session_id = session_id
        self.user_id = user_id
        self.temporary = temporary
        return ChatResponse(
            content=self.content,
            usage={"total_tokens": 3},
            finish_reason="stop",
        )


class ChatUseCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_delegates_to_agent_port(self) -> None:
        agent = RecordingAgent()
        use_case = ChatUseCase(agent)

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
        use_case = ChatUseCase(agent)

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
        use_case = ChatUseCase(RecordingAgent(content="  "))

        with self.assertRaises(EmptyAgentResponseError):
            await use_case.execute(
                ChatCommand(
                    message="Hello",
                    model="qwen",
                    user_id="user-1",
                )
            )


class LangChainAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_from_settings_enables_response_gate(self) -> None:
        adapter = LangChainAdapter.from_config(infrastructure_settings())

        self.assertTrue(adapter._agent_config.response_gate.enabled)

    def test_from_settings_registers_search_only_when_key_is_present(self) -> None:
        without_key = LangChainAdapter.from_config(infrastructure_settings())
        with_key_settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://litellm:4000",
            "LANGSEARCH_API_KEY": "secret",
        })
        with_key = LangChainAdapter.from_config(with_key_settings)

        self.assertEqual(without_key._tools, ())
        self.assertEqual([tool.name for tool in with_key._tools], ["search_web"])

    async def test_chat_maps_agent_response(self) -> None:
        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(
                    content="LangChain reply",
                    response_metadata={
                        "finish_reason": "provider_length_limit"
                    },
                    usage_metadata={
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                    },
                )
            ]
        )
        adapter = LangChainAdapter.from_config(
            infrastructure_settings(response_gate_enabled=False),
            model_factory=lambda request: model,
        )

        response = await adapter.chat(
            ChatRequest(
                model="qwen",
                messages=(),
            ),
            session_id="session-1",
            user_id="user-1",
        )

        self.assertEqual(response.content, "LangChain reply")
        self.assertEqual(response.usage, {
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        })
        self.assertEqual(response.finish_reason, "provider_length_limit")

    async def test_chat_continues_a_named_conversation(self) -> None:
        checkpointer = InMemorySaver()
        model = FakeMessagesListChatModel(
            responses=[
                AIMessage(content="First reply"),
                AIMessage(content="Second reply"),
            ]
        )
        adapter = LangChainAdapter.from_config(
            infrastructure_settings(response_gate_enabled=False),
            checkpointer=checkpointer,
            model_factory=lambda request: model,
        )

        for content in ("First question", "Second question"):
            await adapter.chat(
                ChatRequest(
                    model="qwen",
                    messages=(ChatMessage(role="user", content=content),),
                ),
                session_id="session-1",
                user_id="user-1",
            )

        checkpoint = checkpointer.get(
            {"configurable": {"thread_id": "session-1"}}
        )
        assert checkpoint is not None
        messages = checkpoint["channel_values"]["messages"]
        self.assertEqual(
            [message.text for message in messages],
            [
                "First question",
                "First reply",
                "Second question",
                "Second reply",
            ],
        )


if __name__ == "__main__":
    unittest.main()
