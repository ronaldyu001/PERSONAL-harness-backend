"""Tests for the LangChain agent adapter."""

from __future__ import annotations

import unittest

from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.checkpoint.memory import InMemorySaver

from application.llm import ChatMessage, ChatRequest
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
