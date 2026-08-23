"""Tests for writing completed turns to the conversation port."""

from __future__ import annotations

import unittest

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.runtime import Runtime

from application.conversation.schemas import (
    ConversationWriteRequest,
    ConversationWriteResult,
)
from application.llm.schemas import ChatMessage, ChatRequest
from infrastructure.agent.LanchChain_adapter import LangChainAdapter
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.agent.middleware import ConversationPersistenceMiddleware
from infrastructure.conversation.Postgres_adapter import (
    UNTITLED_CONVERSATION,
    truncate_title,
)
from infrastructure.settings import load_infrastructure_settings


class RecordingConversations:
    """ConversationPort test double."""

    def __init__(self) -> None:
        self.written: list[ConversationWriteRequest] = []

    async def write(
        self,
        request: ConversationWriteRequest,
    ) -> ConversationWriteResult:
        self.written.append(request)
        return ConversationWriteResult(
            message_id=request.message.message_id or "generated",
            conversation_id=request.message.conversation_id,
        )


class FailingConversations:
    """ConversationPort double standing in for a database outage."""

    async def write(
        self,
        request: ConversationWriteRequest,
    ) -> ConversationWriteResult:
        raise RuntimeError("postgres is unreachable")


def _runtime() -> Runtime:
    return Runtime(
        context=AgentRuntimeContext(
            user_id="user-1",
            session_id="session-1",
        )
    )


class ConversationPersistenceMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_after_agent_writes_user_then_assistant(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        result = await middleware.aafter_agent(
            {
                "messages": [
                    HumanMessage(content="Explain checkpointing", id="human-1"),
                    AIMessage(content="A checkpoint stores state.", id="ai-1"),
                ]
            },
            _runtime(),
        )

        self.assertIsNone(result)
        self.assertEqual(len(conversations.written), 2)

        user_message = conversations.written[0].message
        assistant_message = conversations.written[1].message
        self.assertEqual(user_message.role, "user")
        self.assertEqual(user_message.content, "Explain checkpointing")
        self.assertEqual(user_message.message_id, "human-1")
        self.assertEqual(assistant_message.role, "assistant")
        self.assertEqual(assistant_message.content, "A checkpoint stores state.")
        self.assertEqual(assistant_message.message_id, "ai-1")

        for message in (user_message, assistant_message):
            self.assertEqual(message.conversation_id, "session-1")
            self.assertEqual(message.user_id, "user-1")

    async def test_after_agent_skips_tool_steps(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        await middleware.aafter_agent(
            {
                "messages": [
                    HumanMessage(content="Weather in Denver?", id="human-1"),
                    AIMessage(
                        content="",
                        id="ai-tool-call",
                        tool_calls=[
                            {
                                "name": "search_web",
                                "args": {"query": "denver weather"},
                                "id": "call-1",
                            }
                        ],
                    ),
                    ToolMessage(content="Sunny, 24C", tool_call_id="call-1"),
                    AIMessage(content="It is sunny and 24C.", id="ai-final"),
                ]
            },
            _runtime(),
        )

        written = [request.message for request in conversations.written]
        self.assertEqual([message.role for message in written], ["user", "assistant"])
        self.assertEqual(
            [message.message_id for message in written],
            ["human-1", "ai-final"],
        )

    async def test_after_agent_writes_nothing_for_empty_response(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        result = await middleware.aafter_agent(
            {
                "messages": [
                    HumanMessage(content="Hello", id="human-1"),
                    AIMessage(content="  ", id="ai-1"),
                ]
            },
            _runtime(),
        )

        self.assertIsNone(result)
        self.assertEqual(conversations.written, [])

    async def test_after_agent_swallows_port_failures(self) -> None:
        middleware = ConversationPersistenceMiddleware(FailingConversations())

        with self.assertLogs(
            "infrastructure.agent.middleware.conversation_middleware",
            level="ERROR",
        ):
            result = await middleware.aafter_agent(
                {
                    "messages": [
                        HumanMessage(content="Hello", id="human-1"),
                        AIMessage(content="Hi there.", id="ai-1"),
                    ]
                },
                _runtime(),
            )

        self.assertIsNone(result)


    async def test_adapter_runs_after_agent_hook_automatically(self) -> None:
        conversations = RecordingConversations()
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="A checkpoint stores state.")]
        )
        settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://litellm.test",
        })
        gate = settings.agent.response_gate.model_copy(
            update={"enabled": False}
        )
        settings = settings.model_copy(update={
            "agent": settings.agent.model_copy(
                update={"response_gate": gate}
            )
        })
        # No POSTGRES_DSN above, so persistence stays off unless injected.
        self.assertIsNone(settings.postgres.dsn)

        adapter = LangChainAdapter.from_config(
            settings,
            model_factory=lambda _: model,
            conversations=conversations,
        )

        await adapter.chat(
            ChatRequest(
                model="test-model",
                messages=(
                    ChatMessage(role="user", content="Explain checkpointing"),
                ),
            ),
            session_id="session-1",
            user_id="user-1",
        )

        self.assertEqual(len(conversations.written), 2)
        self.assertEqual(
            [request.message.role for request in conversations.written],
            ["user", "assistant"],
        )
        self.assertEqual(
            conversations.written[0].message.conversation_id,
            "session-1",
        )


class TitleDerivationTests(unittest.TestCase):
    def test_short_message_is_kept_whole(self) -> None:
        self.assertEqual(truncate_title("Explain checkpointing"), "Explain checkpointing")

    def test_long_message_is_cut_on_a_word_boundary(self) -> None:
        title = truncate_title(
            "Explain how LangGraph checkpointers persist conversation state"
        )
        self.assertEqual(title, "Explain how LangGraph checkpointers persist…")
        self.assertLessEqual(len(title.rstrip("…")), 44)

    def test_single_long_word_is_cut_at_the_limit(self) -> None:
        title = truncate_title("x" * 60)
        self.assertEqual(title, f"{'x' * 44}…")

    def test_blank_message_falls_back(self) -> None:
        self.assertEqual(truncate_title("   "), UNTITLED_CONVERSATION)

    def test_whitespace_is_collapsed_to_one_line(self) -> None:
        self.assertEqual(truncate_title("Explain\n  checkpointing"), "Explain checkpointing")


if __name__ == "__main__":
    unittest.main()
