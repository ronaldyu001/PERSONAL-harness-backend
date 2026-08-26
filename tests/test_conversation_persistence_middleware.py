"""Tests for writing completed turns to the conversation port."""

from __future__ import annotations

import unittest

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.runtime import Runtime

from application.conversation import (
    ConversationWriteRequest,
    ConversationWriteResult,
)
from application.llm import ChatMessage, ChatRequest
from infrastructure.agent.adapter_langchain import LangChainAdapter
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.agent.middleware import ConversationPersistenceMiddleware
from infrastructure.conversation.adapter_postgres import (
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


def _runtime(*, temporary: bool = False, model: str | None = "qwen") -> Runtime:
    return Runtime(
        context=AgentRuntimeContext(
            user_id="user-1",
            session_id="session-1",
            model=model,
            temporary=temporary,
        )
    )


def _turn() -> dict[str, list[object]]:
    return {
        "messages": [
            HumanMessage(content="Explain checkpointing", id="human-1"),
            AIMessage(content="A checkpoint stores state.", id="ai-1"),
        ]
    }


def _gateless_settings():
    """Settings with the response gate off, so one fake reply is enough."""
    settings = load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm.test",
    })
    gate = settings.agent.response_gate.model_copy(update={"enabled": False})
    return settings.model_copy(update={
        "agent": settings.agent.model_copy(update={"response_gate": gate})
    })


async def _run_turn(settings, conversations, *, temporary: bool = False):
    """Run one real agent turn against a fake model."""
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="A checkpoint stores state.")]
    )
    adapter = LangChainAdapter.from_config(
        settings,
        model_factory=lambda _: model,
        conversations=conversations,
    )

    return await adapter.chat(
        ChatRequest(
            model="test-model",
            messages=(ChatMessage(role="user", content="Explain checkpointing"),),
        ),
        session_id="session-1",
        user_id="user-1",
        temporary=temporary,
    )


class ConversationPersistenceMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_hooks_write_the_user_then_the_assistant(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)
        runtime = _runtime()
        state = {
            "messages": [
                HumanMessage(content="Explain checkpointing", id="human-1"),
                AIMessage(content="A checkpoint stores state.", id="ai-1"),
            ]
        }

        self.assertIsNone(await middleware.abefore_agent(state, runtime))
        # The conversation row exists from here on, so anything else recorded
        # during the turn has a conversation to belong to.
        self.assertEqual(len(conversations.written), 1)

        self.assertIsNone(await middleware.aafter_agent(state, runtime))
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

    async def test_before_agent_writes_only_the_user_message(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        await middleware.abefore_agent(
            {"messages": [HumanMessage(content="Explain checkpointing", id="human-1")]},
            _runtime(),
        )

        written = [request.message for request in conversations.written]
        self.assertEqual([message.role for message in written], ["user"])

    async def test_hooks_skip_tool_steps(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)
        runtime = _runtime()
        state = {
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
        }

        await middleware.abefore_agent(state, runtime)
        await middleware.aafter_agent(state, runtime)

        written = [request.message for request in conversations.written]
        self.assertEqual([message.role for message in written], ["user", "assistant"])
        self.assertEqual(
            [message.message_id for message in written],
            ["human-1", "ai-final"],
        )

    async def test_a_tool_call_only_reply_still_records_the_user_message(self) -> None:
        # The empty-assistant guard suppresses the reply, not the question.
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)
        runtime = _runtime()
        state = {
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
            ]
        }

        await middleware.abefore_agent(state, runtime)
        await middleware.aafter_agent(state, runtime)

        written = [request.message for request in conversations.written]
        self.assertEqual([message.role for message in written], ["user"])

    async def test_before_agent_swallows_port_failures(self) -> None:
        middleware = ConversationPersistenceMiddleware(FailingConversations())

        with self.assertLogs(
            "infrastructure.agent.middleware.middleware_conversation",
            level="ERROR",
        ):
            result = await middleware.abefore_agent(
                {"messages": [HumanMessage(content="Hello", id="human-1")]},
                _runtime(),
            )

        self.assertIsNone(result)

    async def test_before_agent_writes_nothing_for_a_temporary_turn(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        result = await middleware.abefore_agent(_turn(), _runtime(temporary=True))

        self.assertIsNone(result)
        self.assertEqual(conversations.written, [])

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
            "infrastructure.agent.middleware.middleware_conversation",
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


    async def test_assistant_message_records_the_model(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        runtime = _runtime(model="qwen")
        await middleware.abefore_agent(_turn(), runtime)
        await middleware.aafter_agent(_turn(), runtime)

        user_message, assistant_message = (
            request.message for request in conversations.written
        )
        self.assertEqual(assistant_message.metadata, {"model": "qwen"})
        # The user did not choose a model, so their message records none.
        self.assertEqual(user_message.metadata, {})

    async def test_unknown_model_is_omitted_rather_than_stored_as_null(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        runtime = _runtime(model=None)
        await middleware.abefore_agent(_turn(), runtime)
        await middleware.aafter_agent(_turn(), runtime)

        assistant_message = conversations.written[1].message
        self.assertEqual(assistant_message.metadata, {})

    async def test_temporary_turn_writes_nothing(self) -> None:
        conversations = RecordingConversations()
        middleware = ConversationPersistenceMiddleware(conversations)

        result = await middleware.aafter_agent(_turn(), _runtime(temporary=True))

        self.assertIsNone(result)
        self.assertEqual(conversations.written, [])

    async def test_adapter_runs_after_agent_hook_automatically(self) -> None:
        conversations = RecordingConversations()
        settings = _gateless_settings()
        # No POSTGRES_DSN above, so persistence stays off unless injected.
        self.assertIsNone(settings.postgres.dsn)

        await _run_turn(settings, conversations)

        self.assertEqual(len(conversations.written), 2)
        self.assertEqual(
            [request.message.role for request in conversations.written],
            ["user", "assistant"],
        )
        self.assertEqual(
            conversations.written[0].message.conversation_id,
            "session-1",
        )
        # The model reaches the transcript from the request, through context.
        self.assertEqual(
            conversations.written[1].message.metadata,
            {"model": "test-model"},
        )

    async def test_adapter_writes_nothing_for_a_temporary_turn(self) -> None:
        conversations = RecordingConversations()

        response = await _run_turn(
            _gateless_settings(),
            conversations,
            temporary=True,
        )

        # The turn is answered normally; only the transcript is suppressed.
        self.assertEqual(response.content, "A checkpoint stores state.")
        self.assertEqual(conversations.written, [])


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
