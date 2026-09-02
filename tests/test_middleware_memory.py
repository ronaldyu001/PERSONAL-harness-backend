"""Tests for transient durable-memory context injection."""

from __future__ import annotations

import unittest

from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.runtime import Runtime

from application.agent import AgentMessage, AgentRequest
from application.memory import (
    MemoryRetrieveRequest,
    MemoryRetrieveResult,
    MemorySaveRequest,
    MemorySaveResult,
    RetrievedMemory,
)
from domain.entities import Memory
from infrastructure.agent.adapter_langchain import LangChainAdapter
from infrastructure.agent.middleware import MemoryMiddleware
from infrastructure.agent.middleware.helpers import USER_MEMORIES_MESSAGE_NAME
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.settings import load_infrastructure_settings


class RecordingMemory:
    """MemoryPort retrieval test double."""

    def __init__(self) -> None:
        self.request: MemoryRetrieveRequest | None = None
        self.saved: list[MemorySaveRequest] = []

    async def retrieve(
        self,
        request: MemoryRetrieveRequest,
    ) -> MemoryRetrieveResult:
        self.request = request
        return MemoryRetrieveResult(
            memories=(
                RetrievedMemory(
                    memory=Memory(
                        content="Prefers concise technical explanations",
                        user_id=request.user_id or "",
                        kind="preference",
                    )
                ),
            )
        )

    async def save(self, request: MemorySaveRequest):
        self.saved.append(request)
        return MemorySaveResult()


class MemoryMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrap_model_call_injects_memory_transiently(self) -> None:
        memory = RecordingMemory()
        settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://litellm.test",
        })
        memory_config = settings.agent.memory.model_copy(
            update={"retrieval_limit": 3}
        )
        middleware = MemoryMiddleware.from_config(
            memory_config,
            memory=memory,
        )
        original_messages = [HumanMessage(content="  Explain checkpointing  ")]
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="unused")]
        )
        request = ModelRequest(
            model=model,
            messages=original_messages,
            runtime=Runtime(
                context=AgentRuntimeContext(
                    user_id="user-1",
                    session_id="session-1",
                ),
            ),
        )
        received_messages = []

        async def handler(enriched_request):
            received_messages.extend(enriched_request.messages)
            return AIMessage(content="response")

        response = await middleware.awrap_model_call(request, handler)

        self.assertEqual(response.text, "response")
        self.assertIsNotNone(memory.request)
        assert memory.request is not None
        self.assertEqual(memory.request.user_id, "user-1")
        self.assertEqual(memory.request.query, "Explain checkpointing")
        self.assertEqual(memory.request.limit, 3)
        self.assertIsInstance(received_messages[0], SystemMessage)
        self.assertIn(
            "Prefers concise technical explanations",
            received_messages[0].text,
        )
        # Named so the response gate can recognize this block on the request.
        # Nothing writes it to agent state, so that is its only chance to.
        self.assertEqual(received_messages[0].name, USER_MEMORIES_MESSAGE_NAME)
        self.assertEqual(received_messages[1:], original_messages)
        self.assertEqual(request.messages, original_messages)

    async def test_after_agent_submits_completed_turn_once(self) -> None:
        memory = RecordingMemory()
        middleware = MemoryMiddleware(memory, limit=5)
        runtime = Runtime(
            context=AgentRuntimeContext(
                user_id="user-1",
                session_id="session-1",
            )
        )

        result = await middleware.aafter_agent(
            {
                "messages": [
                    HumanMessage(content="Please keep answers concise"),
                    AIMessage(
                        content="I will keep my answers concise.",
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 6,
                            "total_tokens": 11,
                        },
                    ),
                ]
            },
            runtime,
        )

        self.assertIsNone(result)
        self.assertEqual(len(memory.saved), 1)
        request = memory.saved[0]
        self.assertEqual(request.user_id, "user-1")
        self.assertEqual(
            request.conversation_id,
            "session-1",
        )
        self.assertEqual(
            request.user_message.content,
            "Please keep answers concise",
        )
        self.assertEqual(
            request.assistant_response.content,
            "I will keep my answers concise.",
        )

    async def test_temporary_turn_retrieves_but_does_not_save(self) -> None:
        memory = RecordingMemory()
        middleware = MemoryMiddleware(memory, limit=5)
        runtime = Runtime(
            context=AgentRuntimeContext(
                user_id="user-1",
                session_id="session-1",
                temporary=True,
            )
        )

        result = await middleware.aafter_agent(
            {
                "messages": [
                    HumanMessage(content="Please keep answers concise"),
                    AIMessage(content="I will keep my answers concise."),
                ]
            },
            runtime,
        )

        self.assertIsNone(result)
        self.assertEqual(memory.saved, [])

        # Retrieval is a separate hook and must still run, so Maia stays herself
        # inside a temporary chat.
        model = FakeMessagesListChatModel(responses=[AIMessage(content="unused")])
        received_messages: list = []

        async def handler(request):
            received_messages.extend(request.messages)
            return AIMessage(content="unused")

        await middleware.awrap_model_call(
            ModelRequest(
                model=model,
                messages=[HumanMessage(content="Explain checkpointing")],
                runtime=runtime,
            ),
            handler,
        )

        self.assertIsInstance(received_messages[0], SystemMessage)
        self.assertIn(
            "Prefers concise technical explanations",
            received_messages[0].text,
        )

    async def test_after_agent_does_not_save_empty_response(self) -> None:
        memory = RecordingMemory()
        middleware = MemoryMiddleware(memory, limit=5)
        runtime = Runtime(
            context=AgentRuntimeContext(
                user_id="user-1",
                session_id="session-1",
            )
        )

        result = await middleware.aafter_agent(
            {
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="  "),
                ]
            },
            runtime,
        )

        self.assertIsNone(result)
        self.assertEqual(memory.saved, [])

    async def test_adapter_runs_after_agent_hook_automatically(self) -> None:
        memory = RecordingMemory()
        model = FakeMessagesListChatModel(
            responses=[AIMessage(content="I will keep my answers concise.")]
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
        adapter = LangChainAdapter.from_config(
            settings,
            model_factory=lambda _: model,
            memory=memory,
        )

        await adapter.chat(
            AgentRequest(
                model="test-model",
                messages=(
                    AgentMessage(
                        role="user",
                        content="Please keep answers concise",
                    ),
                ),
            ),
            session_id="session-1",
            user_id="user-1",
        )

        self.assertEqual(len(memory.saved), 1)
        self.assertEqual(memory.saved[0].user_id, "user-1")


if __name__ == "__main__":
    unittest.main()
