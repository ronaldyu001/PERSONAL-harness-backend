"""Tests for API error mapping and conversation history routes."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.conversation import (
    ConversationListRequest,
    ConversationListResult,
    ConversationReadRequest,
    ConversationReadResult,
    ConversationInfo,
)
from application.llm import ChatResponse
from application.use_cases import ChatUseCase, ReadConversationHistoryUseCase
from domain.entities import Conversation, ConversationMessage
from presentation.api import router


class EmptyAgent:
    """Agent test double that returns no user-visible content."""

    async def chat(
        self,
        request,
        *,
        session_id: str,
        user_id: str,
        temporary: bool = False,
    ) -> ChatResponse:
        return ChatResponse(content="")


class RecordingAgent:
    """Agent test double that records how each turn was invoked."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        request,
        *,
        session_id: str,
        user_id: str,
        temporary: bool = False,
    ) -> ChatResponse:
        self.calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "temporary": temporary,
            }
        )
        return ChatResponse(content="Answered.")


class StubConversations:
    """ConversationPort read double."""

    def __init__(self, conversation: Conversation | None = None) -> None:
        self.conversation = conversation
        self.list_requests: list[ConversationListRequest] = []
        self.read_requests: list[ConversationReadRequest] = []

    async def write(self, request):
        raise NotImplementedError

    async def list_conversations(
        self,
        request: ConversationListRequest,
    ) -> ConversationListResult:
        self.list_requests.append(request)
        return ConversationListResult(
            conversations=(
                ConversationInfo(
                    conversation_id="session-1",
                    title="Explain checkpointing",
                    created_at=datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
                    last_updated=datetime(2026, 8, 22, 12, 5, tzinfo=UTC),
                ),
            )
        )

    async def get_conversation(
        self,
        request: ConversationReadRequest,
    ) -> ConversationReadResult:
        self.read_requests.append(request)
        # Ownership scoping lives in the adapter; mimic it here.
        if self.conversation is None or request.user_id != self.conversation.user_id:
            return ConversationReadResult()
        return ConversationReadResult(conversation=self.conversation)


def _conversation() -> Conversation:
    created = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
    return Conversation(
        conversation_id="session-1",
        user_id="user-1",
        title="Explain checkpointing",
        messages=(
            ConversationMessage(
                conversation_id="session-1",
                role="user",
                content="Explain checkpointing",
                message_id="human-1",
                user_id="user-1",
                created_at=created,
            ),
            ConversationMessage(
                conversation_id="session-1",
                role="assistant",
                content="A checkpoint stores state.",
                message_id="ai-1",
                user_id="user-1",
                created_at=created,
                metadata={"model": "qwen"},
            ),
        ),
        created_at=created,
    )


def _history(conversations: StubConversations) -> ReadConversationHistoryUseCase:
    return ReadConversationHistoryUseCase(
        conversations,
        default_list_size=50,
        max_list_size=200,
    )


def _client(app: FastAPI) -> TestClient:
    app.include_router(router)
    return TestClient(app)


class ChatRoutesTests(unittest.TestCase):
    def test_empty_agent_response_maps_to_bad_gateway(self) -> None:
        app = FastAPI()
        app.state.chat_use_case = ChatUseCase(EmptyAgent())

        with _client(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "message": "Hello",
                    "model": "qwen",
                    "user_id": "user-1",
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "The agent returned an empty response."},
        )

    def test_chat_and_temp_chat_differ_only_by_the_flag(self) -> None:
        agent = RecordingAgent()
        app = FastAPI()
        app.state.chat_use_case = ChatUseCase(agent)
        body = {
            "message": "Hello",
            "model": "qwen",
            "user_id": "user-1",
            "session_id": "session-1",
        }

        with _client(app) as client:
            self.assertEqual(client.post("/api/chat", json=body).status_code, 200)
            self.assertEqual(client.post("/api/temp-chat", json=body).status_code, 200)

        self.assertEqual([call["temporary"] for call in agent.calls], [False, True])
        for call in agent.calls:
            self.assertEqual(call["session_id"], "session-1")
            self.assertEqual(call["user_id"], "user-1")


class ConversationHistoryRoutesTests(unittest.TestCase):
    def test_list_returns_summaries_for_the_user(self) -> None:
        conversations = StubConversations()
        app = FastAPI()
        app.state.read_conversation_history_use_case = _history(conversations)

        with _client(app) as client:
            response = client.get("/api/conversations", params={"user_id": "user-1"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["conversation_id"], "session-1")
        self.assertEqual(payload[0]["title"], "Explain checkpointing")
        self.assertNotIn("messages", payload[0])
        self.assertEqual(conversations.list_requests[0].user_id, "user-1")

    def test_limit_falls_back_to_configuration_and_is_clamped(self) -> None:
        conversations = StubConversations()
        app = FastAPI()
        app.state.read_conversation_history_use_case = ReadConversationHistoryUseCase(
            conversations,
            default_list_size=25,
            max_list_size=100,
        )

        with _client(app) as client:
            omitted = client.get("/api/conversations", params={"user_id": "user-1"})
            oversized = client.get(
                "/api/conversations",
                params={"user_id": "user-1", "limit": 5000},
            )
            rejected = client.get(
                "/api/conversations",
                params={"user_id": "user-1", "limit": 0},
            )

        self.assertEqual(omitted.status_code, 200)
        self.assertEqual(oversized.status_code, 200)
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(
            [request.limit for request in conversations.list_requests],
            [25, 100],
        )

    def test_detail_returns_messages_in_order(self) -> None:
        app = FastAPI()
        app.state.read_conversation_history_use_case = _history(
            StubConversations(_conversation())
        )

        with _client(app) as client:
            response = client.get(
                "/api/conversations/session-1",
                params={"user_id": "user-1"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [message["role"] for message in payload["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(payload["messages"][1]["metadata"], {"model": "qwen"})

    def test_detail_hides_a_conversation_owned_by_someone_else(self) -> None:
        app = FastAPI()
        app.state.read_conversation_history_use_case = _history(
            StubConversations(_conversation())
        )

        with _client(app) as client:
            response = client.get(
                "/api/conversations/session-1",
                params={"user_id": "someone-else"},
            )

        self.assertEqual(response.status_code, 404)

    def test_history_degrades_when_persistence_is_disabled(self) -> None:
        app = FastAPI()
        app.state.read_conversation_history_use_case = None

        with _client(app) as client:
            listed = client.get("/api/conversations", params={"user_id": "user-1"})
            detail = client.get(
                "/api/conversations/session-1",
                params={"user_id": "user-1"},
            )

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json(), [])
        self.assertEqual(detail.status_code, 404)


if __name__ == "__main__":
    unittest.main()
