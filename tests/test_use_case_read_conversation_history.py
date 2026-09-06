"""Tests for reading conversation history back through the port."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from application.conversation import (
    ConversationListRequest,
    ConversationListResult,
    ConversationReadRequest,
    ConversationReadResult,
    ConversationInfo,
)
from application.use_cases import UseCaseReadConversationHistory
from domain.entities import Conversation, ConversationMessage


class RecordingConversations:
    """PortConversation read double that records what it was asked."""

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
        return ConversationReadResult(conversation=self.conversation)


def _use_case(
    conversations: RecordingConversations,
    *,
    default_list_size: int = 50,
    max_list_size: int = 200,
) -> UseCaseReadConversationHistory:
    return UseCaseReadConversationHistory(
        conversations,
        default_list_size=default_list_size,
        max_list_size=max_list_size,
    )


class UseCaseReadConversationHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_passes_the_request_through(self) -> None:
        conversations = RecordingConversations()

        result = await _use_case(conversations).list(
            ConversationListRequest(user_id="user-1", limit=10)
        )

        self.assertEqual(len(result.conversations), 1)
        self.assertEqual(result.conversations[0].conversation_id, "session-1")
        self.assertEqual(conversations.list_requests[0].user_id, "user-1")
        self.assertEqual(conversations.list_requests[0].limit, 10)

    async def test_unspecified_limit_uses_the_configured_default(self) -> None:
        conversations = RecordingConversations()

        await _use_case(conversations, default_list_size=25).list(
            ConversationListRequest(user_id="user-1")
        )

        self.assertEqual(conversations.list_requests[0].limit, 25)

    async def test_oversized_limit_is_clamped_to_the_configured_ceiling(self) -> None:
        conversations = RecordingConversations()

        await _use_case(conversations, max_list_size=100).list(
            ConversationListRequest(user_id="user-1", limit=5000)
        )

        # Stores always receive a concrete, bounded count.
        self.assertEqual(conversations.list_requests[0].limit, 100)

    def test_a_default_above_the_ceiling_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _use_case(
                RecordingConversations(),
                default_list_size=500,
                max_list_size=100,
            )

    async def test_get_returns_the_conversation(self) -> None:
        conversation = Conversation(
            conversation_id="session-1",
            user_id="user-1",
            title="Explain checkpointing",
            messages=(
                ConversationMessage(
                    conversation_id="session-1",
                    role="user",
                    content="Explain checkpointing",
                    message_id="human-1",
                ),
            ),
        )
        use_case = _use_case(RecordingConversations(conversation))

        result = await use_case.get(
            ConversationReadRequest(conversation_id="session-1", user_id="user-1")
        )

        self.assertIsNotNone(result.conversation)
        self.assertEqual(result.conversation.conversation_id, "session-1")
        self.assertTrue(result.conversation.has_messages)

    async def test_get_returns_nothing_when_the_store_has_no_match(self) -> None:
        use_case = _use_case(RecordingConversations())

        result = await use_case.get(
            ConversationReadRequest(conversation_id="missing", user_id="user-1")
        )

        self.assertIsNone(result.conversation)


class ConversationRequestValidationTests(unittest.TestCase):
    def test_list_rejects_a_blank_user(self) -> None:
        with self.assertRaises(ValueError):
            ConversationListRequest(user_id="   ")

    def test_list_rejects_a_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            ConversationListRequest(user_id="user-1", limit=0)

    def test_list_allows_an_unspecified_limit(self) -> None:
        self.assertIsNone(ConversationListRequest(user_id="user-1").limit)

    def test_read_requires_both_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            ConversationReadRequest(conversation_id="", user_id="user-1")
        with self.assertRaises(ValueError):
            ConversationReadRequest(conversation_id="session-1", user_id="")


if __name__ == "__main__":
    unittest.main()
