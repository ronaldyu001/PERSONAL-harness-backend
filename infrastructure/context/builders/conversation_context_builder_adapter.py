"""Conversation context builder implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from application.context.builders.schemas import ConversationContext
from application.context.schemas import ContextBlock, ContextSource
from application.llm.schemas import ChatMessage
from application.memory.memory_port import MemoryPort
from application.memory.schemas import MemoryRetrieveRequest


class ConversationContextBuilder:
    """Build a conversation context block for one assistant turn."""

    def __init__(self, memory: MemoryPort | None = None) -> None:
        self._memory = memory

    async def build(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        user_id: str,
        current_user_message: ChatMessage,
        recent_messages: Sequence[ChatMessage] = (),
        summary: str | None = None,
        memory_limit: int = 5,
    ) -> ContextBlock[ConversationContext]:
        memories = await self._retrieve_memories(
            query=current_user_message.content,
            user_id=user_id,
            conversation_id=conversation_id,
            limit=memory_limit,
        )

        context = ConversationContext(
            conversation_id=conversation_id,
            turn_id=turn_id,
            current_user_message=current_user_message,
            recent_messages=tuple(recent_messages),
            summary=summary,
            metadata={"user_id": user_id, "memories": memories},
        )

        return ContextBlock(
            component="conversation",
            context=context,
            title="Conversation",
            source=ContextSource(
                source_type="conversation",
                source_id=conversation_id,
                metadata={"turn_id": turn_id},
            ),
            priority="high",
            metadata={"memory_count": len(memories)},
        )

    async def _retrieve_memories(
        self,
        *,
        query: str,
        user_id: str,
        conversation_id: str,
        limit: int,
    ) -> tuple[dict[str, Any], ...]:
        if self._memory is None or limit <= 0:
            return ()

        result = await self._memory.retrieve(
            MemoryRetrieveRequest(
                query=query,
                user_id=user_id,
                conversation_id=conversation_id,
                limit=limit,
            )
        )

        return tuple(
            {
                "content": item.memory.content,
                "kind": item.memory.kind,
                "score": item.score,
                "memory_id": item.memory.memory_id,
            }
            for item in result.memories
        )
