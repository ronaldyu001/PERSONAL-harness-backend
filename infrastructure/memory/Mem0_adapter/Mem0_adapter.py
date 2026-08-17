"""Mem0 implementation of the application memory port."""

from __future__ import annotations

import asyncio
from typing import Any

from application.memory.schemas import (
    MemoryRetrieveRequest,
    MemoryRetrieveResult,
    MemorySaveRequest,
    MemorySaveResult,
    RetrievedMemory,
)
from domain.entities.memory import Memory as DomainMemory
from mem0.memory.main import Memory as Mem0Memory

from infrastructure.memory.Mem0_adapter.Mem0_config import build_memory_config


class Mem0Adapter:
    """Thin adapter around Mem0's memory client."""

    def __init__(self, memory: Mem0Memory | None = None) -> None:
        self.memory = memory or Mem0Memory(config=build_memory_config())

    async def save(self, request: MemorySaveRequest) -> MemorySaveResult:
        """Let Mem0 infer zero or more durable memories from a completed turn."""
        metadata = dict(request.metadata)
        if request.conversation_id:
            metadata["conversation_id"] = request.conversation_id

        result = await asyncio.to_thread(
            self.memory.add,
            [
                {
                    "role": "user",
                    "content": request.user_message.content,
                },
                {
                    "role": "assistant",
                    "content": request.assistant_response.content,
                },
            ],
            user_id=request.user_id,
            run_id=request.conversation_id,
            metadata=metadata,
            infer=True,
        )

        return MemorySaveResult(
            memories=tuple(
                self._to_saved_memory(item, request)
                for item in result.get("results", [])
                if item.get("id") and item.get("memory")
            )
        )

    async def retrieve(self, request: MemoryRetrieveRequest) -> MemoryRetrieveResult:
        if not request.user_id:
            raise ValueError("retrieve requires user_id for Mem0")

        filters = dict(request.metadata)
        filters["user_id"] = request.user_id
        if request.conversation_id:
            filters["run_id"] = request.conversation_id

        result = await asyncio.to_thread(
            self.memory.search,
            request.query,
            top_k=request.limit,
            filters=filters,
            threshold=(
                request.min_score if request.min_score is not None else 0.1
            ),
        )

        return MemoryRetrieveResult(
            memories=tuple(
                self._to_retrieved_memory(item, request)
                for item in result.get("results", [])
            )
        )

    @staticmethod
    def _to_saved_memory(
        item: dict[str, Any], request: MemorySaveRequest
    ) -> DomainMemory:
        """Map a Mem0 extraction result into the domain memory entity."""
        item_metadata = item.get("metadata") or {}
        metadata = {
            **request.metadata,
            **item_metadata,
            "event": item.get("event", "ADD"),
        }

        return DomainMemory(
            content=str(item["memory"]),
            user_id=request.user_id,
            kind="other",
            memory_id=str(item["id"]),
            conversation_id=request.conversation_id,
            metadata=metadata,
        )

    @staticmethod
    def _to_retrieved_memory(
        item: dict[str, Any], request: MemoryRetrieveRequest
    ) -> RetrievedMemory:
        metadata = item.get("metadata") or {}

        memory = DomainMemory(
            content=item.get("memory", ""),
            user_id=metadata.get("user_id", request.user_id),
            kind=metadata.get("kind", "other"),
            memory_id=item.get("id"),
            conversation_id=metadata.get("conversation_id", request.conversation_id),
            source_message_ids=tuple(metadata.get("source_message_ids", ())),
            confidence=metadata.get("confidence"),
            metadata=metadata,
        )

        return RetrievedMemory(memory=memory, score=item.get("score"))
