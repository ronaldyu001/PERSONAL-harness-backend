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
        memory = request.memory
        metadata = memory.retrieval_attributes()

        if memory.memory_id and request.upsert:
            await asyncio.to_thread(
                self.memory.update,
                memory.memory_id,
                data=memory.content,
                metadata=metadata,
            )
            return MemorySaveResult(memory_id=memory.memory_id, created=False)

        result = await asyncio.to_thread(
            self.memory.add,
            memory.content,
            user_id=memory.user_id,
            run_id=memory.conversation_id,
            metadata=metadata,
            infer=False,
        )

        return MemorySaveResult(memory_id=self._first_id(result), created=True)

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
            threshold=request.min_score or 0.1,
        )

        return MemoryRetrieveResult(
            memories=tuple(
                self._to_retrieved_memory(item, request)
                for item in result.get("results", [])
            )
        )

    @staticmethod
    def _first_id(result: dict[str, Any]) -> str:
        results = result.get("results", [])
        if results and results[0].get("id"):
            return str(results[0]["id"])

        memory_id = result.get("id") or result.get("memory_id")
        if memory_id:
            return str(memory_id)

        raise RuntimeError("Mem0 did not return a memory id")

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
