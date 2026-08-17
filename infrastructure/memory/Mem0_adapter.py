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
from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.memory.main import Memory as Mem0Memory
from mem0.memory.main import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig

from infrastructure.settings import Mem0Config


class Mem0Adapter:
    """Thin adapter around Mem0's memory client."""

    def __init__(
        self,
        memory: Mem0Memory,
    ) -> None:
        self.memory = memory

    @classmethod
    def from_config(cls, config: Mem0Config) -> Mem0Adapter:
        """Build the Mem0 client from its resolved config section."""
        return cls(Mem0Memory(config=_build_memory_config(config)))

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


def _build_memory_config(config: Mem0Config) -> MemoryConfig:
    """Translate Maia's Mem0 settings at the external SDK boundary."""
    return MemoryConfig(
        vector_store=_build_vector_store_config(config),
        llm=_build_llm_config(config),
        embedder=_build_embedder_config(config),
        history_db_path=str(config.history_db_path),
        reranker=None,
        custom_instructions=config.custom_instructions,
    )


def _build_llm_config(settings: Mem0Config) -> LlmConfig:
    config: dict[str, object] = {
        "model": settings.llm.model,
        "api_key": settings.llm.api_key,
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
        "top_p": settings.llm.top_p,
        "top_k": settings.llm.top_k,
        "openai_base_url": _openai_base_url(settings.llm.base_url),
    }
    return LlmConfig(provider="openai", config=config)


def _openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _build_embedder_config(settings: Mem0Config) -> EmbedderConfig:
    config: dict[str, object] = {
        "model": settings.embedder.model,
        "embedding_dims": settings.embedder.dimensions,
    }
    if settings.embedder.base_url:
        config["ollama_base_url"] = settings.embedder.base_url
    return EmbedderConfig(provider="ollama", config=config)


def _build_vector_store_config(settings: Mem0Config) -> VectorStoreConfig:
    config: dict[str, object] = {
        "collection_name": settings.collection_name,
        "embedding_model_dims": settings.embedder.dimensions,
        "on_disk": settings.vector_store.on_disk,
    }
    if settings.vector_store.url:
        config["url"] = settings.vector_store.url
        if settings.vector_store.api_key:
            config["api_key"] = settings.vector_store.api_key
    elif settings.vector_store.host:
        config["host"] = settings.vector_store.host
        config["port"] = settings.vector_store.port
    else:
        config["path"] = str(settings.vector_store.local_path)
    return VectorStoreConfig(provider="qdrant", config=config)
