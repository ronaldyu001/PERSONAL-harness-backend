"""Build Mem0 SDK configuration from Harness infrastructure settings."""

from __future__ import annotations

from mem0.embeddings.configs import EmbedderConfig
from mem0.llms.configs import LlmConfig
from mem0.memory.main import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings


def build_memory_config(settings: Mem0Settings | None = None) -> MemoryConfig:
    """Build the Mem0 configuration used by the memory adapter."""
    settings = settings or Mem0Settings.from_env()

    return MemoryConfig(
        vector_store=_build_vector_store_config(settings),
        llm=_build_llm_config(settings),
        embedder=_build_embedder_config(settings),
        history_db_path=settings.history_db_path,
        reranker=None,
        custom_instructions=settings.custom_instructions,
    )


def _build_llm_config(settings: Mem0Settings) -> LlmConfig:
    """Configure the Ollama model Mem0 uses for memory inference."""
    config: dict[str, object] = {
        "model": settings.llm_model,
        "temperature": 0.1,
        "max_tokens": 512,
        "top_p": 0.9,
        "top_k": 40,
    }
    if settings.ollama_base_url:
        config["ollama_base_url"] = settings.ollama_base_url

    return LlmConfig(provider="ollama", config=config)


def _build_embedder_config(settings: Mem0Settings) -> EmbedderConfig:
    """Configure the Ollama embedding model used by Mem0."""
    config: dict[str, object] = {
        "model": settings.embedder_model,
        "embedding_dims": settings.embedding_dims,
    }
    if settings.ollama_base_url:
        config["ollama_base_url"] = settings.ollama_base_url

    return EmbedderConfig(provider="ollama", config=config)


def _build_vector_store_config(settings: Mem0Settings) -> VectorStoreConfig:
    """Configure Qdrant for remote deployment or local development."""
    config: dict[str, object] = {
        "collection_name": settings.collection_name,
        "embedding_model_dims": settings.embedding_dims,
        "on_disk": False,
    }

    if settings.qdrant_url:
        config["url"] = settings.qdrant_url
        if settings.qdrant_api_key:
            config["api_key"] = settings.qdrant_api_key
    elif settings.qdrant_host:
        config["host"] = settings.qdrant_host
        config["port"] = settings.qdrant_port
    else:
        config["path"] = settings.local_qdrant_path

    return VectorStoreConfig(provider="qdrant", config=config)
