"""Mem0 vector store configuration builder."""

from __future__ import annotations

from mem0.vector_stores.configs import VectorStoreConfig

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings


def build_vector_store_config(settings: Mem0Settings) -> VectorStoreConfig:
    """Build Mem0's vector store config from app infrastructure settings."""
    config: dict[str, object] = {
        "collection_name": settings.collection_name,
        "embedding_model_dims": 768,
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
