"""Mem0 embedder configuration builder."""

from __future__ import annotations

from mem0.embeddings.configs import EmbedderConfig

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings


def build_embedder_config(settings: Mem0Settings) -> EmbedderConfig:
    """Build Mem0's embedder config from app infrastructure settings."""
    config: dict[str, object] = {
        "model": settings.embedder_model,
        "embedding_dims": 768,
    }

    if settings.ollama_base_url:
        config["ollama_base_url"] = settings.ollama_base_url

    return EmbedderConfig(provider="ollama", config=config)
