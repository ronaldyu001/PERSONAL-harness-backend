"""Top-level Mem0 MemoryConfig builder."""

from __future__ import annotations

from mem0.memory.main import MemoryConfig

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings
from infrastructure.memory.Mem0_adapter.Mem0_config.settings.database import build_history_db_path
from infrastructure.memory.Mem0_adapter.Mem0_config.settings.embedder import build_embedder_config
from infrastructure.memory.Mem0_adapter.Mem0_config.settings.llm import build_llm_config
from infrastructure.memory.Mem0_adapter.Mem0_config.settings.reranker import build_reranker_config
from infrastructure.memory.Mem0_adapter.Mem0_config.settings.vector_store import build_vector_store_config


def build_memory_config(settings: Mem0Settings | None = None) -> MemoryConfig:
    """Build the Mem0 configuration used by the memory adapter."""
    settings = settings or Mem0Settings.from_env()

    return MemoryConfig(
        vector_store=build_vector_store_config(settings),
        llm=build_llm_config(settings),
        embedder=build_embedder_config(settings),
        history_db_path=build_history_db_path(settings),
        reranker=build_reranker_config(settings),
    )
