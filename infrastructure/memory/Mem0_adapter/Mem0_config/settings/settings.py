"""Environment-backed settings for Mem0 infrastructure config."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Mem0Settings:
    """Deployment values used to assemble Mem0 configuration."""

    qdrant_url: str | None
    qdrant_host: str | None
    qdrant_port: int
    qdrant_api_key: str | None
    collection_name: str
    ollama_base_url: str | None
    llm_model: str
    embedder_model: str
    history_db_path: str
    local_qdrant_path: str

    @classmethod
    def from_env(cls) -> Mem0Settings:
        """Build settings from environment variables."""
        mem0_dir = os.getenv("MEM0_DIR", "/tmp/mem0")

        return cls(
            qdrant_url=os.getenv("MEM0_QDRANT_URL"),
            qdrant_host=os.getenv("MEM0_QDRANT_HOST"),
            qdrant_port=int(os.getenv("MEM0_QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("MEM0_QDRANT_API_KEY"),
            collection_name=os.getenv("MEM0_COLLECTION_NAME", "harness_memories"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
            llm_model=os.getenv("MEM0_LLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b")),
            embedder_model=os.getenv("MEM0_EMBEDDER_MODEL", "nomic-embed-text"),
            history_db_path=os.getenv(
                "MEM0_HISTORY_DB_PATH", os.path.join(mem0_dir, "history.db")
            ),
            local_qdrant_path=os.getenv(
                "MEM0_QDRANT_PATH", os.path.join(mem0_dir, "qdrant")
            ),
        )
