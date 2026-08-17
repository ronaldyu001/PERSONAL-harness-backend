"""Environment-backed settings for Mem0 infrastructure."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_CUSTOM_INSTRUCTIONS = """
Extract durable information that will be useful in future conversations.

Always retain:
- Explicit user requests to remember information.
- Stable user preferences and constraints.
- Durable personal facts stated by the user.
- Ongoing projects, goals, and commitments.

Exclude:
- Temporary requests or one-off conversational details.
- Assistant assumptions or unsupported conclusions.
- Speculation presented as fact.
- Sensitive information unless the user explicitly asks for it to be remembered.
- Tool output that the user has not confirmed.

Return no memories when the turn contains nothing durable.
""".strip()


@dataclass(frozen=True, slots=True)
class Mem0Settings:
    """Deployment values used to assemble Mem0's SDK configuration."""

    qdrant_url: str | None
    qdrant_host: str | None
    qdrant_port: int
    qdrant_api_key: str | None
    collection_name: str
    ollama_base_url: str | None
    litellm_base_url: str | None
    litellm_api_key: str
    llm_model: str
    embedder_model: str
    embedding_dims: int
    history_db_path: str
    local_qdrant_path: str
    custom_instructions: str

    @classmethod
    def from_env(cls) -> Mem0Settings:
        """Build settings from environment variables."""
        mem0_dir = os.getenv("MEM0_DIR", "/tmp/mem0")

        return cls(
            qdrant_url=os.getenv("MEM0_QDRANT_URL"),
            qdrant_host=os.getenv("MEM0_QDRANT_HOST"),
            qdrant_port=int(os.getenv("MEM0_QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("MEM0_QDRANT_API_KEY"),
            collection_name=os.getenv(
                "MEM0_COLLECTION_NAME", "harness_memories"
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
            litellm_base_url=os.getenv("LITELLM_BASE_URL"),
            litellm_api_key=os.getenv("LITELLM_API_KEY", "EMPTY"),
            llm_model=os.getenv(
                "MEM0_LLM_MODEL", "qwen"
            ),
            embedder_model=os.getenv(
                "MEM0_EMBEDDER_MODEL", "nomic-embed-text"
            ),
            embedding_dims=int(os.getenv("MEM0_EMBEDDING_DIMS") or "768"),
            history_db_path=os.getenv(
                "MEM0_HISTORY_DB_PATH", os.path.join(mem0_dir, "history.db")
            ),
            local_qdrant_path=os.getenv(
                "MEM0_QDRANT_PATH", os.path.join(mem0_dir, "qdrant")
            ),
            custom_instructions=(
                os.getenv("MEM0_CUSTOM_INSTRUCTIONS")
                or DEFAULT_CUSTOM_INSTRUCTIONS
            ),
        )
