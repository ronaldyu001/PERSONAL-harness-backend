"""Mem0 LLM configuration builder."""

from __future__ import annotations

from mem0.llms.configs import LlmConfig

from infrastructure.memory.Mem0_adapter.Mem0_config.settings import Mem0Settings


def build_llm_config(settings: Mem0Settings) -> LlmConfig:
    """Build Mem0's LLM config from app infrastructure settings."""
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
