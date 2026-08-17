"""Load Maia's complete, validated infrastructure settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class _Config(BaseModel):
    """Shared validation policy for every configuration section."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GatewayConfig(_Config):
    base_url: str = Field(min_length=1)
    api_key: str
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0)


class SummarizationConfig(_Config):
    trigger_tokens: int = Field(gt=0)
    keep_messages: int = Field(gt=0)


class AgentMemoryConfig(_Config):
    retrieval_limit: int = Field(gt=0)


class ResponseGateConfig(_Config):
    enabled: bool
    max_repairs: int = Field(ge=0)
    evaluator_max_tokens: int = Field(gt=0)
    evaluator_prompt: str = Field(min_length=1)
    repair_prompt: str = Field(min_length=1)
    fallback_response: str = Field(min_length=1)


class AgentConfig(_Config):
    system_prompt: str = Field(min_length=1)
    summarization: SummarizationConfig
    memory: AgentMemoryConfig
    response_gate: ResponseGateConfig


LogMode = Literal["off", "structure", "full"]


class LoggingConfig(_Config):
    context_mode: LogMode
    response_gate_mode: LogMode
    context_dir: Path | None
    response_gate_dir: Path | None


class Mem0LlmConfig(_Config):
    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key: str
    temperature: float = Field(ge=0)
    max_tokens: int = Field(gt=0)
    top_p: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)


class Mem0EmbedderConfig(_Config):
    model: str = Field(min_length=1)
    base_url: str | None
    dimensions: int = Field(gt=0)


class Mem0VectorStoreConfig(_Config):
    on_disk: bool
    url: str | None
    host: str | None
    port: int = Field(gt=0, le=65535)
    api_key: str | None
    local_path: Path


class Mem0Config(_Config):
    data_dir: Path
    history_db_path: Path
    collection_name: str = Field(min_length=1)
    llm: Mem0LlmConfig
    embedder: Mem0EmbedderConfig
    vector_store: Mem0VectorStoreConfig
    custom_instructions: str = Field(min_length=1)


class ApiConfig(_Config):
    cors_allow_origins: tuple[str, ...]


class InfrastructureSettings(_Config):
    """Single resolved configuration object shared at the composition root."""

    gateway: GatewayConfig
    agent: AgentConfig
    logging: LoggingConfig
    mem0: Mem0Config
    api: ApiConfig


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_infrastructure_settings(
    *,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path | None = None,
) -> InfrastructureSettings:
    """Merge YAML with deployment values and validate the resulting object."""
    if environ is None:
        load_dotenv()
        environ = os.environ

    resolved_path = Path(
        config_path
        or environ.get("HARNESS_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    ).expanduser().resolve()
    payload = _read_yaml(resolved_path)

    gateway = _mapping(payload, "gateway")
    gateway["base_url"] = _required(environ, "LITELLM_BASE_URL")
    gateway["api_key"] = environ.get("LITELLM_API_KEY", "EMPTY")

    logging = _mapping(payload, "logging")
    context_mode = _log_mode(
        environ.get("AGENT_CONTEXT_LOGGING", logging.get("context_mode", "off")),
        name="AGENT_CONTEXT_LOGGING",
    )
    response_value = environ.get(
        "AGENT_RESPONSE_GATE_LOGGING",
        str(logging.get("response_gate_mode", "inherit")),
    ).strip().lower()
    response_mode = (
        context_mode
        if response_value == "inherit"
        else _log_mode(response_value, name="AGENT_RESPONSE_GATE_LOGGING")
    )
    context_dir = _optional_path(environ, "AGENT_CONTEXT_LOG_DIR")
    logging.update({
        "context_mode": context_mode,
        "response_gate_mode": response_mode,
        "context_dir": context_dir,
        "response_gate_dir": (
            _optional_path(environ, "AGENT_RESPONSE_GATE_LOG_DIR")
            or context_dir
        ),
    })

    mem0_dir = Path(environ.get("MEM0_DIR", "/tmp/mem0")).expanduser()
    mem0 = _mapping(payload, "mem0")
    mem0.update({
        "data_dir": mem0_dir,
        "history_db_path": mem0_dir / "history.db",
    })
    mem0_llm = _mapping(mem0, "llm")
    mem0_llm.update({
        "base_url": gateway["base_url"],
        "api_key": gateway["api_key"],
    })
    mem0_embedder = _mapping(mem0, "embedder")
    mem0_embedder.update({
        "model": environ.get(
            "MEM0_EMBEDDER_MODEL",
            str(mem0_embedder.get("model", "")),
        ),
        "base_url": _optional(environ, "OLLAMA_BASE_URL"),
    })
    vector_store = _mapping(mem0, "vector_store")
    vector_store.update({
        "url": _optional(environ, "MEM0_QDRANT_URL"),
        "host": _optional(environ, "MEM0_QDRANT_HOST"),
        "port": int(environ.get("MEM0_QDRANT_PORT", "6333")),
        "api_key": _optional(environ, "MEM0_QDRANT_API_KEY"),
        "local_path": mem0_dir / "qdrant",
    })

    payload["api"] = {
        "cors_allow_origins": tuple(
            origin.strip()
            for origin in environ.get("CORS_ALLOW_ORIGINS", "").split(",")
            if origin.strip()
        )
    }
    return InfrastructureSettings.model_validate(payload)


def _read_yaml(path: Path) -> dict[str, object]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Infrastructure config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in infrastructure config: {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Infrastructure config must be a YAML mapping: {path}")
    return payload


def _mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Infrastructure config section must be a mapping: {key}")
    return value


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set.")
    return value


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def _optional_path(environ: Mapping[str, str], name: str) -> Path | None:
    value = _optional(environ, name)
    return Path(value).expanduser() if value else None


def _log_mode(value: object, *, name: str) -> LogMode:
    normalized = str(value).strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return "off"
    if normalized in {"1", "true", "yes", "on", "full"}:
        return "full"
    if normalized == "structure":
        return "structure"
    raise ValueError(f"{name} must be one of: off, structure, full")
