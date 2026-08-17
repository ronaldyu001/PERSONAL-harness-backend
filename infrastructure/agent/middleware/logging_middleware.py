"""Development-only logging for the effective context sent to the model."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage

from infrastructure.agent.logging.config import (
    normalize_log_mode,
    resolve_log_dir,
)
from infrastructure.agent.logging.schemas import ModelContextLogEvent
from infrastructure.settings import LoggingConfig


logger = logging.getLogger(__name__)

_CONTEXT_LOG_FILENAME = "agent-context.jsonl"
_FILE_LOCK = Lock()


class ContextLoggingMiddleware(AgentMiddleware):
    """Write each effective model request to a local JSON Lines file."""

    def __init__(
        self,
        *,
        mode: str,
        log_dir: str | Path | None = None,
    ) -> None:
        self._mode = normalize_log_mode(
            mode,
            env_name="AGENT_CONTEXT_LOGGING",
        )
        self._invocation_id = str(uuid4())
        self._model_call = 0
        self._log_path = resolve_log_dir(log_dir) / _CONTEXT_LOG_FILENAME

        if self.enabled:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: LoggingConfig) -> ContextLoggingMiddleware:
        """Build model-context logging from its resolved config section."""
        return cls(
            mode=config.context_mode,
            log_dir=config.context_dir,
        )

    @property
    def enabled(self) -> bool:
        """Return whether model context logging is active."""
        return self._mode != "off"

    @property
    def log_path(self) -> Path:
        """Return the JSON Lines destination used by this middleware."""
        return self._log_path

    async def awrap_model_call(self, request: ModelRequest, handler):
        """Log the model-visible request and provider-reported token usage."""
        if not self.enabled:
            return await handler(request)

        self._model_call += 1
        event = self._build_event(request)

        try:
            response = await handler(request)
        except Exception:
            event.status = "error"
            event.usage = None
            await self._write_event(event)
            raise

        event.status = "success"
        event.usage = self._response_usage(response)
        await self._write_event(event)
        return response

    def _build_event(self, request: ModelRequest) -> ModelContextLogEvent:
        runtime_context = request.runtime.context if request.runtime else None
        system_message = request.system_message

        return ModelContextLogEvent(
            timestamp=datetime.now(UTC).isoformat(),
            invocation_id=self._invocation_id,
            model_call=self._model_call,
            session_id=getattr(runtime_context, "session_id", None),
            model=self._model_name(request),
            mode=self._mode,
            system_message=(
                self._serialize_message(system_message)
                if system_message is not None
                else None
            ),
            messages=[
                self._serialize_message(message) for message in request.messages
            ],
            tools=[self._serialize_tool(tool) for tool in request.tools],
        )

    def _serialize_message(self, message: BaseMessage) -> dict[str, Any]:
        content = message.content
        payload: dict[str, Any] = {
            "type": message.type,
            "id": message.id,
            "name": message.name,
            "content_characters": self._content_characters(content),
        }

        if self._mode == "full":
            payload["content"] = content

        if isinstance(message, AIMessage) and message.tool_calls:
            payload["tool_calls"] = [
                self._serialize_tool_call(tool_call)
                for tool_call in message.tool_calls
            ]

        if isinstance(message, ToolMessage):
            payload.update({
                "tool_call_id": message.tool_call_id,
                "status": message.status,
                # Artifacts are application data and are not sent to the model.
                "artifact_excluded": message.artifact is not None,
            })

        return payload

    def _serialize_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "name": tool_call.get("name"),
            "id": tool_call.get("id"),
        }
        if self._mode == "full":
            payload["args"] = tool_call.get("args")
        return payload

    def _serialize_tool(self, tool: object) -> dict[str, Any]:
        if isinstance(tool, dict):
            payload = {
                "name": tool.get("name")
                or tool.get("function", {}).get("name"),
            }
            if self._mode == "full":
                payload["schema"] = tool
            return payload

        payload = {"name": getattr(tool, "name", type(tool).__name__)}
        if self._mode == "full":
            payload.update({
                "description": getattr(tool, "description", None),
                "args": getattr(tool, "args", None),
            })
        return payload

    def _append_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with _FILE_LOCK, self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{line}\n")

    async def _write_event(self, event: ModelContextLogEvent) -> None:
        try:
            await asyncio.to_thread(self._append_event, event.model_dump())
        except OSError:
            logger.exception(
                "Failed to write Maia model context to %s",
                self._log_path,
            )

    @staticmethod
    def _response_usage(response: object) -> dict[str, Any] | None:
        direct_usage = getattr(response, "usage_metadata", None)
        if direct_usage is not None:
            return dict(direct_usage)

        result = getattr(response, "result", None)
        if isinstance(result, BaseMessage):
            result = [result]
        if not isinstance(result, (list, tuple)):
            return None

        for message in reversed(result):
            usage = getattr(message, "usage_metadata", None)
            if usage is not None:
                return dict(usage)
        return None

    @staticmethod
    def _model_name(request: ModelRequest) -> str:
        model = request.model
        return str(
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or type(model).__name__
        )

    @staticmethod
    def _content_characters(content: object) -> int:
        if isinstance(content, str):
            return len(content)
        return len(json.dumps(content, ensure_ascii=False, default=str))
