"""Record the effective context sent to the model, through the trace sink."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage

from application.observability import (
    ModelContextTrace,
    ModelContextWriteRequest,
    ObservabilityPort,
)
from infrastructure.settings import LoggingConfig


logger = logging.getLogger(__name__)


class ContextLoggingMiddleware(AgentMiddleware):
    """Record each effective model request and what came back from it."""

    def __init__(
        self,
        *,
        mode: str,
        observability: ObservabilityPort | None = None,
    ) -> None:
        self._mode = mode
        self._observability = observability
        self._model_call = 0

    @classmethod
    def from_config(
        cls,
        config: LoggingConfig,
        *,
        observability: ObservabilityPort | None = None,
    ) -> ContextLoggingMiddleware:
        """Build model-context recording from its resolved config section."""
        return cls(mode=config.context_mode, observability=observability)

    @property
    def enabled(self) -> bool:
        """Return whether model context is being recorded."""
        return self._mode != "off" and self._observability is not None

    async def awrap_model_call(self, request: ModelRequest, handler):
        """Record the model-visible request and provider-reported token usage."""
        if not self.enabled:
            return await handler(request)

        self._model_call += 1
        # Captured before the call, because that is when the request was made.
        parts = self._request_parts(request)

        try:
            response = await handler(request)
        except Exception:
            await self._record(parts, status="error", usage=None)
            raise

        await self._record(
            parts,
            status="success",
            usage=self._response_usage(response),
        )
        return response

    def _request_parts(self, request: ModelRequest) -> dict[str, Any]:
        """Collect everything knowable about the request before it is sent."""
        context = request.runtime.context if request.runtime else None
        system_message = request.system_message

        return {
            "occurred_at": datetime.now(UTC),
            "invocation_id": self._resolve_invocation_id(context),
            # A temporary turn writes no conversation, so it references none.
            "session_id": (
                None
                if getattr(context, "temporary", False)
                else getattr(context, "session_id", None)
            ),
            "user_id": getattr(context, "user_id", None),
            "model": self._model_name(request),
            "model_call": self._model_call,
            "system_message": (
                self._serialize_message(system_message)
                if system_message is not None
                else None
            ),
            "messages": tuple(
                self._serialize_message(message) for message in request.messages
            ),
            "tools": tuple(self._serialize_tool(tool) for tool in request.tools),
        }

    async def _record(
        self,
        parts: dict[str, Any],
        *,
        status: str,
        usage: dict[str, Any] | None,
    ) -> None:
        """Hand one trace to the sink; a failure never breaks the turn."""
        trace = ModelContextTrace(
            mode=self._mode,
            status=status,
            usage=usage,
            **parts,
        )
        try:
            await self._observability.record_model_context(
                ModelContextWriteRequest(trace=trace)
            )
        except Exception:
            # Losing a trace is not a reason to lose the answer.
            logger.exception("Observability write failed; continuing the turn")

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

        return self._json_safe(payload)

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
            return self._json_safe(payload)

        payload = {"name": getattr(tool, "name", type(tool).__name__)}
        if self._mode == "full":
            payload.update({
                "description": getattr(tool, "description", None),
                "args": getattr(tool, "args", None),
            })
        return self._json_safe(payload)

    @staticmethod
    def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
        """Return a payload every sink can store.

        Message content can hold provider objects that JSON has no
        representation for. The file sink used to absorb that at write time; a
        JSONB column raises instead, mid-turn, so it is flattened here, where
        the LangChain types are still understood.
        """
        return json.loads(
            json.dumps(payload, ensure_ascii=False, default=str)
        )

    @staticmethod
    def _resolve_invocation_id(context: object) -> str:
        """Return the shared id for the turn, or a fresh one without a context.

        Never ``None``: readers group records by this id, so a null would
        collapse unrelated turns into one group.
        """
        return getattr(context, "invocation_id", None) or str(uuid4())

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
