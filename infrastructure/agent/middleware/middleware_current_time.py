"""Inject authoritative invocation time into model requests transiently."""

from __future__ import annotations

import pytz
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.messages import SystemMessage

from infrastructure.agent.context import ContextRuntime
from infrastructure.settings import TimeContextConfig


class MiddlewareCurrentTime(AgentMiddleware):
    """Expose local invocation time without adding it to conversation state."""

    def __init__(self, *, timezone: str) -> None:
        self._timezone = pytz.timezone(timezone)
        self._timezone_name = timezone

    @classmethod
    def from_config(cls, config: TimeContextConfig) -> MiddlewareCurrentTime:
        """Build time middleware from its resolved configuration section."""
        return cls(timezone=config.timezone)

    async def awrap_model_call(self, request: ModelRequest, handler):
        """Add one authoritative clock line only to the effective model request."""
        runtime = request.runtime
        if runtime is None or not isinstance(runtime.context, ContextRuntime):
            return await handler(request)

        local_time = runtime.context.invocation_time_utc.astimezone(self._timezone)
        time_line = (
            f"Current time: {local_time.isoformat(timespec='seconds')} "
            f"({self._timezone_name})."
        )
        original_system = request.system_message
        system_content = (
            f"{original_system.text}\n\n{time_line}"
            if original_system is not None
            else time_line
        )
        return await handler(request.override(
            system_message=SystemMessage(content=system_content)
        ))
