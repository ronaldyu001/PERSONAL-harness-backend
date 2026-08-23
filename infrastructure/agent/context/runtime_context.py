"""Typed runtime context passed to LangChain agent middleware."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone as datetime_timezone

import pytz


def _utc_now() -> datetime:
    """Return the canonical instant used for a new agent invocation."""
    return datetime.now(datetime_timezone.utc)


@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """Immutable identity and clock context for one agent invocation."""

    user_id: str
    session_id: str
    invocation_time_utc: datetime = field(default_factory=_utc_now)
    timezone: str = "UTC"
    # The model that answered this turn, recorded alongside the transcript.
    model: str | None = None
    # A temporary turn is answered normally but leaves nothing behind.
    temporary: bool = False

    def __post_init__(self) -> None:
        """Normalize the canonical instant and validate its display timezone."""
        if (
            self.invocation_time_utc.tzinfo is None
            or self.invocation_time_utc.utcoffset() is None
        ):
            raise ValueError("invocation_time_utc must be timezone-aware")
        object.__setattr__(
            self,
            "invocation_time_utc",
            self.invocation_time_utc.astimezone(datetime_timezone.utc),
        )
        try:
            pytz.timezone(self.timezone)
        except pytz.UnknownTimeZoneError as exc:
            raise ValueError(f"Unknown IANA timezone: {self.timezone}") from exc

    @property
    def current_time_iso(self) -> str:
        """Return invocation time in the configured local timezone."""
        local_time = self.invocation_time_utc.astimezone(
            pytz.timezone(self.timezone)
        )
        return local_time.isoformat(timespec="seconds")
