"""Dedicated JSONL writer for response-gate decisions."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from infrastructure.agent.logging.config import (
    normalize_log_mode,
    resolve_log_dir,
)
from infrastructure.agent.logging.schemas import ResponseGateLogEvent


logger = logging.getLogger(__name__)

_LOG_FILENAME = "response-gate.jsonl"
_FILE_LOCK = Lock()


class ResponseGateLogWriter:
    """Write response-gate decisions to a dedicated JSON Lines file."""

    def __init__(
        self,
        *,
        mode: str = "off",
        log_dir: str | Path | None = None,
    ) -> None:
        self._mode = normalize_log_mode(
            mode,
            env_name="AGENT_RESPONSE_GATE_LOGGING",
        )
        self._invocation_id = str(uuid4())
        self._log_path = resolve_log_dir(log_dir) / _LOG_FILENAME

        if self.enabled:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> ResponseGateLogWriter:
        """Build gate logging, inheriting context-log settings by default."""
        return cls(
            mode=os.getenv(
                "AGENT_RESPONSE_GATE_LOGGING",
                os.getenv("AGENT_CONTEXT_LOGGING", "off"),
            ),
            log_dir=(
                os.getenv("AGENT_RESPONSE_GATE_LOG_DIR")
                or os.getenv("AGENT_CONTEXT_LOG_DIR")
            ),
        )

    @property
    def enabled(self) -> bool:
        """Return whether response-gate logging is active."""
        return self._mode != "off"

    @property
    def log_path(self) -> Path:
        """Return the dedicated response-gate JSONL destination."""
        return self._log_path

    async def log_evaluation(
        self,
        *,
        session_id: str | None,
        model: str,
        evaluation_call: int,
        repair_attempt: int,
        decision: Literal["allow", "retry", "fallback", "allow_on_error"],
        passed: bool | None,
        violations: list[str],
        feedback: str | None,
        candidate_message_id: str | None,
        candidate: str,
        available_tools: list[str],
        tools_used: list[str],
        usage: dict[str, Any] | None,
        error: Exception | None = None,
    ) -> None:
        """Append one schema-validated gate event when logging is enabled."""
        if not self.enabled:
            return

        event = ResponseGateLogEvent(
            timestamp=datetime.now(UTC).isoformat(),
            invocation_id=self._invocation_id,
            session_id=session_id,
            model=model,
            mode=self._mode,
            evaluation_call=evaluation_call,
            repair_attempt=repair_attempt,
            decision=decision,
            passed=passed,
            violations=violations,
            feedback=feedback if self._mode == "full" else None,
            candidate_message_id=candidate_message_id,
            candidate_characters=len(candidate),
            candidate=candidate if self._mode == "full" else None,
            available_tools=available_tools,
            tools_used=tools_used,
            usage=usage,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )
        try:
            await asyncio.to_thread(self._append_event, event)
        except OSError:
            logger.exception(
                "Failed to write Maia response gate to %s",
                self._log_path,
            )

    def _append_event(self, event: ResponseGateLogEvent) -> None:
        line = event.model_dump_json()
        with _FILE_LOCK, self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"{line}\n")
