"""Validate Maia's final draft and request one repair when needed."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, ModelRequest, hook_config
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph.message import RemoveMessage
from pydantic import BaseModel, Field, field_validator

from application.observability import (
    ObservabilityPort,
    ResponseGateTrace,
    ResponseGateWriteRequest,
)
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.agent.middleware.helpers import USER_MEMORIES_MESSAGE_NAME
from infrastructure.settings import ResponseGateConfig

logger = logging.getLogger(__name__)


class ToolTrace(BaseModel):
    """One completed tool call and its already-budgeted model evidence."""

    tool_call_id: str
    name: str
    evidence: str
    # How many user turns back the lookup happened; 0 is the turn being
    # answered now. The evaluator needs this to tell a claim restated from
    # earlier evidence apart from one citing something stale.
    turns_ago: int = 0


class ResponseEvaluation(BaseModel):
    """Structured decision returned by the response-policy evaluator."""

    passed: bool = Field(
        description="True only when the candidate has no concrete violation."
    )
    violations: list[str] = Field(
        default_factory=list,
        description="Short descriptions of concrete policy violations.",
    )
    feedback: str = Field(
        default="",
        description="A concise instruction for repairing a failed candidate.",
    )

    @field_validator("violations", mode="before")
    @classmethod
    def normalize_null_violations(cls, value: object) -> object:
        """Treat a local model's null violations as an empty list."""
        return [] if value is None else value

    @field_validator("feedback", mode="before")
    @classmethod
    def normalize_null_feedback(cls, value: object) -> object:
        """Treat a local model's null pass feedback as an empty string."""
        return "" if value is None else value


class ModelResponseGateMiddleware(AgentMiddleware):
    """Gate final natural-language responses and retry the model once."""

    def __init__(
        self,
        *,
        model: BaseChatModel,
        system_prompt: str,
        max_repair_attempts: int,
        fallback_response: str,
        evaluator_prompt: str,
        repair_prompt: str,
        evaluator_max_tokens: int,
        evidence_turns: int,
        prior_evidence_characters: int,
        observability: ObservabilityPort | None = None,
        mode: str = "off",
        evaluator: Any | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        if not fallback_response.strip():
            raise ValueError("fallback_response cannot be empty")
        if evaluator_max_tokens <= 0:
            raise ValueError("evaluator_max_tokens must be positive")
        if evidence_turns <= 0:
            raise ValueError("evidence_turns must be positive")
        if prior_evidence_characters <= 0:
            raise ValueError("prior_evidence_characters must be positive")

        self._model_name = self._resolve_model_name(model)
        self._system_prompt = system_prompt
        self._observability = observability
        self._mode = mode
        self._max_repair_attempts = max_repair_attempts
        self._fallback_response = fallback_response.strip()
        self._evaluator_prompt = evaluator_prompt.strip()
        self._repair_prompt = repair_prompt.strip()
        self._evidence_turns = evidence_turns
        self._prior_evidence_characters = prior_evidence_characters
        # LiteLLM's Ollama path reliably honors JSON-object mode, while its
        # JSON-schema mode can still return a correct verdict as Markdown.
        self._evaluator = evaluator or model.bind(
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=evaluator_max_tokens,
        )
        self._repair_attempts = 0
        self._evaluation_calls = 0
        # Only the verdict is carried forward. Handing the rejected draft back
        # to the model reproduces it rather than avoiding it.
        self._pending_repair: ResponseEvaluation | None = None
        self._available_tools: tuple[str, ...] = ()
        # Both are read off the request in flight. Memories never enter agent
        # state, and the effective system prompt is assembled by the middleware
        # ahead of this one, so neither is recoverable once the call returns.
        self._recent_memories: tuple[str, ...] = ()
        self._effective_system_prompt: str | None = None

    @classmethod
    def from_config(
        cls,
        config: ResponseGateConfig,
        *,
        model: BaseChatModel,
        system_prompt: str,
        observability: ObservabilityPort | None = None,
        mode: str = "off",
        evaluator: Any | None = None,
    ) -> ModelResponseGateMiddleware:
        """Build the response gate from its resolved config section."""
        return cls(
            model=model,
            system_prompt=system_prompt,
            max_repair_attempts=config.max_repairs,
            fallback_response=config.fallback_response,
            evaluator_prompt=config.evaluator_prompt,
            repair_prompt=config.repair_prompt,
            evaluator_max_tokens=config.evaluator_max_tokens,
            evidence_turns=config.evidence_turns,
            prior_evidence_characters=config.prior_evidence_characters,
            observability=observability,
            mode=mode,
            evaluator=evaluator,
        )

    async def awrap_model_call(self, request: ModelRequest, handler):
        """Capture capabilities and inject repair feedback for one model call."""
        self._available_tools = tuple(
            name
            for tool in request.tools
            if (name := self._tool_name(tool)) is not None
        )
        # Read from the incoming request, before the repair override below, so
        # the evaluator judges the context Maia was actually given. Repair
        # instructions are transient scaffolding and are never part of that.
        #
        # Every one of these is reassigned on every call, empty included.
        # MemoryMiddleware skips injection both when retrieval returns nothing
        # and when it fails, so writing only when a block is present would
        # judge a repair pass against the previous call's memories.
        self._recent_memories = tuple(
            message.text
            for message in request.messages
            if isinstance(message, SystemMessage)
            and message.name == USER_MEMORIES_MESSAGE_NAME
        )
        self._effective_system_prompt = (
            request.system_message.text
            if request.system_message is not None
            else None
        )

        pending_repair = self._pending_repair
        if pending_repair is None:
            return await handler(request)

        repair_message = self._repair_prompt.format(
            feedback=self._repair_feedback(pending_repair),
        )
        original_system = request.system_message
        system_content = (
            f"{original_system.text}\n\n{repair_message}"
            if original_system is not None
            else repair_message
        )

        try:
            return await handler(
                request.override(
                    system_message=SystemMessage(content=system_content)
                )
            )
        finally:
            # Repair instructions are transient and must never enter checkpoints.
            self._pending_repair = None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        """Evaluate the latest final draft and either allow, retry, or replace it."""
        candidate = self._latest_ai_message(state.get("messages", ()))
        if candidate is None or candidate.tool_calls:
            return None

        self._evaluation_calls += 1
        context = getattr(runtime, "context", None)
        # A temporary turn writes no conversation, so it references none.
        temporary = bool(getattr(context, "temporary", False))
        session_id = None if temporary else getattr(context, "session_id", None)
        user_id = getattr(context, "user_id", None)
        invocation_id = self._resolve_invocation_id(runtime)
        # One window, two views of it. Deriving both from the same boundary is
        # what stops a claim and the evidence for it landing on opposite sides.
        window = self._evidence_window(
            state.get("messages", ()),
            turns=self._evidence_turns,
        )
        tool_traces = self._tool_traces(
            window,
            prior_characters=self._prior_evidence_characters,
        )
        tools_used = tuple(trace.name for trace in tool_traces)
        time_context = self._time_context(getattr(runtime, "context", None))

        try:
            evaluation, usage = await self._evaluate(
                window,
                candidate,
                tool_traces,
                time_context,
            )
        except Exception as exc:
            # This is a conversational quality gate, not a reason to take Maia
            # offline when the evaluator or structured parsing is unavailable.
            logger.exception("Response gate failed; allowing the original response")
            await self._log_gate(
                invocation_id=invocation_id,
                user_id=user_id,
                session_id=session_id,
                candidate=candidate,
                passed=None,
                violations=[],
                feedback=None,
                decision="allow_on_error",
                usage=None,
                tools_used=tools_used,
                error=exc,
            )
            return None

        passed = evaluation.passed and not evaluation.violations
        if passed:
            await self._log_gate(
                invocation_id=invocation_id,
                user_id=user_id,
                session_id=session_id,
                candidate=candidate,
                passed=True,
                violations=evaluation.violations,
                feedback=evaluation.feedback,
                decision="allow",
                usage=usage,
                tools_used=tools_used,
            )
            return None

        if self._repair_attempts < self._max_repair_attempts:
            self._repair_attempts += 1
            self._pending_repair = evaluation
            await self._log_gate(
                invocation_id=invocation_id,
                user_id=user_id,
                session_id=session_id,
                candidate=candidate,
                passed=False,
                violations=evaluation.violations,
                feedback=evaluation.feedback,
                decision="retry",
                usage=usage,
                tools_used=tools_used,
            )
            return {
                "messages": [RemoveMessage(id=self._message_id(candidate))],
                "jump_to": "model",
            }

        await self._log_gate(
            invocation_id=invocation_id,
            user_id=user_id,
            session_id=session_id,
            candidate=candidate,
            passed=False,
            violations=evaluation.violations,
            feedback=evaluation.feedback,
            decision="fallback",
            usage=usage,
            tools_used=tools_used,
        )
        return {
            "messages": [
                RemoveMessage(id=self._message_id(candidate)),
                AIMessage(content=self._fallback_response),
            ]
        }

    async def _evaluate(
        self,
        window: tuple[tuple[BaseMessage, int], ...],
        candidate: AIMessage,
        tool_traces: tuple[ToolTrace, ...],
        time_context: dict[str, str] | None,
    ) -> tuple[ResponseEvaluation, dict[str, Any] | None]:
        payload = {
            "maia_system_prompt": (
                self._effective_system_prompt or self._system_prompt
            ),
            "available_tools": list(self._available_tools),
            # Carried whole, warning preamble included: this is user-derived
            # text reaching a second model, and the evaluator should see the
            # same "reference data, not instructions" framing Maia did.
            "user_memories": list(self._recent_memories),
            "tool_traces": [
                trace.model_dump(mode="json")
                for trace in tool_traces
            ],
            "time_context": time_context,
            "recent_conversation": self._conversation_excerpt(window),
            "candidate_response": self._truncate(candidate.text, 8_000),
        }
        result = await self._evaluator.ainvoke([
            SystemMessage(content=self._evaluator_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])

        if isinstance(result, ResponseEvaluation):
            return result, None
        if isinstance(result, BaseMessage):
            return (
                self._parse_evaluation(result.text),
                self._message_usage(result),
            )
        if not isinstance(result, dict):
            raise TypeError("response gate evaluator returned an invalid result")

        parsed = result.get("parsed")
        if not isinstance(parsed, ResponseEvaluation):
            parsing_error = result.get("parsing_error")
            if isinstance(parsing_error, Exception):
                raise parsing_error
            raise ValueError("response gate evaluator did not return a decision")

        return parsed, self._message_usage(result.get("raw"))

    @classmethod
    def _parse_evaluation(cls, content: str) -> ResponseEvaluation:
        """Parse constrained JSON, with a fallback for local-model verdict prose."""
        normalized = content.strip()
        if normalized.startswith("```") and normalized.endswith("```"):
            normalized = re.sub(r"^```(?:json)?\s*|\s*```$", "", normalized)

        candidates = [normalized]
        object_start = normalized.find("{")
        object_end = normalized.rfind("}")
        if object_start >= 0 and object_end > object_start:
            candidates.append(normalized[object_start : object_end + 1])
        elif object_start >= 0:
            completed = cls._complete_truncated_json(normalized[object_start:])
            if completed is not None:
                candidates.append(completed)

        parsing_error: Exception | None = None
        for candidate in candidates:
            try:
                return ResponseEvaluation.model_validate_json(candidate)
            except Exception as exc:
                parsing_error = exc

        prose = re.sub(r"[*_#`]", "", normalized)
        fail_match = re.search(
            r"(?im)^\s*(?:evaluation|verdict)\s*:\s*"
            r"(?:fail(?:ed)?|reject(?:ed)?)\b",
            prose,
        )
        if fail_match:
            feedback = prose[fail_match.end() :].strip(" \n:-")
            return ResponseEvaluation(
                passed=False,
                violations=["The response evaluator rejected the candidate."],
                feedback=cls._truncate(feedback, 1_000),
            )

        pass_match = re.search(
            r"(?im)^\s*(?:evaluation|verdict)\s*:\s*pass(?:ed)?\b",
            prose,
        )
        if pass_match:
            return ResponseEvaluation(passed=True, violations=[], feedback="")

        # A truncated constrained response can still contain an unambiguous
        # boolean verdict. Preserve a failing verdict instead of failing open.
        json_verdict = re.search(
            r'"passed"\s*:\s*(true|false)\b',
            normalized,
            flags=re.IGNORECASE,
        )
        if json_verdict and json_verdict.group(1).lower() == "false":
            return ResponseEvaluation(
                passed=False,
                violations=[
                    "The evaluator rejected the response but its JSON was truncated."
                ],
                feedback="Rewrite the response conservatively using only supported facts.",
            )
        if json_verdict and json_verdict.group(1).lower() == "true":
            return ResponseEvaluation(passed=True, violations=[], feedback="")

        if parsing_error is not None:
            raise parsing_error
        raise ValueError("response gate evaluator returned an empty decision")

    @staticmethod
    def _complete_truncated_json(content: str) -> str | None:
        """Close an otherwise valid JSON prefix cut off at the token limit."""
        stack: list[str] = []
        in_string = False
        escaped = False

        for character in content:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character in "{[":
                stack.append(character)
            elif character == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif character == "]" and stack and stack[-1] == "[":
                stack.pop()

        if not stack:
            return None

        suffix = '"' if in_string and not escaped else ""
        suffix += "".join("}" if item == "{" else "]" for item in reversed(stack))
        return f"{content}{suffix}"

    @staticmethod
    def _resolve_invocation_id(runtime: object) -> str:
        """Return the turn's shared id, or a fresh one without a context.

        Never ``None``: readers group records by this id, so a null would
        collapse unrelated turns into one group.
        """
        context = getattr(runtime, "context", None)
        return getattr(context, "invocation_id", None) or str(uuid4())

    async def _log_gate(
        self,
        *,
        invocation_id: str,
        user_id: str | None,
        session_id: str | None,
        candidate: AIMessage,
        passed: bool | None,
        violations: list[str],
        feedback: str | None,
        decision: str,
        usage: dict[str, Any] | None,
        tools_used: tuple[str, ...],
        error: Exception | None = None,
    ) -> None:
        """Record one gate decision; a sink failure never breaks the turn."""
        if self._observability is None or self._mode == "off":
            return

        text = candidate.text
        trace = ResponseGateTrace(
            invocation_id=invocation_id,
            occurred_at=datetime.now(UTC),
            model=self._model_name,
            mode=self._mode,
            evaluation_call=self._evaluation_calls,
            repair_attempt=self._repair_attempts,
            decision=decision,
            candidate_characters=len(text),
            passed=passed,
            session_id=session_id,
            user_id=user_id,
            violations=tuple(violations),
            # Structure mode keeps the decision and drops the text.
            feedback=feedback if self._mode == "full" else None,
            candidate_message_id=candidate.id,
            candidate=text if self._mode == "full" else None,
            available_tools=tuple(self._available_tools),
            tools_used=tuple(tools_used),
            usage=usage,
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

        try:
            await self._observability.record_response_gate(
                ResponseGateWriteRequest(trace=trace)
            )
        except Exception:
            # Losing a trace is not a reason to lose the answer.
            logger.exception("Observability write failed; continuing the turn")

    @staticmethod
    def _latest_ai_message(messages: object) -> AIMessage | None:
        if not isinstance(messages, (list, tuple)):
            return None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    @staticmethod
    def _evidence_window(
        messages: object,
        *,
        turns: int,
    ) -> tuple[tuple[BaseMessage, int], ...]:
        """Return recent messages paired with how many turns back they sit.

        A turn opens at a ``HumanMessage``, so everything after the most recent
        one belongs to the turn being answered now and carries 0. This is the
        only boundary the evaluator payload is built from: the conversation
        excerpt and the tool traces are two views of this same slice, which is
        what keeps an answer visible while the evidence behind it is not.
        """
        if not isinstance(messages, (list, tuple)):
            return ()

        window: list[tuple[BaseMessage, int]] = []
        turns_ago = 0
        for message in reversed(messages):
            if not isinstance(message, BaseMessage):
                continue
            if turns_ago >= turns:
                break
            window.append((message, turns_ago))
            if isinstance(message, HumanMessage):
                turns_ago += 1
        window.reverse()
        return tuple(window)

    @classmethod
    def _conversation_excerpt(
        cls,
        window: tuple[tuple[BaseMessage, int], ...],
    ) -> list[dict[str, str]]:
        """Return the window's non-tool messages, minus the candidate.

        Tool activity is left out because ``tool_traces`` carries it in full
        with its evidence attached. The window is the only bound on length; a
        separate message cap here is how this view and the traces drifted onto
        different spans of history in the first place.
        """
        excerpt: list[dict[str, str]] = []
        for message, _ in window[:-1]:
            if isinstance(message, ToolMessage):
                continue
            if isinstance(message, AIMessage) and message.tool_calls:
                continue
            excerpt.append({
                "role": message.type,
                "content": cls._truncate(message.text, 1_500),
            })
        return excerpt

    @classmethod
    def _tool_traces(
        cls,
        window: tuple[tuple[BaseMessage, int], ...],
        *,
        prior_characters: int,
    ) -> tuple[ToolTrace, ...]:
        """Pair tool calls in the window with their ToolMessage evidence."""
        calls_by_id: dict[str, str] = {}
        traces: list[ToolTrace] = []
        for message, turns_ago in window:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    call_id = tool_call.get("id")
                    name = tool_call.get("name")
                    if call_id and name:
                        calls_by_id[str(call_id)] = str(name)
                continue

            if not isinstance(message, ToolMessage):
                continue
            call_id = str(message.tool_call_id)
            name = calls_by_id.get(call_id) or message.name
            if not name:
                continue
            traces.append(ToolTrace(
                tool_call_id=call_id,
                name=str(name),
                # The current turn's evidence is what the answer was written
                # from, so it travels whole. Earlier turns are supporting
                # context and are budgeted instead.
                evidence=(
                    message.text
                    if turns_ago == 0
                    else cls._truncate_middle(message.text, prior_characters)
                ),
                turns_ago=turns_ago,
            ))
        return tuple(traces)

    @staticmethod
    def _tool_name(tool: object) -> str | None:
        if isinstance(tool, dict):
            value = tool.get("name") or tool.get("function", {}).get("name")
        else:
            value = getattr(tool, "name", None)
        return str(value) if value else None

    @staticmethod
    def _time_context(context: object) -> dict[str, str] | None:
        """Serialize the invocation clock context for the evaluator."""
        if not isinstance(context, AgentRuntimeContext):
            return None
        return {
            "current_time": context.current_time_iso,
            "timezone": context.timezone,
        }

    @staticmethod
    def _message_usage(message: object) -> dict[str, Any] | None:
        usage = getattr(message, "usage_metadata", None)
        if usage is not None:
            return dict(usage)
        metadata = getattr(message, "response_metadata", None)
        if isinstance(metadata, dict):
            token_usage = metadata.get("token_usage")
            if isinstance(token_usage, dict):
                return dict(token_usage)
        return None

    @staticmethod
    def _message_id(message: AIMessage) -> str:
        if not message.id:
            raise RuntimeError("response gate candidate did not have a message ID")
        return message.id

    @staticmethod
    def _repair_feedback(evaluation: ResponseEvaluation) -> str:
        parts = [
            *(
                f"- {ModelResponseGateMiddleware._truncate(item, 500)}"
                for item in evaluation.violations[:5]
            ),
            ModelResponseGateMiddleware._truncate(
                evaluation.feedback.strip(), 1_000
            ),
        ]
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _resolve_model_name(model: BaseChatModel) -> str:
        return str(
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or type(model).__name__
        )

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[:limit]}..."

    @staticmethod
    def _truncate_middle(value: str, limit: int) -> str:
        """Drop the middle of an over-long value, keeping both ends.

        Search results routinely settle the question in their closing lines, so
        cutting only the tail would remove the part that grounds the answer.
        """
        if len(value) <= limit:
            return value

        marker = "\n...\n"
        keep = limit - len(marker)
        if keep <= 0:
            return value[:limit]
        head = keep // 2
        return f"{value[:head]}{marker}{value[-(keep - head):]}"
