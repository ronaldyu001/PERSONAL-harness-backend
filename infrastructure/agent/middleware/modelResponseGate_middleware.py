"""Validate Maia's final draft and request one repair when needed."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, hook_config
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph.message import RemoveMessage
from pydantic import BaseModel, Field, field_validator

from infrastructure.agent.logging import ResponseGateLogWriter
from infrastructure.settings import ResponseGateConfig

logger = logging.getLogger(__name__)


class ToolTrace(BaseModel):
    """One completed tool call and its already-budgeted model evidence."""

    tool_call_id: str
    name: str
    evidence: str


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
        log_writer: ResponseGateLogWriter | None = None,
        evaluator: Any | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts cannot be negative")
        if not fallback_response.strip():
            raise ValueError("fallback_response cannot be empty")
        if evaluator_max_tokens <= 0:
            raise ValueError("evaluator_max_tokens must be positive")

        self._model_name = self._resolve_model_name(model)
        self._system_prompt = system_prompt
        self._log_writer = log_writer
        self._max_repair_attempts = max_repair_attempts
        self._fallback_response = fallback_response.strip()
        self._evaluator_prompt = evaluator_prompt.strip()
        self._repair_prompt = repair_prompt.strip()
        # LiteLLM's Ollama path reliably honors JSON-object mode, while its
        # JSON-schema mode can still return a correct verdict as Markdown.
        self._evaluator = evaluator or model.bind(
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=evaluator_max_tokens,
        )
        self._repair_attempts = 0
        self._evaluation_calls = 0
        self._pending_repair: tuple[str, ResponseEvaluation] | None = None
        self._available_tools: tuple[str, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: ResponseGateConfig,
        *,
        model: BaseChatModel,
        system_prompt: str,
        log_writer: ResponseGateLogWriter | None = None,
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
            log_writer=log_writer,
            evaluator=evaluator,
        )

    async def awrap_model_call(self, request: ModelRequest, handler):
        """Capture capabilities and inject repair feedback for one model call."""
        self._available_tools = tuple(
            name
            for tool in request.tools
            if (name := self._tool_name(tool)) is not None
        )

        pending_repair = self._pending_repair
        if pending_repair is None:
            return await handler(request)

        draft, evaluation = pending_repair
        feedback = self._repair_feedback(evaluation)
        repair_message = self._repair_prompt.format(
            feedback=feedback,
            draft=self._truncate(draft, 4_000),
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
        session_id = getattr(runtime.context, "session_id", None)
        tool_traces = self._tool_traces_for_current_turn(
            state.get("messages", ())
        )
        tools_used = tuple(trace.name for trace in tool_traces)

        try:
            evaluation, usage = await self._evaluate(
                state.get("messages", ()),
                candidate,
                tool_traces,
            )
        except Exception as exc:
            # This is a conversational quality gate, not a reason to take Maia
            # offline when the evaluator or structured parsing is unavailable.
            logger.exception("Response gate failed; allowing the original response")
            await self._log_gate(
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
            self._pending_repair = (candidate.text, evaluation)
            await self._log_gate(
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
        messages: object,
        candidate: AIMessage,
        tool_traces: tuple[ToolTrace, ...],
    ) -> tuple[ResponseEvaluation, dict[str, Any] | None]:
        payload = {
            "maia_system_prompt": self._system_prompt,
            "available_tools": list(self._available_tools),
            "tool_traces": [
                trace.model_dump(mode="json")
                for trace in tool_traces
            ],
            "recent_conversation": self._conversation_excerpt(messages),
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

    async def _log_gate(
        self,
        *,
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
        if self._log_writer is None:
            return

        await self._log_writer.log_evaluation(
            session_id=session_id,
            model=self._model_name,
            evaluation_call=self._evaluation_calls,
            repair_attempt=self._repair_attempts,
            decision=decision,
            passed=passed,
            violations=violations,
            feedback=feedback,
            candidate_message_id=candidate.id,
            candidate=candidate.text,
            available_tools=list(self._available_tools),
            tools_used=list(tools_used),
            usage=usage,
            error=error,
        )

    @staticmethod
    def _latest_ai_message(messages: object) -> AIMessage | None:
        if not isinstance(messages, (list, tuple)):
            return None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None

    @classmethod
    def _conversation_excerpt(cls, messages: object) -> list[dict[str, str]]:
        """Return recent non-tool messages without duplicating tool traces."""
        if not isinstance(messages, (list, tuple)):
            return []

        excerpt: list[dict[str, str]] = []
        for message in reversed(messages[:-1]):
            if not isinstance(message, BaseMessage):
                continue
            if isinstance(message, ToolMessage):
                continue
            if isinstance(message, AIMessage) and message.tool_calls:
                continue
            excerpt.append({
                "role": message.type,
                "content": cls._truncate(message.text, 1_500),
            })
            if len(excerpt) == 6:
                break
        excerpt.reverse()
        return excerpt

    @staticmethod
    def _tool_traces_for_current_turn(
        messages: object,
    ) -> tuple[ToolTrace, ...]:
        """Pair current-turn tool calls with their complete ToolMessage evidence."""
        if not isinstance(messages, (list, tuple)):
            return ()

        turn_messages: list[BaseMessage] = []
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, BaseMessage):
                turn_messages.append(message)
        turn_messages.reverse()

        calls_by_id: dict[str, str] = {}
        traces: list[ToolTrace] = []
        for message in turn_messages:
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
                evidence=message.text,
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
