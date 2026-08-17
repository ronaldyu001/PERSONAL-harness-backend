"""Tests for Maia's final-response quality gate."""

from __future__ import annotations

import tempfile
import unittest

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langgraph.graph.message import RemoveMessage
from langgraph.runtime import Runtime

from infrastructure.agent.logging import (
    ResponseGateLogEvent,
    ResponseGateLogWriter,
)
from infrastructure.agent.middleware.modelResponseGate_middleware import (
    ModelResponseGateMiddleware,
    ResponseEvaluation,
)
from infrastructure.agent.runtime_context import AgentRuntimeContext
from infrastructure.settings import load_infrastructure_settings


class QueueEvaluator:
    """Return predefined structured gate decisions."""

    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.requests: list[object] = []

    async def ainvoke(self, request):
        self.requests.append(request)
        return self._results.pop(0)


def runtime() -> Runtime[AgentRuntimeContext]:
    return Runtime(
        context=AgentRuntimeContext(
            user_id="user-1",
            session_id="session-1",
        )
    )


def model() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="unused")]
    )


def gate_config(*, max_repairs: int = 1):
    config = load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm:4000",
    }).agent.response_gate
    return config.model_copy(update={"max_repairs": max_repairs})


class ModelResponseGateMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_evaluator_usage_to_the_gate_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gate_writer = ResponseGateLogWriter(
                mode="full",
                log_dir=temp_dir,
            )
            evaluator = QueueEvaluator({
                "parsed": ResponseEvaluation(
                    passed=True,
                    violations=[],
                    feedback="",
                ),
                "raw": AIMessage(
                    content="",
                    usage_metadata={
                        "input_tokens": 80,
                        "output_tokens": 12,
                        "total_tokens": 92,
                    },
                ),
                "parsing_error": None,
            })
            middleware = ModelResponseGateMiddleware.from_config(
                gate_config(),
                model=model(),
                system_prompt="Answer directly.",
                log_writer=gate_writer,
                evaluator=evaluator,
            )

            await middleware.aafter_model(
                {
                    "messages": [
                        HumanMessage(content="Which game should I play?"),
                        AIMessage(content="Play Pokemon.", id="candidate-1"),
                    ]
                },
                runtime(),
            )

            event = ResponseGateLogEvent.model_validate_json(
                gate_writer.log_path.read_text(encoding="utf-8")
            )
            self.assertEqual(event.decision, "allow")
            self.assertEqual(event.candidate, "Play Pokemon.")
            self.assertEqual(event.usage, {
                "input_tokens": 80,
                "output_tokens": 12,
                "total_tokens": 92,
            })

    async def test_create_agent_retries_without_persisting_rejected_draft(self) -> None:
        chat_model = FakeMessagesListChatModel(
            responses=[
                AIMessage(content="I can check DoorDash for you."),
                AIMessage(content="I can't check DoorDash, but I can compare options you share."),
            ]
        )
        evaluator = QueueEvaluator(
            ResponseEvaluation(
                passed=False,
                violations=["Offered an unavailable delivery search."],
                feedback="State the limitation and answer without that offer.",
            ),
            ResponseEvaluation(passed=True, violations=[], feedback=""),
        )
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=chat_model,
            system_prompt="Use only available tools.",
            evaluator=evaluator,
        )
        agent = create_agent(
            model=chat_model,
            tools=[],
            system_prompt="Use only available tools.",
            middleware=[middleware],
            context_schema=AgentRuntimeContext,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Can you check DoorDash?"}]},
            context=AgentRuntimeContext(
                user_id="user-1",
                session_id="session-1",
            ),
        )

        self.assertEqual(
            [message.text for message in result["messages"]],
            [
                "Can you check DoorDash?",
                "I can't check DoorDash, but I can compare options you share.",
            ],
        )
        self.assertEqual(len(evaluator.requests), 2)

    async def test_allows_a_passing_final_response(self) -> None:
        evaluator = QueueEvaluator(
            ResponseEvaluation(passed=True, violations=[], feedback="")
        )
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=model(),
            system_prompt="Answer directly.",
            evaluator=evaluator,
        )

        result = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="Which game should I play?"),
                    AIMessage(content="Play Pokemon.", id="candidate-1"),
                ]
            },
            runtime(),
        )

        self.assertIsNone(result)
        self.assertEqual(len(evaluator.requests), 1)

    async def test_recovers_a_markdown_failure_verdict_from_local_model(self) -> None:
        evaluator = QueueEvaluator(AIMessage(content=(
            "**Evaluation:** FAIL\n\n"
            "**Failure Reason:** Offered to check a live menu without tools."
        )))
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=model(),
            system_prompt="Use only available tools.",
            evaluator=evaluator,
        )

        update = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="Can you check the menu?"),
                    AIMessage(
                        content="I can check the live menu.",
                        id="candidate-1",
                    ),
                ]
            },
            runtime(),
        )

        assert update is not None
        self.assertEqual(update["jump_to"], "model")
        self.assertEqual(update["messages"][0].id, "candidate-1")

    async def test_accepts_null_feedback_and_violations_on_json_pass(self) -> None:
        evaluator = QueueEvaluator(AIMessage(content=(
            '{"passed": true, "violations": null, "feedback": null}'
        )))
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=model(),
            system_prompt="Answer directly.",
            evaluator=evaluator,
        )

        result = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="What should I eat?"),
                    AIMessage(
                        content="Have a high-protein meal.",
                        id="candidate-1",
                    ),
                ]
            },
            runtime(),
        )

        self.assertIsNone(result)

    async def test_recovers_a_truncated_json_failure_verdict(self) -> None:
        evaluator = QueueEvaluator(AIMessage(content=(
            '{"passed": false, "violations": ["Invented a preference"], '
            '"feedback": "Remove the unsupported preference."'
        )))
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=model(),
            system_prompt="Use only supported facts.",
            evaluator=evaluator,
        )

        update = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="I already ordered falafel."),
                    AIMessage(
                        content="Rice conflicts with your usual preference.",
                        id="candidate-1",
                    ),
                ]
            },
            runtime(),
        )

        assert update is not None
        self.assertEqual(update["jump_to"], "model")
        self.assertEqual(update["messages"][0].id, "candidate-1")

    async def test_rejects_then_injects_transient_repair_feedback(self) -> None:
        evaluator = QueueEvaluator(
            ResponseEvaluation(
                passed=False,
                violations=["Offered an unavailable web search."],
                feedback="Answer without offering to browse.",
            )
        )
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=model(),
            system_prompt="Use only available tools.",
            evaluator=evaluator,
        )
        candidate = AIMessage(
            content="I can search the web for that.",
            id="candidate-1",
        )

        update = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="Do you know any options?"),
                    candidate,
                ]
            },
            runtime(),
        )

        assert update is not None
        self.assertEqual(update["jump_to"], "model")
        self.assertIsInstance(update["messages"][0], RemoveMessage)
        self.assertEqual(update["messages"][0].id, "candidate-1")

        request = ModelRequest(
            model=model(),
            system_message=SystemMessage(content="Use only available tools."),
            messages=[HumanMessage(content="Do you know any options?")],
            tools=[],
        )
        received: list[ModelRequest] = []

        async def handler(repaired_request):
            received.append(repaired_request)
            return AIMessage(content="I can't search, but here are general options.")

        await middleware.awrap_model_call(request, handler)
        self.assertEqual(len(received), 1)
        repair_system = received[0].system_message
        assert repair_system is not None
        self.assertIn("Offered an unavailable web search", repair_system.text)
        self.assertIn(candidate.text, repair_system.text)
        self.assertEqual(request.system_message.text, "Use only available tools.")

    async def test_skips_responses_with_pending_tool_calls(self) -> None:
        evaluator = QueueEvaluator()
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(),
            model=model(),
            system_prompt="Use tools when needed.",
            evaluator=evaluator,
        )

        result = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="Search for this."),
                    AIMessage(
                        content="",
                        id="candidate-1",
                        tool_calls=[{
                            "name": "web_search",
                            "args": {"query": "this"},
                            "id": "call-1",
                            "type": "tool_call",
                        }],
                    ),
                ]
            },
            runtime(),
        )

        self.assertIsNone(result)
        self.assertEqual(evaluator.requests, [])

    async def test_replaces_a_rejected_response_when_retries_are_exhausted(self) -> None:
        evaluator = QueueEvaluator(
            ResponseEvaluation(
                passed=False,
                violations=["Unsupported capability claim."],
                feedback="Remove the claim.",
            )
        )
        middleware = ModelResponseGateMiddleware.from_config(
            gate_config(max_repairs=0),
            model=model(),
            system_prompt="Be reliable.",
            evaluator=evaluator,
        )

        update = await middleware.aafter_model(
            {
                "messages": [
                    HumanMessage(content="Can you check?"),
                    AIMessage(content="Yes, I checked.", id="candidate-1"),
                ]
            },
            runtime(),
        )

        assert update is not None
        self.assertNotIn("jump_to", update)
        self.assertIsInstance(update["messages"][0], RemoveMessage)
        self.assertEqual(
            update["messages"][1].text,
            gate_config(max_repairs=0).fallback_response,
        )


if __name__ == "__main__":
    unittest.main()
