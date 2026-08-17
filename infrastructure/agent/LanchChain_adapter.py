"""LangChain and LangGraph implementation of the application agent port."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from application.llm.schemas import ChatRequest, ChatResponse
from application.memory.memory_port import MemoryPort
from infrastructure.agent.logging import ResponseGateLogWriter
from infrastructure.agent.middleware import (
    ContextLoggingMiddleware,
    MemoryMiddleware,
    ModelResponseGateMiddleware,
)
from infrastructure.agent.runtime_context import AgentRuntimeContext
from infrastructure.agent.tools import SearchWebTool
from infrastructure.agent.tools.adapters import LangSearchAdapter
from infrastructure.settings import (
    AgentConfig,
    GatewayConfig,
    InfrastructureSettings,
    LoggingConfig,
)


ModelFactory = Callable[[ChatRequest], BaseChatModel]
logger = logging.getLogger(__name__)

class LangChainAdapter:
    """Run chat requests through a LangChain agent backed by LangGraph state."""

    def __init__(
        self,
        *,
        gateway_config: GatewayConfig,
        agent_config: AgentConfig,
        logging_config: LoggingConfig,
        checkpointer: BaseCheckpointSaver[object] | None = None,
        model_factory: ModelFactory | None = None,
        memory: MemoryPort | None = None,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        """Configure the agent from explicit infrastructure sections."""
        self._gateway_config = gateway_config
        self._agent_config = agent_config
        self._logging_config = logging_config
        self._checkpointer = checkpointer or InMemorySaver()
        self._model_factory = model_factory or self._create_model
        self._memory = memory
        self._tools = tuple(tools)

    @classmethod
    def from_config(
        cls,
        config: InfrastructureSettings,
        *,
        checkpointer: BaseCheckpointSaver[object] | None = None,
        model_factory: ModelFactory | None = None,
        memory: MemoryPort | None = None,
        tools: Sequence[BaseTool] | None = None,
    ) -> LangChainAdapter:
        """Build the agent from the resolved infrastructure configuration."""
        return cls(
            gateway_config=config.gateway,
            agent_config=config.agent,
            logging_config=config.logging,
            checkpointer=checkpointer,
            model_factory=model_factory,
            memory=memory,
            tools=(
                tuple(tools)
                if tools is not None
                else cls._tools_from_config(config)
            ),
        )

    async def chat(
        self,
        request: ChatRequest,
        *,
        session_id: str,
        user_id: str,
    ) -> ChatResponse:
        """Invoke the agent and map its final message to an application response."""
        model = self._model_factory(request)
        agent_config = self._agent_config
        middleware: list[AgentMiddleware] = [
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", agent_config.summarization.trigger_tokens),
                keep=("messages", agent_config.summarization.keep_messages),
            )
        ]
        context_logging = ContextLoggingMiddleware.from_config(
            self._logging_config
        )
        response_gate_log_writer = ResponseGateLogWriter.from_config(
            self._logging_config
        )
        if self._memory is not None:
            middleware.append(
                MemoryMiddleware.from_config(
                    agent_config.memory,
                    memory=self._memory,
                )
            )
        if agent_config.response_gate.enabled:
            middleware.append(
                ModelResponseGateMiddleware.from_config(
                    agent_config.response_gate,
                    model=model,
                    system_prompt=agent_config.system_prompt,
                    log_writer=response_gate_log_writer,
                )
            )
        if context_logging.enabled:
            # Keep this last so it sees transient context injected by earlier
            # model-call middleware immediately before the provider call.
            middleware.append(context_logging)

        agent = create_agent(
            model=model,
            tools=list(self._tools),
            system_prompt=agent_config.system_prompt,
            middleware=middleware,
            checkpointer=self._checkpointer,
            context_schema=AgentRuntimeContext,
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ]
            },
            {
                "configurable": {
                    "thread_id": session_id,
                }
            },
            context=AgentRuntimeContext(
                user_id=user_id,
                session_id=session_id,
            ),
        )

        final_message = self._last_ai_message(result.get("messages", ()))
        usage = final_message.usage_metadata
        if usage is None:
            usage = final_message.response_metadata.get("token_usage")
        finish_reason = self._finish_reason(final_message)
        logger.info("Agent completed with finish_reason=%r", finish_reason)

        return ChatResponse(
            content=final_message.text,
            usage=dict(usage) if usage else None,
            finish_reason=finish_reason,
        )

    def _create_model(self, request: ChatRequest) -> BaseChatModel:
        """Create a LangChain chat model targeting the LiteLLM proxy."""
        return ChatOpenAI(
            model=request.model,
            base_url=self._normalize_base_url(self._gateway_config.base_url),
            api_key=self._gateway_config.api_key,
            temperature=request.temperature,
            max_completion_tokens=request.max_tokens,
            timeout=self._gateway_config.timeout_seconds,
            max_retries=self._gateway_config.max_retries,
            use_responses_api=False,
        )

    @staticmethod
    def _last_ai_message(messages: object) -> AIMessage:
        """Return the final assistant message from an agent result."""
        if not isinstance(messages, (list, tuple)):
            raise RuntimeError("agent result did not contain a message sequence")

        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message

        raise RuntimeError("agent result did not contain an assistant response")

    @staticmethod
    def _finish_reason(message: AIMessage) -> str | None:
        """Return the provider's completion reason without interpretation."""
        raw_reason = message.response_metadata.get("finish_reason")
        if raw_reason is None:
            raw_reason = message.response_metadata.get("stop_reason")
        return raw_reason if isinstance(raw_reason, str) and raw_reason else None

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Return the OpenAI-compatible API root exposed by LiteLLM."""
        normalized = base_url.rstrip("/")
        chat_completions_suffix = "/chat/completions"

        if normalized.endswith(chat_completions_suffix):
            normalized = normalized[: -len(chat_completions_suffix)]
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"

        return normalized

    @staticmethod
    def _tools_from_config(
        config: InfrastructureSettings,
    ) -> tuple[BaseTool, ...]:
        """Build optional agent tools whose deployment credentials are present."""
        if config.langsearch.api_key is None:
            return ()
        provider = LangSearchAdapter.from_config(config.langsearch)
        tool = SearchWebTool(
            search=provider,
            max_context_tokens=config.langsearch.max_context_tokens,
        )
        return (tool.as_tool(),)
