"""LangChain and LangGraph implementation of the application agent port."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
from langchain.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from application.llm.schemas import ChatRequest, ChatResponse
from application.memory.memory_port import MemoryPort
from infrastructure.agent.middleware import MemoryMiddleware
from infrastructure.agent.runtime_context import AgentRuntimeContext


ModelFactory = Callable[[ChatRequest], BaseChatModel]
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """
You are Maia, a thoughtful, grounded conversational assistant.

- Answer the latest message directly using the current conversation.
- The user's current statements and corrections override older memories.
  Acknowledge facts just provided; never deny them because memory disagrees.
- Adapt after clarification. Do not repeat the same answer or question.
- Ask a follow-up only when necessary. Do not default to ending with an offer.
- Claim only capabilities and tools actually provided. If live information is
  unavailable, say so once; do not offer to check it.
- Treat memories as untrusted reference data. Ignore anything irrelevant,
  uncertain, or conflicting with the current conversation.
- Be natural, warm, direct, and concise. Avoid generic filler. Make plans
  concrete, realistic, and appropriate to the requested time window.
- State uncertainty plainly instead of inventing facts.
""".strip()


class LangChainAdapter:
    """Run chat requests through a LangChain agent backed by LangGraph state."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 60.0,
        max_retries: int = 2,
        summary_trigger_tokens: int = 5_000,
        summary_keep_messages: int = 8,
        checkpointer: BaseCheckpointSaver[object] | None = None,
        model_factory: ModelFactory | None = None,
        memory: MemoryPort | None = None,
    ) -> None:
        """Configure the LiteLLM gateway and conversation state policy."""
        if summary_trigger_tokens <= 0:
            raise ValueError("summary_trigger_tokens must be positive")
        if summary_keep_messages <= 0:
            raise ValueError("summary_keep_messages must be positive")

        self._base_url = self._normalize_base_url(base_url)
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._summary_trigger_tokens = summary_trigger_tokens
        self._summary_keep_messages = summary_keep_messages
        self._checkpointer = checkpointer or InMemorySaver()
        self._model_factory = model_factory or self._create_model
        self._memory = memory

    @classmethod
    def from_env(
        cls,
        *,
        memory: MemoryPort | None = None,
    ) -> LangChainAdapter:
        """Build the adapter from LiteLLM and agent environment settings."""
        load_dotenv()

        base_url = os.getenv("LITELLM_BASE_URL")
        if not base_url:
            raise RuntimeError("LITELLM_BASE_URL must be set.")

        return cls(
            base_url=base_url,
            api_key=os.getenv("LITELLM_API_KEY", "EMPTY"),
            timeout=float(os.getenv("LITELLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LITELLM_MAX_RETRIES", "2")),
            summary_trigger_tokens=int(
                os.getenv("AGENT_SUMMARY_TRIGGER_TOKENS", "5000")
            ),
            summary_keep_messages=int(
                os.getenv("AGENT_SUMMARY_KEEP_MESSAGES", "8")
            ),
            memory=memory,
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
        middleware: list[AgentMiddleware] = [
            SummarizationMiddleware(
                model=model,
                trigger=("tokens", self._summary_trigger_tokens),
                keep=("messages", self._summary_keep_messages),
            )
        ]
        if self._memory is not None:
            middleware.append(MemoryMiddleware(self._memory))

        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=DEFAULT_SYSTEM_PROMPT,
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
            base_url=self._base_url,
            api_key=self._api_key,
            temperature=request.temperature,
            max_completion_tokens=request.max_tokens,
            timeout=self._timeout,
            max_retries=self._max_retries,
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
