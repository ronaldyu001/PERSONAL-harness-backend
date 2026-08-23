"""Persist each completed conversation turn through the conversation port."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, HumanMessage

from application.conversation import (
    ConversationPort,
    ConversationWriteRequest,
)
from domain.entities import (
    ConversationMessage,
    ConversationMessageRole,
)
from infrastructure.agent.context import AgentRuntimeContext
from infrastructure.agent.middleware.helpers import latest_completed_turn


logger = logging.getLogger(__name__)


class ConversationPersistenceMiddleware(AgentMiddleware):
    """Record the user message and Maia's reply once a turn completes."""

    def __init__(self, conversations: ConversationPort) -> None:
        self._conversations = conversations

    async def aafter_agent(self, state, runtime):
        """Write one completed turn without touching agent state."""
        if not isinstance(runtime.context, AgentRuntimeContext):
            return None

        # A temporary turn leaves no transcript behind.
        if runtime.context.temporary:
            return None

        turn = latest_completed_turn(state.get("messages", ()))
        if turn is None:
            return None

        user_message, assistant_message = turn
        # A tool-call-only response carries no text, and the conversation
        # entity rejects empty content.
        if not assistant_message.text.strip():
            return None

        try:
            messages = (
                self._to_entity(user_message, "user", runtime.context),
                self._to_entity(
                    assistant_message,
                    "assistant",
                    runtime.context,
                    # Which model answered is only knowable at write time.
                    metadata={"model": runtime.context.model},
                ),
            )
            for message in messages:
                await self._conversations.write(
                    ConversationWriteRequest(message=message)
                )
        except Exception:
            logger.exception(
                "Conversation persistence failed; returning the agent response"
            )

        return None

    @staticmethod
    def _to_entity(
        message: HumanMessage | AIMessage,
        role: ConversationMessageRole,
        context: AgentRuntimeContext,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConversationMessage:
        """Map one LangChain message onto the conversation domain entity."""
        return ConversationMessage(
            conversation_id=context.session_id,
            role=role,
            content=message.text,
            # The agent's own id keeps a re-run of this turn idempotent.
            message_id=message.id,
            user_id=context.user_id,
            metadata={
                key: value
                for key, value in (metadata or {}).items()
                if value is not None
            },
        )
