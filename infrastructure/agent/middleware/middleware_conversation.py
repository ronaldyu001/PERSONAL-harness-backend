"""Persist each conversation turn through the conversation port."""

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
from infrastructure.agent.middleware.helpers import (
    latest_completed_turn,
    latest_user_message,
)


logger = logging.getLogger(__name__)


class ConversationPersistenceMiddleware(AgentMiddleware):
    """Record the user message as the turn opens, and Maia's reply as it closes.

    The two halves are written at different times on purpose. Writing the user
    message first creates the conversation row before the agent makes a model
    call, so anything else recorded during the turn has a conversation to
    belong to.
    """

    def __init__(self, conversations: ConversationPort) -> None:
        self._conversations = conversations

    async def abefore_agent(self, state, runtime):
        """Write the user message, creating the conversation on first sight."""
        context = self._persisting_context(runtime)
        if context is None:
            return None

        user_message = latest_user_message(state.get("messages", ()))
        if user_message is None:
            return None

        await self._write(self._to_entity(user_message, "user", context))
        return None

    async def aafter_agent(self, state, runtime):
        """Write Maia's reply without touching agent state."""
        context = self._persisting_context(runtime)
        if context is None:
            return None

        turn = latest_completed_turn(state.get("messages", ()))
        if turn is None:
            return None

        _, assistant_message = turn
        # A tool-call-only response carries no text, and the conversation
        # entity rejects empty content.
        if not assistant_message.text.strip():
            return None

        await self._write(
            self._to_entity(
                assistant_message,
                "assistant",
                context,
                # Which model answered is only knowable at write time.
                metadata={"model": context.model},
            )
        )
        return None

    @staticmethod
    def _persisting_context(runtime) -> AgentRuntimeContext | None:
        """Return the context to persist under, or nothing when we must not."""
        context = getattr(runtime, "context", None)
        if not isinstance(context, AgentRuntimeContext):
            return None
        # A temporary turn leaves no transcript behind.
        if context.temporary:
            return None
        return context

    async def _write(self, message: ConversationMessage) -> None:
        """Persist one message; a store failure never breaks the turn."""
        try:
            await self._conversations.write(
                ConversationWriteRequest(message=message)
            )
        except Exception:
            logger.exception(
                "Conversation persistence failed; returning the agent response"
            )

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
