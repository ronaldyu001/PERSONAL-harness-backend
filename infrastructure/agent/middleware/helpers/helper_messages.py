"""Names that let one middleware recognize another's injected messages."""

from __future__ import annotations


# MiddlewareMemory prepends retrieved memories to the model request rather than
# writing them into agent state, so nothing downstream of the provider call can
# recover them from ``state["messages"]``. The response gate reads them back off
# the request by this name. Matching on ``SystemMessage`` alone would work today
# only because no other middleware prepends one.
USER_MEMORIES_MESSAGE_NAME = "user_memories"
