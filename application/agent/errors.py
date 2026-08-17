"""Application errors raised by conversational agents."""


class EmptyAgentResponseError(RuntimeError):
    """Raised when an agent completes without user-visible content."""
