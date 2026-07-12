"""Mem0 memory adapter package.

The adapter import touches the Mem0 package, which may initialize local Mem0
state. Keep it lazy so config-only imports stay side-effect light.
"""

__all__ = ["Mem0Adapter"]


def __getattr__(name: str):
    """Lazily expose the Mem0 adapter."""
    if name == "Mem0Adapter":
        from infrastructure.memory.Mem0_adapter.Mem0_adapter import Mem0Adapter

        return Mem0Adapter

    raise AttributeError(name)
