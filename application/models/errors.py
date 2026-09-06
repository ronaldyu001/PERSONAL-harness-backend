"""Errors raised across the model catalog boundary."""


class ListModelsError(RuntimeError):
    """Report a model gateway failure without exposing provider details."""
