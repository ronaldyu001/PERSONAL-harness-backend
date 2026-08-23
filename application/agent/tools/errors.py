"""Errors raised across the web-search tool boundary."""

from __future__ import annotations


class SearchWebError(RuntimeError):
    """Report a provider failure without exposing provider-specific details."""
