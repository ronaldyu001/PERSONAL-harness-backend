"""Provider-neutral contract for searching the public web."""

from __future__ import annotations

from typing import Protocol

from application.agent.tools.schemas import SearchFreshness, SearchResponse


class SearchWebPort(Protocol):
    """Contract implemented by interchangeable web-search providers."""

    async def search(
        self,
        query: str,
        *,
        freshness: SearchFreshness | None = None,
    ) -> SearchResponse:
        """Search the web and return normalized results."""
        ...
