"""Provider-neutral contract for searching the public web."""

from __future__ import annotations

from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


SearchFreshness: TypeAlias = Literal[
    "day",
    "week",
    "month",
    "year",
    "any",
]


class _SearchSchema(BaseModel):
    """Keep search data immutable and reject unexpected provider fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchResult(_SearchSchema):
    """One normalized web result returned by any search provider."""

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str | None = None
    summary: str | None = None
    date_published: str | None = None


class SearchResponse(_SearchSchema):
    """Provider-neutral result set plus trace metadata for one search."""

    provider: str = Field(min_length=1)
    query: str = Field(min_length=1)
    request_id: str | None = None
    results: tuple[SearchResult, ...]


class SearchWebError(RuntimeError):
    """Report a provider failure without exposing provider-specific details."""


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
