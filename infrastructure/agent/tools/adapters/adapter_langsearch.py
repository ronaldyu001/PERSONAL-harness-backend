"""LangSearch implementation of the application web-search port."""

from __future__ import annotations

import httpx

from application.agent.tools import (
    SearchFreshness,
    SearchResponse,
    SearchResult,
    SearchWebError,
)
from infrastructure.settings import LangSearchConfig


_LANGSEARCH_FRESHNESS: dict[SearchFreshness, str] = {
    "day": "oneDay",
    "week": "oneWeek",
    "month": "oneMonth",
    "year": "oneYear",
    "any": "noLimit",
}


class LangSearchAdapter:
    """Call LangSearch and normalize its response into application schemas."""

    def __init__(
        self,
        *,
        config: LangSearchConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if config.api_key is None:
            raise ValueError("LANGSEARCH_API_KEY is required to enable web search")
        self._config = config
        self._transport = transport

    @classmethod
    def from_config(
        cls,
        config: LangSearchConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> LangSearchAdapter:
        """Build the provider adapter from resolved infrastructure settings."""
        return cls(config=config, transport=transport)

    async def search(
        self,
        query: str,
        *,
        freshness: SearchFreshness | None = None,
    ) -> SearchResponse:
        """Execute a LangSearch request and return provider-neutral results."""
        normalized_query = query.strip()
        if not normalized_query:
            raise SearchWebError("Web search requires a non-empty query.")

        payload = {
            "query": normalized_query,
            "freshness": _LANGSEARCH_FRESHNESS[
                freshness or self._config.freshness
            ],
            "summary": self._config.include_summaries,
            "count": self._config.result_count,
        }
        headers = {
            "Authorization": (
                f"Bearer {self._config.api_key.get_secret_value()}"
            ),
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self._config.base_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            results = self._parse_results(body)
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise SearchWebError("LangSearch request failed.") from exc

        return SearchResponse(
            provider="langsearch",
            query=normalized_query,
            request_id=(
                self._text(body.get("log_id")) or None
                if isinstance(body, dict)
                else None
            ),
            results=tuple(results),
        )

    def _parse_results(self, body: object) -> list[SearchResult]:
        """Validate the provider envelope and normalize usable web-page fields."""
        if not isinstance(body, dict) or body.get("code") != 200:
            raise ValueError("LangSearch returned an unsuccessful response")

        data = body.get("data")
        web_pages = data.get("webPages") if isinstance(data, dict) else None
        values = web_pages.get("value") if isinstance(web_pages, dict) else None
        if values is None:
            return []
        if not isinstance(values, list):
            raise TypeError("LangSearch webPages.value must be a list")

        results: list[SearchResult] = []
        for value in values[: self._config.result_count]:
            if not isinstance(value, dict):
                continue
            title = self._text(value.get("name"))
            url = self._text(value.get("url"))
            if not title or not url:
                continue
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=self._text(value.get("snippet")) or None,
                summary=self._text(value.get("summary")) or None,
                date_published=self._text(value.get("datePublished")) or None,
            ))
        return results

    @staticmethod
    def _text(value: object) -> str:
        """Accept only provider string fields and normalize surrounding space."""
        return value.strip() if isinstance(value, str) else ""
