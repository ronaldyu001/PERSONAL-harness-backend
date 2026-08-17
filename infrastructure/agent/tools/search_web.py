"""Bounded LangSearch web-search tool for Maia."""

from __future__ import annotations

import math
from typing import Any

import httpx
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from infrastructure.settings import LangSearchConfig, SearchFreshness


class WebSearchInput(BaseModel):
    """Arguments the model may provide to the web-search tool."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="A focused web-search query.",
    )
    freshness: SearchFreshness | None = Field(
        default=None,
        description="Optional time range; omit when no recency filter is needed.",
    )


class WebSearchResult(BaseModel):
    """Normalized subset of one LangSearch result."""

    title: str
    url: str
    snippet: str | None = None
    summary: str | None = None
    date_published: str | None = None


class LangSearchWebSearch:
    """Call LangSearch and expose only bounded, normalized context to the model."""

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
    ) -> LangSearchWebSearch:
        """Build the search service from resolved infrastructure settings."""
        return cls(config=config, transport=transport)

    def as_tool(self) -> StructuredTool:
        """Return the model-facing LangChain tool."""
        return StructuredTool.from_function(
            coroutine=self.search,
            name="search_web",
            description=(
                "Search the live web for current, local, niche, or externally "
                "verifiable information. Use only when the conversation alone "
                "cannot answer reliably. Base the answer on the results and cite "
                "their URLs."
            ),
            args_schema=WebSearchInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
        )

    async def search(
        self,
        query: str,
        freshness: SearchFreshness | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return bounded model context plus normalized result metadata."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ToolException("Web search requires a non-empty query.")

        payload = {
            "query": normalized_query,
            "freshness": freshness or self._config.freshness,
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
            raise ToolException(
                "Live web search is temporarily unavailable. Do not invent "
                "current information; briefly tell the user the search failed."
            ) from exc

        artifact = {
            "provider": "langsearch",
            "query": normalized_query,
            "log_id": body.get("log_id") if isinstance(body, dict) else None,
            "results": [result.model_dump(exclude_none=True) for result in results],
        }
        return self._model_context(normalized_query, results), artifact

    def _parse_results(self, body: object) -> list[WebSearchResult]:
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

        results: list[WebSearchResult] = []
        for value in values[: self._config.result_count]:
            if not isinstance(value, dict):
                continue
            title = self._text(value.get("name"))
            url = self._text(value.get("url"))
            if not title or not url:
                continue
            results.append(WebSearchResult(
                title=title,
                url=url,
                snippet=self._text(value.get("snippet")) or None,
                summary=self._text(value.get("summary")) or None,
                date_published=self._text(value.get("datePublished")) or None,
            ))
        return results

    def _model_context(
        self,
        query: str,
        results: list[WebSearchResult],
    ) -> str:
        """Format ranked results into the bounded text that the model can see."""
        if not results:
            return f'No web results were found for "{query}".'

        header = f'Web results for "{query}":'
        blocks = [header]
        for index, result in enumerate(results, start=1):
            details = [f"[{index}] {result.title}", f"URL: {result.url}"]
            if result.date_published:
                details.append(f"Published: {result.date_published}")
            content = result.summary or result.snippet
            if content:
                details.append(f"Content: {content}")
            block = "\n".join(details)
            candidate = "\n\n".join([*blocks, block])
            if self._estimated_tokens(candidate) <= self._config.max_context_tokens:
                blocks.append(block)
                continue

            remaining = self._fit_to_budget(
                prefix="\n\n".join(blocks),
                value=block,
            )
            if remaining:
                blocks.append(remaining)
            break
        return "\n\n".join(blocks)

    def _fit_to_budget(self, *, prefix: str, value: str) -> str:
        """Binary-search the largest result fragment that fits the token budget."""
        low = 0
        high = len(value)
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate = f"{prefix}\n\n{value[:midpoint]}…"
            if self._estimated_tokens(candidate) <= self._config.max_context_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        return f"{value[:low].rstrip()}…" if low else ""

    @staticmethod
    def _estimated_tokens(value: str) -> int:
        """Estimate tokens conservatively without invoking a model tokenizer."""
        return math.ceil(len(value) / 4) + 3

    @staticmethod
    def _text(value: object) -> str:
        """Accept only provider string fields and normalize surrounding space."""
        return value.strip() if isinstance(value, str) else ""
