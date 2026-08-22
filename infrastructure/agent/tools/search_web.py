"""LangChain tool adapter for provider-neutral web search."""

from __future__ import annotations

import math
from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from application.agent.tools import (
    SearchFreshness,
    SearchResponse,
    SearchResult,
    SearchWebError,
    SearchWebPort,
)


_EVIDENCE_PREAMBLE = (
    "Present only facts supported by the search evidence below. Preserve the "
    "user's key constraints and do not infer missing details. If no exact result "
    "matches, say so and label any closest alternatives. Cite returned URLs."
)


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


class SearchWebTool:
    """Expose any search port as bounded model context and a LangChain artifact."""

    def __init__(
        self,
        *,
        search: SearchWebPort,
        max_context_tokens: int,
    ) -> None:
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        self._search = search
        self._max_context_tokens = max_context_tokens

    def as_tool(self) -> StructuredTool:
        """Return the model-facing LangChain tool around the injected provider."""
        return StructuredTool.from_function(
            coroutine=self.search,
            name="search_web",
            description=(
                "Use this to retrieve live web results for current, local, niche, or externally "
                "verifiable information. Faithfully present or summarize only the "
                "returned evidence and cite its URLs. Preserve the user's key "
                "constraints; do not fill gaps with assumptions. Do not call this "
                "tool unecessarily."
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
        """Run the provider and split its response into context and artifact."""
        try:
            response = await self._search.search(
                query,
                freshness=freshness,
            )
        except SearchWebError as exc:
            raise ToolException(
                "Live web search is temporarily unavailable. Do not invent "
                "current information; briefly tell the user the search failed."
            ) from exc

        artifact = response.model_dump(exclude_none=True, mode="json")
        return self._model_context(response), artifact

    def _model_context(self, response: SearchResponse) -> str:
        """Format ranked results into the bounded text that the model can see."""
        if not response.results:
            return f'No web results were found for "{response.query}".'

        header = f'Web results for "{response.query}":'
        blocks = [_EVIDENCE_PREAMBLE, header]
        for index, result in enumerate(response.results, start=1):
            block = self._result_block(index, result)
            candidate = "\n\n".join([*blocks, block])
            if self._estimated_tokens(candidate) <= self._max_context_tokens:
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

    @staticmethod
    def _result_block(index: int, result: SearchResult) -> str:
        """Render one normalized result without any provider-specific fields."""
        details = [f"[{index}] {result.title}", f"URL: {result.url}"]
        if result.date_published:
            details.append(f"Published: {result.date_published}")
        content = result.summary or result.snippet
        if content:
            details.append(f"Content: {content}")
        return "\n".join(details)

    def _fit_to_budget(self, *, prefix: str, value: str) -> str:
        """Binary-search the largest result fragment that fits the token budget."""
        low = 0
        high = len(value)
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate = f"{prefix}\n\n{value[:midpoint]}…"
            if self._estimated_tokens(candidate) <= self._max_context_tokens:
                low = midpoint
            else:
                high = midpoint - 1
        return f"{value[:low].rstrip()}…" if low else ""

    @staticmethod
    def _estimated_tokens(value: str) -> int:
        """Estimate tokens conservatively without invoking a model tokenizer."""
        return math.ceil(len(value) / 4) + 3
