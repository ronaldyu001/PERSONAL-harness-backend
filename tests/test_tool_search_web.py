"""Tests for the provider-neutral web-search tool."""

from __future__ import annotations

import unittest

from langchain_core.tools import ToolException

from application.agent.tools import SearchResponse, SearchResult, SearchWebError
from infrastructure.agent.tools import SearchWebTool


class RecordingSearchProvider:
    """Provider-neutral test double used to isolate the LangChain tool adapter."""

    def __init__(
        self,
        *,
        response: SearchResponse | None = None,
        error: SearchWebError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.query: str | None = None
        self.freshness: str | None = None

    async def search(self, query: str, *, freshness=None) -> SearchResponse:
        self.query = query
        self.freshness = freshness
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class SearchWebToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_returns_bounded_context_and_full_artifact(self) -> None:
        response = SearchResponse(
            provider="test-search",
            query="bounded result",
            request_id="trace-1",
            results=(SearchResult(
                title="Long result",
                url="https://example.test/long",
                summary="detail " * 1000,
            ),),
        )
        provider = RecordingSearchProvider(response=response)
        search = SearchWebTool(search=provider, max_context_tokens=80)

        content, artifact = await search.search(
            "bounded result",
            freshness="week",
        )

        self.assertLessEqual(search._estimated_tokens(content), 80)
        self.assertTrue(content.endswith("…"))
        self.assertIn("Present only facts supported", content)
        self.assertIn("do not infer missing details", content)
        self.assertEqual(artifact["provider"], "test-search")
        self.assertGreater(
            len(artifact["results"][0]["summary"]),
            len(content),
        )
        self.assertEqual(provider.query, "bounded result")
        self.assertEqual(provider.freshness, "week")

    async def test_tool_converts_port_failure_to_safe_tool_error(self) -> None:
        provider = RecordingSearchProvider(
            error=SearchWebError("provider-specific failure")
        )
        search = SearchWebTool(search=provider, max_context_tokens=2000)

        with self.assertRaisesRegex(ToolException, "temporarily unavailable"):
            await search.search("current information")

    def test_langchain_tool_uses_content_and_artifact_response(self) -> None:
        provider = RecordingSearchProvider(response=SearchResponse(
            provider="test-search",
            query="query",
            results=(),
        ))
        search = SearchWebTool(search=provider, max_context_tokens=2000)

        tool = search.as_tool()

        self.assertEqual(tool.name, "search_web")
        self.assertEqual(tool.response_format, "content_and_artifact")
        self.assertTrue(tool.handle_tool_error)


if __name__ == "__main__":
    unittest.main()
