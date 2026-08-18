"""Tests for the provider-neutral search tool and LangSearch adapter."""

from __future__ import annotations

import json
import unittest

import httpx
from langchain_core.tools import ToolException

from application.agent.tools import SearchResponse, SearchResult, SearchWebError
from infrastructure.agent.tools import SearchWebTool
from infrastructure.agent.tools.adapters import LangSearchAdapter
from infrastructure.settings import load_infrastructure_settings


def search_config():
    settings = load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm:4000",
        "LANGSEARCH_API_KEY": "search-secret",
    })
    return settings.langsearch


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


class LangSearchAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_sends_config_and_returns_normalized_response(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["authorization"], "Bearer search-secret")
            self.assertEqual(json.loads(request.content), {
                "query": "Denver weather",
                "freshness": "oneDay",
                "summary": True,
                "count": 5,
            })
            return httpx.Response(200, json={
                "code": 200,
                "log_id": "request-1",
                "data": {
                    "webPages": {
                        "value": [{
                            "name": "Denver forecast",
                            "url": "https://weather.example/denver",
                            "snippet": "A short forecast.",
                            "summary": "A detailed forecast.",
                            "datePublished": "2026-08-17",
                        }]
                    }
                },
            })

        adapter = LangSearchAdapter.from_config(
            search_config(),
            transport=httpx.MockTransport(handler),
        )

        response = await adapter.search(
            " Denver weather ",
            freshness="day",
        )

        self.assertEqual(response.provider, "langsearch")
        self.assertEqual(response.query, "Denver weather")
        self.assertEqual(response.request_id, "request-1")
        self.assertEqual(response.results[0].title, "Denver forecast")

    async def test_provider_failure_raises_port_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="provider unavailable")

        adapter = LangSearchAdapter.from_config(
            search_config(),
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaisesRegex(SearchWebError, "LangSearch request failed"):
            await adapter.search("current information")


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
