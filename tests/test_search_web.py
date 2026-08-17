"""Tests for the bounded LangSearch agent tool."""

from __future__ import annotations

import json
import unittest

import httpx
from langchain_core.tools import ToolException

from infrastructure.agent.tools import LangSearchWebSearch
from infrastructure.settings import load_infrastructure_settings


def search_config(*, max_context_tokens: int = 2000):
    settings = load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm:4000",
        "LANGSEARCH_API_KEY": "search-secret",
    })
    return settings.langsearch.model_copy(
        update={"max_context_tokens": max_context_tokens}
    )


class LangSearchWebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_sends_config_and_returns_context_and_artifact(self) -> None:
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

        search = LangSearchWebSearch.from_config(
            search_config(),
            transport=httpx.MockTransport(handler),
        )

        content, artifact = await search.search(
            " Denver weather ",
            freshness="oneDay",
        )

        self.assertIn("Denver forecast", content)
        self.assertIn("https://weather.example/denver", content)
        self.assertEqual(artifact["provider"], "langsearch")
        self.assertEqual(artifact["log_id"], "request-1")
        self.assertEqual(len(artifact["results"]), 1)

    async def test_search_truncates_model_context_to_configured_budget(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "code": 200,
                "data": {
                    "webPages": {
                        "value": [{
                            "name": "Long result",
                            "url": "https://example.test/long",
                            "summary": "detail " * 1000,
                        }]
                    }
                },
            })

        search = LangSearchWebSearch.from_config(
            search_config(max_context_tokens=80),
            transport=httpx.MockTransport(handler),
        )

        content, artifact = await search.search("bounded result")

        self.assertLessEqual(search._estimated_tokens(content), 80)
        self.assertTrue(content.endswith("…"))
        self.assertGreater(len(artifact["results"][0]["summary"]), len(content))

    async def test_search_converts_provider_failure_to_safe_tool_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="provider unavailable")

        search = LangSearchWebSearch.from_config(
            search_config(),
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaisesRegex(ToolException, "temporarily unavailable"):
            await search.search("current information")

    def test_langchain_tool_uses_content_and_artifact_response(self) -> None:
        search = LangSearchWebSearch.from_config(search_config())

        tool = search.as_tool()

        self.assertEqual(tool.name, "search_web")
        self.assertEqual(tool.response_format, "content_and_artifact")
        self.assertTrue(tool.handle_tool_error)


if __name__ == "__main__":
    unittest.main()
