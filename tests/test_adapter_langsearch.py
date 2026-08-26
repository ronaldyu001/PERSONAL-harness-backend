"""Tests for the LangSearch adapter."""

from __future__ import annotations

import json
import unittest

import httpx

from application.agent.tools import SearchWebError
from infrastructure.agent.tools.adapters import LangSearchAdapter
from infrastructure.settings import load_infrastructure_settings


def search_config():
    settings = load_infrastructure_settings(environ={
        "LITELLM_BASE_URL": "http://litellm:4000",
        "LANGSEARCH_API_KEY": "search-secret",
    })
    return settings.langsearch


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
if __name__ == "__main__":
    unittest.main()
