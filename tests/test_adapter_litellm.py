"""Tests for the LiteLLM model catalog adapter."""

from __future__ import annotations

import unittest

import httpx

from application.models import ListModelsError
from infrastructure.models import AdapterLiteLLM
from infrastructure.settings import GatewayConfig


def gateway_config(*, base_url: str = "http://litellm:4000/v1") -> GatewayConfig:
    return GatewayConfig(
        base_url=base_url,
        api_key="gateway-secret",
        timeout_seconds=5,
        max_retries=0,
    )


class AdapterLiteLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_models_calls_openai_endpoint_and_normalizes_ids(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(
                str(request.url),
                "http://litellm:4000/v1/models",
            )
            self.assertEqual(
                request.headers["authorization"],
                "Bearer gateway-secret",
            )
            return httpx.Response(200, json={
                "object": "list",
                "data": [
                    {"id": " qwen ", "object": "model"},
                    {"id": "llama", "object": "model"},
                    {"id": "qwen", "object": "model"},
                    {"id": "  ", "object": "model"},
                    {"id": 42, "object": "model"},
                    "not-a-model",
                ],
            })

        adapter = AdapterLiteLLM.from_config(
            gateway_config(),
            transport=httpx.MockTransport(handler),
        )

        result = await adapter.list_models()

        self.assertEqual(result.models, ("qwen", "llama"))

    async def test_base_url_without_api_version_still_calls_v1_models(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                str(request.url),
                "http://litellm:4000/v1/models",
            )
            return httpx.Response(200, json={"data": []})

        adapter = AdapterLiteLLM.from_config(
            gateway_config(base_url="http://litellm:4000/"),
            transport=httpx.MockTransport(handler),
        )

        result = await adapter.list_models()

        self.assertEqual(result.models, ())

    async def test_http_failure_raises_application_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="gateway details")

        adapter = AdapterLiteLLM.from_config(
            gateway_config(),
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(ListModelsError):
            await adapter.list_models()

    async def test_malformed_envelope_raises_application_error(self) -> None:
        for payload in ([], {}, {"data": {}}, {"data": "models"}):
            with self.subTest(payload=payload):
                async def handler(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json=payload)

                adapter = AdapterLiteLLM.from_config(
                    gateway_config(),
                    transport=httpx.MockTransport(handler),
                )

                with self.assertRaises(ListModelsError):
                    await adapter.list_models()


if __name__ == "__main__":
    unittest.main()
