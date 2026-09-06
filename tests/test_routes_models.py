"""Tests for the model catalog HTTP route."""

from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.models import ListModelsError, ModelListResult
from application.use_cases import UseCaseListModels
from presentation.api import router


class StubModels:
    """PortListModels double with a configurable result or failure."""

    def __init__(
        self,
        *,
        result: ModelListResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or ModelListResult()
        self.error = error

    async def list_models(self) -> ModelListResult:
        if self.error is not None:
            raise self.error
        return self.result


def client_for(models: StubModels) -> TestClient:
    app = FastAPI()
    app.state.use_case_list_models = UseCaseListModels(models)
    app.include_router(router)
    return TestClient(app)


class ModelRoutesTests(unittest.TestCase):
    def test_list_models_returns_the_frontend_contract(self) -> None:
        models = StubModels(
            result=ModelListResult(models=("qwen", "llama")),
        )

        with client_for(models) as client:
            response = client.get("/api/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": ["qwen", "llama"]})

    def test_provider_failure_maps_to_safe_bad_gateway_response(self) -> None:
        models = StubModels(
            error=ListModelsError("sensitive provider response"),
        )

        with client_for(models) as client:
            response = client.get("/api/models")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "Unable to retrieve available models."},
        )
        self.assertNotIn("sensitive", response.text)


if __name__ == "__main__":
    unittest.main()
