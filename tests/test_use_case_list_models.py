"""Tests for listing models through the application boundary."""

from __future__ import annotations

import unittest

from application.models import ModelListResult
from application.use_cases import UseCaseListModels


class RecordingModels:
    """PortListModels double that records catalog reads."""

    def __init__(self, result: ModelListResult) -> None:
        self.result = result
        self.calls = 0

    async def list_models(self) -> ModelListResult:
        self.calls += 1
        return self.result


class UseCaseListModelsTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_returns_the_port_result(self) -> None:
        expected = ModelListResult(models=("qwen", "llama"))
        models = RecordingModels(expected)

        result = await UseCaseListModels(models).execute()

        self.assertIs(result, expected)
        self.assertEqual(models.calls, 1)


if __name__ == "__main__":
    unittest.main()
