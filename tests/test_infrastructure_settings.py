"""Tests for centralized infrastructure configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from infrastructure.gateway.LiteLLM_adapter import LiteLLMAdapter
from infrastructure.settings import (
    DEFAULT_CONFIG_PATH,
    load_infrastructure_settings,
)


class InfrastructureSettingsTests(unittest.TestCase):
    def test_checked_in_config_is_valid(self) -> None:
        settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://gateway:4000",
        })

        self.assertEqual(settings.agent.summarization.trigger_tokens, 5000)
        self.assertEqual(settings.mem0.embedder.dimensions, 768)
        self.assertEqual(settings.langsearch.result_count, 5)
        self.assertIsNone(settings.langsearch.api_key)
        self.assertTrue(settings.agent.response_gate.enabled)

    def test_environment_overrides_only_runtime_values(self) -> None:
        settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://gateway:4000",
            "LITELLM_API_KEY": "secret",
            "MEM0_EMBEDDER_MODEL": "alternate-embedder",
            "CORS_ALLOW_ORIGINS": "http://one.test, http://two.test",
            "AGENT_CONTEXT_LOGGING": "structure",
            "LANGSEARCH_API_KEY": "search-secret",
        })

        self.assertEqual(settings.gateway.api_key, "secret")
        self.assertEqual(settings.mem0.embedder.model, "alternate-embedder")
        self.assertEqual(
            settings.api.cors_allow_origins,
            ("http://one.test", "http://two.test"),
        )
        self.assertEqual(settings.logging.context_mode, "structure")
        self.assertEqual(settings.logging.response_gate_mode, "structure")
        assert settings.langsearch.api_key is not None
        self.assertEqual(
            settings.langsearch.api_key.get_secret_value(),
            "search-secret",
        )
        self.assertNotIn("search-secret", repr(settings))
        self.assertEqual(
            settings.gateway.timeout_seconds,
            60,
        )

    def test_gateway_factory_uses_resolved_gateway_config(self) -> None:
        settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://gateway:4000",
            "LITELLM_API_KEY": "secret",
        })

        with patch(
            "infrastructure.gateway.LiteLLM_adapter.AsyncOpenAI"
        ) as client_type:
            LiteLLMAdapter.from_config(settings.gateway)

        client_type.assert_called_once_with(
            base_url="http://gateway:4000/v1",
            api_key="secret",
            timeout=settings.gateway.timeout_seconds,
            max_retries=settings.gateway.max_retries,
        )

    def test_mem0_paths_are_derived_from_mem0_dir(self) -> None:
        settings = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://gateway:4000",
            "MEM0_DIR": "/var/lib/maia",
        })

        self.assertEqual(
            settings.mem0.history_db_path,
            Path("/var/lib/maia/history.db"),
        )
        self.assertEqual(
            settings.mem0.vector_store.local_path,
            Path("/var/lib/maia/qdrant"),
        )

    def test_missing_gateway_url_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "LITELLM_BASE_URL"):
            load_infrastructure_settings(environ={})

    def test_unknown_yaml_key_is_rejected(self) -> None:
        source = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                source.replace(
                    "  max_retries: 2",
                    "  max_retries: 2\n  max_retriez: 3",
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                load_infrastructure_settings(
                    environ={
                        "LITELLM_BASE_URL": "http://gateway:4000",
                    },
                    config_path=config_path,
                )


if __name__ == "__main__":
    unittest.main()
