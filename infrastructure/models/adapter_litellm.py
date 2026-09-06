"""LiteLLM implementation of the application model catalog port."""

from __future__ import annotations

import httpx

from application.models import ListModelsError, ModelListResult
from infrastructure.settings import GatewayConfig


class AdapterLiteLLM:
    """Fetch and normalize LiteLLM's OpenAI-compatible model catalog."""

    def __init__(
        self,
        *,
        config: GatewayConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport

    @classmethod
    def from_config(
        cls,
        config: GatewayConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> AdapterLiteLLM:
        """Build the provider adapter from resolved gateway settings."""
        return cls(config=config, transport=transport)

    async def list_models(self) -> ModelListResult:
        """Return unique, non-empty model names in gateway order."""
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.get(
                    self._models_url(self._config.base_url),
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
            return ModelListResult(models=self._parse_models(body))
        except (httpx.HTTPError, TypeError, ValueError) as exc:
            raise ListModelsError("LiteLLM model listing failed.") from exc

    @staticmethod
    def _models_url(base_url: str) -> str:
        """Resolve the OpenAI models resource from either gateway URL form."""
        normalized = base_url.rstrip("/")
        api_root = normalized if normalized.endswith("/v1") else f"{normalized}/v1"
        return f"{api_root}/models"

    @staticmethod
    def _parse_models(body: object) -> tuple[str, ...]:
        """Validate the OpenAI envelope and keep only usable model ids."""
        if not isinstance(body, dict):
            raise TypeError("LiteLLM model response must be an object")

        data = body.get("data")
        if not isinstance(data, list):
            raise TypeError("LiteLLM model response data must be a list")

        models: list[str] = []
        seen: set[str] = set()
        for item in data:
            model_id = item.get("id") if isinstance(item, dict) else None
            name = model_id.strip() if isinstance(model_id, str) else ""
            if name and name not in seen:
                seen.add(name)
                models.append(name)
        return tuple(models)
