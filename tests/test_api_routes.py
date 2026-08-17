"""Tests for API error mapping."""

from __future__ import annotations

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from application.llm.schemas import ChatResponse
from application.use_cases.chat import ChatUseCase
from presentation.api.routes import router


class EmptyAgent:
    """Agent test double that returns no user-visible content."""

    async def chat(self, request, *, session_id: str, user_id: str) -> ChatResponse:
        return ChatResponse(content="")


class ChatRoutesTests(unittest.TestCase):
    def test_empty_agent_response_maps_to_bad_gateway(self) -> None:
        app = FastAPI()
        app.state.chat_use_case = ChatUseCase(EmptyAgent())
        app.include_router(router)

        with TestClient(app) as client:
            response = client.post(
                "/api/chat",
                json={
                    "message": "Hello",
                    "model": "qwen",
                    "user_id": "user-1",
                },
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "The agent returned an empty response."},
        )


if __name__ == "__main__":
    unittest.main()
