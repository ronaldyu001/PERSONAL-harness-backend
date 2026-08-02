"""Tests for Mem0 smart memory inference and configuration."""

from __future__ import annotations

import os
import tempfile
import unittest

# Mem0 initializes its local support directory during import.
os.environ.setdefault(
    "MEM0_DIR", os.path.join(tempfile.gettempdir(), "harness-test-mem0")
)

from application.llm.schemas import ChatMessage, ChatResponse
from application.memory.schemas import MemorySaveRequest
from infrastructure.memory.Mem0_adapter import Mem0Adapter
from infrastructure.memory.Mem0_adapter.Mem0_config import (
    Mem0Settings,
    build_memory_config,
)


class RecordingMem0Client:
    """Synchronous Mem0 SDK test double used through asyncio.to_thread."""

    def __init__(self) -> None:
        self.messages = None
        self.options = None

    def add(self, messages, **options):
        self.messages = messages
        self.options = options
        return {
            "results": [
                {
                    "id": "memory-1",
                    "memory": "User prefers concise answers",
                    "event": "ADD",
                },
                {
                    "id": "memory-2",
                    "memory": "User is learning LangGraph",
                    "event": "ADD",
                },
            ]
        }


class Mem0AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_infers_memories_from_completed_turn(self) -> None:
        client = RecordingMem0Client()
        adapter = Mem0Adapter(memory=client)

        result = await adapter.save(
            MemorySaveRequest(
                user_message=ChatMessage(
                    role="user",
                    content="Remember that I prefer concise answers.",
                ),
                assistant_response=ChatResponse(
                    content="I will keep that in mind.",
                ),
                user_id="user-1",
                conversation_id="session-1",
                metadata={"source": "chat"},
            )
        )

        self.assertEqual(
            client.messages,
            [
                {
                    "role": "user",
                    "content": "Remember that I prefer concise answers.",
                },
                {
                    "role": "assistant",
                    "content": "I will keep that in mind.",
                },
            ],
        )
        self.assertIsNotNone(client.options)
        assert client.options is not None
        self.assertTrue(client.options["infer"])
        self.assertEqual(client.options["user_id"], "user-1")
        self.assertEqual(client.options["run_id"], "session-1")
        self.assertEqual(
            client.options["metadata"],
            {"source": "chat", "conversation_id": "session-1"},
        )
        self.assertEqual(result.saved_count, 2)
        self.assertEqual(result.memories[0].memory_id, "memory-1")
        self.assertEqual(result.memories[0].user_id, "user-1")


class Mem0ConfigurationTests(unittest.TestCase):
    def test_configuration_uses_ollama_for_inference_and_embeddings(self) -> None:
        settings = Mem0Settings(
            qdrant_url="http://qdrant:6333",
            qdrant_host=None,
            qdrant_port=6333,
            qdrant_api_key=None,
            collection_name="harness_memories",
            ollama_base_url="http://ollama:11434",
            llm_model="qwen3:4b",
            embedder_model="nomic-embed-text",
            embedding_dims=768,
            history_db_path="/tmp/mem0/history.db",
            local_qdrant_path="/tmp/mem0/qdrant",
            custom_instructions="Remember durable preferences.",
        )

        config = build_memory_config(settings)

        self.assertEqual(config.llm.provider, "ollama")
        self.assertEqual(config.llm.config["model"], "qwen3:4b")
        self.assertEqual(
            config.llm.config["ollama_base_url"], "http://ollama:11434"
        )
        self.assertEqual(config.embedder.provider, "ollama")
        self.assertEqual(config.embedder.config["embedding_dims"], 768)
        self.assertEqual(config.vector_store.provider, "qdrant")
        self.assertEqual(config.custom_instructions, "Remember durable preferences.")


if __name__ == "__main__":
    unittest.main()
