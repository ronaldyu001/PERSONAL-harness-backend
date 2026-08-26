"""Tests for Mem0 smart memory inference and configuration."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

# Mem0 initializes its local support directory during import.
os.environ.setdefault(
    "MEM0_DIR", os.path.join(tempfile.gettempdir(), "harness-test-mem0")
)

from application.llm import ChatMessage, ChatResponse
from application.memory import MemorySaveRequest
from infrastructure.memory import Mem0Adapter
from infrastructure.memory.adapter_mem0 import _build_memory_config
from infrastructure.settings import load_infrastructure_settings


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
    def test_adapter_factory_builds_the_sdk_client_from_config(self) -> None:
        infrastructure = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://litellm:4000",
            "MEM0_QDRANT_URL": "http://qdrant:6333",
        })

        with patch(
            "infrastructure.memory.adapter_mem0.Mem0Memory"
        ) as memory_type:
            adapter = Mem0Adapter.from_config(infrastructure.mem0)

        sdk_config = memory_type.call_args.kwargs["config"]
        self.assertIs(adapter.memory, memory_type.return_value)
        self.assertEqual(
            sdk_config.vector_store.config.collection_name,
            infrastructure.mem0.collection_name,
        )

    def test_configuration_uses_ollama_for_inference_and_embeddings(self) -> None:
        infrastructure = load_infrastructure_settings(environ={
            "LITELLM_BASE_URL": "http://litellm:4000",
            "LITELLM_API_KEY": "test-key",
            "OLLAMA_BASE_URL": "http://ollama:11434",
            "MEM0_QDRANT_URL": "http://qdrant:6333",
        })
        settings = infrastructure.mem0.model_copy(update={
            "custom_instructions": "Remember durable preferences.",
        })

        config = _build_memory_config(settings)

        self.assertEqual(config.llm.provider, "openai")
        self.assertEqual(config.llm.config["model"], "llama")
        self.assertEqual(
            config.llm.config["openai_base_url"],
            "http://litellm:4000/v1",
        )
        self.assertEqual(config.llm.config["api_key"], "test-key")
        self.assertEqual(config.llm.config["max_tokens"], 512)
        self.assertEqual(config.embedder.provider, "ollama")
        self.assertEqual(config.embedder.config["embedding_dims"], 768)
        self.assertEqual(config.vector_store.provider, "qdrant")
        self.assertEqual(config.custom_instructions, "Remember durable preferences.")


if __name__ == "__main__":
    unittest.main()
