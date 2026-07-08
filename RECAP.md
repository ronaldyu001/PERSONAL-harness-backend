# Recap

This project is a local-first AI assistant backend built with Python, FastAPI, and Clean Architecture. The current direction is to keep the application layer independent from specific LLM providers, memory providers, vector stores, and API frameworks.

This file is a checkpoint for where the architecture discussion left off and what was being thought through next.

## Where Things Stand

The architecture now separates core concepts from application ports and infrastructure adapters:

- `domain/entities/conversation.py` defines `Conversation` and `ConversationMessage`.
- `domain/entities/memory.py` defines `Memory`.
- `application/conversation/` defines the conversation persistence port and request/result schemas.
- `application/memory/` defines the memory storage port and request/result schemas.
- `application/memory_handler/` defines the post-response memory digestion port.
- `infrastructure/llm/` contains LLM adapters such as LiteLLM, Ollama, and vLLM.

The important design change was moving `ConversationMessage` out of the application layer and into the domain. A `Conversation` aggregate was added alongside it. Memory was also defined as a domain entity so the app can reason about memory without coupling itself to Mem0 or Qdrant.

## Thought Process So Far

The main architectural question was where stateful assistant concepts belong.

The conclusion was:

- `Conversation` and `ConversationMessage` are domain concepts because the assistant fundamentally has conversations, regardless of whether messages are stored in SQLite, Postgres, or another database.
- `Memory` is also a domain concept because personalization depends on retained knowledge about the user. Mem0 and Qdrant are only one way to store and retrieve it.
- `Context` is not a domain entity right now. It exists because the app needs to prepare information for an LLM call, so it belongs in the application layer.
- LLM providers and runtimes are infrastructure. LiteLLM, Ollama, and vLLM should be swappable without changing use-case logic.

Another decision was to avoid a generic database port. Instead, each application capability owns the port it needs:

```text
ConversationPort -> exact conversation timeline persistence
MemoryPort -> durable semantic memory save/retrieve
LLMPort -> provider-agnostic chat completion
```

This keeps the application language aligned with assistant behavior instead of storage mechanics.

## Memory Direction

Memory is now modeled as an application/domain concept first:

```text
Memory.content
  -> the retained fact, preference, summary, or instruction

Memory.user_id / kind / confidence / conversation_id / source_message_ids
  -> metadata used for scoping, filtering, provenance, and lifecycle
```

Mem0 and Qdrant should remain infrastructure details. The future `Mem0Adapter` should implement `MemoryPort` by mapping the domain `Memory` into Mem0:

```text
Memory.content -> Mem0 memory text, embedded for semantic search
Memory.user_id -> Mem0 top-level user scope
Memory fields -> Mem0 metadata / Qdrant payload filters
```

Filterable fields should stay flat in Mem0 metadata, such as `kind`, `conversation_id`, `confidence`, and `source_message_ids`.

The current mental model for Qdrant is:

```text
Qdrant point id -> memory id
Qdrant vector -> embedding of Memory.content
Qdrant payload -> flat metadata fields from Memory
```

So the domain class does not need vector fields. It needs the fields the assistant cares about and the adapter will translate those into Mem0/Qdrant storage.

## Open Design Thread

The next implementation question is how much of Mem0 should be wrapped.

The likely answer is to keep the adapter thin but intentional:

- It should accept `MemorySaveRequest` and `MemoryRetrieveRequest`.
- It should call Mem0 with the right `user_id`, memory text, and metadata filters.
- It should normalize Mem0 result shapes back into application/domain objects.
- It should hide Mem0-specific response formats, filter syntax, Qdrant payload details, and any client setup.

The use case should not call Mem0 directly.

## Likely Next Steps

1. Implement `infrastructure/memory/Mem0_adapter.py` against `MemoryPort`.
2. Map `MemorySaveRequest.memory` into `mem0.add(...)`.
3. Map `MemoryRetrieveRequest` into `mem0.search(...)` with filters for `user_id`, `kinds`, `conversation_id`, and `min_score`.
4. Convert Mem0 results back into `RetrievedMemory` and `MemoryRetrieveResult`.
5. Add a simple memory context builder that retrieves relevant memories and outputs context blocks.
6. Update `ChatUseCase` so the full flow can become:

```text
save user message
build context
render LLM request
call LLM
save assistant message
digest completed turn for memory
save selected memories
return response
```

The immediate implementation focus is Mem0. The key rule is that Mem0/Qdrant details should stay inside the adapter; the use case should only depend on `MemoryPort`.

## Useful Implementation Notes

When implementing `Mem0Adapter.save`, the mapping should probably look like:

```text
request.memory.content -> mem0.add text
request.memory.user_id -> mem0 user_id
request.memory.kind -> metadata["kind"]
request.memory.conversation_id -> metadata["conversation_id"]
request.memory.confidence -> metadata["confidence"]
request.memory.source_message_ids -> metadata["source_message_ids"]
```

When implementing `Mem0Adapter.retrieve`, the mapping should probably look like:

```text
request.query -> mem0.search query
request.user_id -> required search filter/scope
request.kinds -> metadata kind filter
request.conversation_id -> metadata conversation filter
request.min_score -> post-filter or provider filter if supported cleanly
request.limit -> search limit
```

One important constraint from the Mem0/Qdrant discussion: keep filterable fields top-level in Mem0 metadata. Do not bury them under a nested `payload` or `memory` object unless there is a clear reason.
