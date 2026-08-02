# Recap

This project is a local-first AI assistant backend built with Python, FastAPI, and Clean Architecture. The current direction is to keep the application layer independent from specific LLM providers, memory providers, vector stores, and API frameworks.

This file is a checkpoint for where the architecture discussion left off and what was being thought through next.

## Where Things Stand

The architecture now separates core concepts from application ports and infrastructure adapters:

- `domain/entities/conversation.py` defines `Conversation` and `ConversationMessage`.
- `domain/entities/memory.py` defines `Memory`.
- `application/conversation/` defines the conversation persistence port and request/result schemas.
- `application/memory/` defines smart memory save/retrieve operations and provider-neutral schemas.
- `application/agent/` defines the conversational agent port.
- `infrastructure/agent/` implements that port with `LangChainAdapter`.
- `infrastructure/llm/` contains LLM adapters such as LiteLLM, Ollama, and vLLM.

The important design change was moving `ConversationMessage` out of the application layer and into the domain. A `Conversation` aggregate was added alongside it. Memory was also defined as a domain entity so the app can reason about memory without coupling itself to Mem0 or Qdrant.

## Thought Process So Far

The main architectural question was where stateful assistant concepts belong.

The conclusion was:

- `Conversation` and `ConversationMessage` are domain concepts because the assistant fundamentally has conversations, regardless of whether messages are stored in SQLite, Postgres, or another database.
- `Memory` is also a domain concept because personalization depends on retained knowledge about the user. Mem0 and Qdrant are only one way to store and retrieve it.
- Conversation message state, checkpointing, and context summarization are handled by LangGraph and LangChain inside the agent adapter.
- LLM providers and runtimes are infrastructure. LiteLLM, Ollama, and vLLM should be swappable without changing use-case logic.

Another decision was to avoid a generic database port. Instead, each application capability owns the port it needs:

```text
ConversationPort -> exact conversation timeline persistence
MemoryPort -> durable semantic memory save/retrieve
LLMPort -> provider-agnostic chat completion
AgentPort -> conversational agent execution
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

Mem0 and Qdrant remain infrastructure details. `Mem0Adapter` implements
`MemoryPort` by mapping a completed user/assistant turn into Mem0 and mapping
Mem0 extraction/search results back into domain `Memory` entities:

```text
MemorySaveRequest user + assistant turn -> Mem0 add messages
MemorySaveRequest.user_id -> Mem0 user_id
MemorySaveRequest.conversation_id -> Mem0 run_id
Mem0 inferred facts -> domain Memory entities
```

Filterable fields should stay flat in Mem0 metadata, such as `kind`, `conversation_id`, `confidence`, and `source_message_ids`.

The current mental model for Qdrant is:

```text
Qdrant point id -> memory id
Qdrant vector -> embedding of Memory.content
Qdrant payload -> flat metadata fields from Memory
```

So the domain class does not need vector fields. It needs the fields the assistant cares about and the adapter will translate those into Mem0/Qdrant storage.

## Current Memory Behavior

The adapter is thin but intentional:

- `MemoryPort.save` accepts a completed turn and calls Mem0 with `infer=True`.
- Mem0's configured Ollama LLM decides whether the turn contains durable facts.
- `MemoryPort.retrieve` searches by user scope and relevance.
- The adapter normalizes Mem0 results into application/domain objects.
- Mem0 response formats, filters, and provider setup stay in infrastructure.

The use case should not call Mem0 directly.

## Likely Next Steps

1. Validate Mem0 retrieval against the running Ollama and Qdrant services.
2. Tune and regression-test Mem0 custom extraction instructions with real turns.
3. Persist checkpoints beyond one backend process when durable sessions are needed.
4. Extend the chat flow with durable message persistence when required:

```text
invoke agent
submit completed user/assistant turn to Mem0
Mem0 infers and saves selected memories
return response
```

The immediate implementation focus is Mem0. The key rule is that Mem0/Qdrant details should stay inside the adapter; the use case should only depend on `MemoryPort`.

## Useful Implementation Notes

`Mem0Adapter.save` maps the smart-save request as follows:

```text
request.user_message + assistant_response -> mem0.add messages
request.user_id -> mem0 user_id
request.conversation_id -> mem0 run_id
request.metadata -> flat Mem0 metadata
infer -> true
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
