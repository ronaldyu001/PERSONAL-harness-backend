# Vision

Build an agentic AI platform that provides a modular foundation for orchestrating execution environments, tooling, context, lifecycle and orchestration, observability, verification, and governance (ETCLOVG).

The initial minimum viable product (MVP) focuses on personalized communication, enabling the agent to continuously adapt its communication style and interactions based on the user.

The platform shall be modular and extensible, allowing new capabilities to be added with minimal architectural impact.

---

# Business Requirements

## User Capabilities

- Users can begin new conversations with the agent.
- Users can continue previous conversations.
- Users can create temporary (Incognito) conversations.
- Users can view and manage conversation history.
- Users can configure agent preferences.

## User Experience

- Conversations should feel natural and coherent.
- The agent should adapt its communication style over time based on the user.
- The agent should maintain conversational context throughout an interaction.
- Responses should be relevant, helpful, and consistent.

---

# Functional Requirements

## Conversation Management

- The system shall create, retrieve, update, and delete conversations.
- The system shall retrieve conversations using a unique identifier.
- The system shall support temporary (Incognito) conversations.
- Temporary conversations shall not persist conversation history.
- Temporary conversations shall not create or modify long-term agent memory.

## Context Management

- The system shall maintain conversational context throughout an active session.
- The system shall assemble runtime context prior to response generation.
- The system shall include relevant conversation history when generating responses.
- The system shall support configurable agent instructions.

## Personalization

- The system shall maintain persistent personalization data for each user.
- The system shall retrieve applicable personalization data during response generation.
- The system shall allow personalization data to be created, updated, and deleted.

---

# Non-Functional Requirements

## Extensibility

- The platform shall support the addition of new capabilities with minimal modification to existing components.
- Major platform capabilities shall communicate through well-defined interfaces.

## Maintainability

- Business logic shall remain independent of infrastructure implementations.
- Components shall have clearly defined responsibilities and boundaries.

## Scalability

- The platform shall support a single user.
- The platform shall support interchangeable implementations for major subsystems (for example, inference engines, memory providers, and tool providers).

## Observability

- [ ] System events shall be logged.
- [ ] Errors shall be captured with sufficient diagnostic information for troubleshooting.

## Reliability

- Conversation history and persistent personalization data shall survive application restarts.
- The system shall recover gracefully from recoverable failures whenever possible.

---

# Software Architecture

![1782931130117](image/README/1782931130117.png)

This project follows Clean Architecture with ports and adapters. The application is organized so core assistant behavior is independent of HTTP, databases, LLM gateways, inference runtimes, vector stores, and memory providers.

Dependencies point inward:

```text
presentation -> application -> domain
infrastructure -> application -> domain
```

The inner layers define what the system means and needs. The outer layers decide how those needs are served.

## Layers

### `domain/`

The domain layer contains core entities: concepts the assistant cares about regardless of framework or storage provider.

Current entities:

- `Conversation`: a multi-turn conversation aggregate.
- `ConversationMessage`: one message in a conversation timeline.
- `Memory`: a durable fact, preference, summary, or instruction retained for future context.

Domain entities should not import application, infrastructure, FastAPI, Mem0, Qdrant, LiteLLM, Ollama, or vLLM.

### `application/`

The application layer contains use cases, ports, and application DTOs. It defines what the system needs, but not how those needs are implemented.

Important areas:

- `application/use_cases/`: orchestration logic such as chat.
- `application/llm/`: provider-agnostic LLM port and chat request/response schemas.
- `application/context/`: structured context assembly and rendering contracts.
- `application/conversation/`: port for writing conversation messages.
- `application/memory/`: port for saving and retrieving durable memories.
- `application/memory_handler/`: post-response decision step that decides whether a completed turn should become memory.

Application ports are interfaces owned by the application. Infrastructure adapters implement those interfaces.

### `infrastructure/`

The infrastructure layer implements application ports using concrete tools.

Current examples:

- `infrastructure/llm/LiteLLM_adapter.py`: calls a LiteLLM gateway through the LLM port.
- `infrastructure/llm/Ollama_adapter.py`: calls Ollama through the LLM port.
- `infrastructure/llm/vLLM_adapter.py`: calls vLLM through the LLM port.
- `infrastructure/memory/Mem0_adapter.py`: intended to implement the memory port with Mem0 and Qdrant.
- `infrastructure/context/builders/`: concrete context builder implementations.

Adapters translate between application/domain models and provider-specific APIs.

### `presentation/`

The presentation layer exposes the system to the outside world.

Current examples:

- `presentation/api/routes.py`: FastAPI routes and request/response mapping.
- `main.py`: application entry point that creates the FastAPI app and wires dependencies.

Presentation should call use cases. It should not contain domain logic, memory logic, or provider-specific LLM behavior.

## Ports And Adapters

A port is an application-owned interface:

```text
application/memory/memory_port.py
application/conversation/conversation_port.py
application/llm/llm_port.py
```

An adapter is an infrastructure implementation:

```text
infrastructure/memory/Mem0_adapter.py
infrastructure/llm/LiteLLM_adapter.py
```

The use case depends on the port, not the adapter. That keeps the use case stable when the implementation changes.

```text
ChatUseCase -> LLMPort -> LiteLLMAdapter
ChatUseCase -> ConversationPort -> future database adapter
ChatUseCase -> MemoryPort -> future Mem0Adapter
```

## Current Flow

The current chat path is intentionally simple:

```text
FastAPI route
  -> ChatUseCase
  -> LLMPort
  -> LLM adapter
```

The target production flow is:

```text
FastAPI route
  -> ChatUseCase
  -> ConversationPort.write(user message)
  -> ContextBuilderPort
  -> ContextRendererPort
  -> LLMPort.chat
  -> ConversationPort.write(assistant message)
  -> MemoryHandlerPort.digest
  -> MemoryPort.save
  -> response DTO
```

Each step stays behind a port so the use case can orchestrate behavior without knowing whether the implementation uses SQLite, Postgres, Mem0, Qdrant, LiteLLM, Ollama, or vLLM.

## Context, Conversation, And Memory

These are intentionally separate concepts:

- Conversation history stores the exact chronological messages in a session.
- Context is the structured application object assembled before an LLM call.
- Memory stores extracted durable facts, preferences, summaries, or instructions for future turns.

Conversation history answers: "What exactly was said?"

Memory answers: "What should the assistant remember later?"

Context answers: "What should be shown to the model for this response?"

That separation lets the assistant save full transcripts without treating every message as long-term memory.

## Memory Storage Direction

The domain `Memory` entity is provider-independent. It contains fields the application cares about:

- `content`: remembered information.
- `user_id`: ownership and retrieval scope.
- `kind`: fact, preference, summary, instruction, or other.
- `conversation_id`: provenance.
- `source_message_ids`: traceability back to conversation messages.
- `confidence`: extraction confidence.
- `metadata`: app-level extension point.

The Mem0 adapter should map those fields into Mem0/Qdrant:

```text
Memory.content -> text that gets embedded
Memory.user_id -> Mem0 user scope
Memory.kind / confidence / conversation_id -> metadata filters
```

Mem0 and Qdrant details should not leak into domain or application use cases.

---

# Database

### UserProfile Table

### Conversations Table

### Messages Table
