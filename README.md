# Harness

Harness is a FastAPI backend for an AI assistant. It keeps the application core separate from concrete providers such as LiteLLM, Ollama, Mem0, and Qdrant.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The API starts on `http://127.0.0.1:8000`.

### Local context logs

Set `AGENT_CONTEXT_LOGGING` to `structure` or `full` to write the effective
input for every model call to `.logs/agent-context.jsonl`. `structure` records
message types, IDs, sizes, and provider-reported token usage; `full` also
records model-visible content and tool arguments. Tauri development enables
`full` through its Compose override
and bind-mounts the backend `.logs` directory into the container. These logs can
contain conversations and retrieved memories, so keep logging disabled outside
local development.

## API

- `GET /api/health`
- `POST /api/chat`

`POST /api/chat` requires a `user_id` and accepts an optional `session_id`. The
response always returns the session ID; send it with later messages to continue
the same conversation. In a production deployment, derive `user_id` from the
authenticated principal instead of trusting a request-body value.

## Architecture

Harness follows a ports-and-adapters structure:

```text
presentation -> application -> domain
infrastructure -> application -> domain
```

Dependencies should point inward. The core application should not depend on FastAPI, LiteLLM, Ollama, Mem0, Qdrant, databases, or other provider details.

### Enterprise Development Architecture

The development environment uses Docker Compose to keep model infrastructure and
stateful services reproducible while the backend preserves ports-and-adapters
boundaries. Mem0 runs as an SDK inside the backend rather than as a separate
service.

```mermaid
flowchart LR
    Developer["Developer or API client"]

    subgraph Compose["Docker Compose development environment"]
        subgraph Backend["Harness backend · FastAPI · :8000"]
            Routes["Presentation<br/>FastAPI routes"]

            subgraph Application["Application layer"]
                Chat["ChatUseCase"]
                AgentPort["AgentPort"]
                MemoryPort["MemoryPort"]
            end

            Domain["Domain entities<br/>Conversation · ConversationMessage · Memory"]

            subgraph Infrastructure["Infrastructure adapters"]
                AgentAdapter["Agent<br/>LangChainAdapter"]
                MemoryMiddleware["Agent middleware<br/>MemoryMiddleware"]
                Mem0Adapter["Memory<br/>Mem0Adapter + Mem0 SDK"]
                History[("Mem0 history<br/>SQLite · development only")]
            end
        end

        LiteLLM["LiteLLM proxy<br/>OpenAI-compatible API · :4000"]
        Ollama["Ollama runtime<br/>LLM and embeddings · :11434"]
        Qdrant["Qdrant<br/>REST :6333 · gRPC :6334"]
        Mem0Volume[("Mem0 history volume")]
        OllamaVolume[("Ollama model volume")]
        QdrantVolume[("Qdrant data volume")]
    end

    Developer -->|"HTTP"| Routes
    Routes --> Chat
    Chat -->|"depends on"| AgentPort
    AgentAdapter -. "implements" .-> AgentPort
    AgentAdapter -->|"OpenAI-compatible HTTP"| LiteLLM
    AgentAdapter -->|"registers"| MemoryMiddleware
    MemoryMiddleware -->|"retrieves through"| MemoryPort
    MemoryMiddleware -->|"submits completed turns through"| MemoryPort
    LiteLLM -->|"model inference"| Ollama

    Chat --> Domain
    Mem0Adapter -. "implements" .-> MemoryPort
    Mem0Adapter -->|"memory inference"| LiteLLM
    Mem0Adapter -->|"embeddings"| Ollama
    Mem0Adapter -->|"vector storage and search"| Qdrant
    Mem0Adapter --> History
    History --- Mem0Volume

    Ollama --- OllamaVolume
    Qdrant --- QdrantVolume

    classDef active fill:#dbeafe,stroke:#2563eb,color:#172554;
    classDef planned fill:#fef3c7,stroke:#d97706,color:#451a03;
    classDef stateful fill:#dcfce7,stroke:#16a34a,color:#052e16;
    class Routes,Chat,AgentPort,AgentAdapter,MemoryMiddleware,MemoryPort,Mem0Adapter,LiteLLM,Ollama active;
    class History planned;
    class Qdrant,Mem0Volume,OllamaVolume,QdrantVolume stateful;
```

Solid arrows show the chat and durable-memory lifecycle. Before model calls,
`MemoryMiddleware` retrieves relevant memories. After each completed turn it
calls `MemoryPort.save`; `Mem0Adapter` submits the user/assistant pair with
`infer=True`, allowing Mem0 to extract zero or more durable memories. Mem0 sends
the model configured by `MEM0_LLM_MODEL` through LiteLLM for extraction, uses
Ollama for embeddings, and uses Qdrant for storage and search. The backend
reaches Compose services by their service names, such as
`http://litellm:4000`, `http://ollama:11434`, and `http://qdrant:6333`.

### Layer Overview

| Layer          | Path                              | Purpose                                                         |
| -------------- | --------------------------------- | --------------------------------------------------------------- |
| Domain         | `domain/`                       | Defines core entities and business rules.                       |
| Application    | `application/`                  | Defines use cases, ports, and provider-neutral schemas.         |
| Infrastructure | `infrastructure/`               | Implements application ports with concrete tools and services.  |
| Presentation   | `presentation/` and `main.py` | Exposes the application through FastAPI and wires dependencies. |

### Domain

| Area                 | Purpose                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `domain/entities/` | Core entities such as`Conversation`, `ConversationMessage`, and `Memory`.                 |
| Business rules       | Validation and behavior that should be true regardless of storage, API, or model provider.      |
| Dependency rule      | Should not import application, infrastructure, FastAPI, Mem0, Qdrant, LiteLLM, Ollama, or vLLM. |

### Application

| Area                            | Purpose                                                                           |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `application/use_cases/`      | Orchestrates workflows such as chat.                                              |
| `application/llm/`            | Defines the provider-neutral LLM port and chat schemas.                           |
| `application/conversation/`   | Defines the conversation persistence port.                                        |
| `application/memory/`         | Defines the durable memory port and schemas.                                      |
| `application/agent/`          | Defines the conversational agent port.                                             |
| Dependency rule                 | May depend on domain, but should not depend on concrete infrastructure providers. |

### Infrastructure

| Area                                    | Purpose                                                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------- |
| `infrastructure/gateway/`             | Gateway adapters such as the OpenAI-compatible LiteLLM adapter.                             |
| `infrastructure/agent/`               | Implements `AgentPort` with LangChain, LangGraph, and the LiteLLM gateway.                  |
| `infrastructure/agent/middleware/`    | Retrieves memories before model calls and submits completed turns after agent runs.         |
| `infrastructure/llm/`                 | Direct model-runtime adapters such as Ollama and vLLM.                                      |
| `infrastructure/memory/Mem0_adapter/` | Memory adapter and Mem0 configuration.                                                      |
| Dependency rule                         | Implements application ports and translates provider-specific APIs into application models. |

### Presentation

| Area                           | Purpose                                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `presentation/api/routes.py` | FastAPI route definitions and request/response mapping.                                                   |
| `presentation/api/schemas/`  | API-facing schemas, when route schemas are split out.                                                     |
| `main.py`                    | FastAPI app creation and dependency wiring.                                                               |
| Dependency rule                | Calls application use cases; should not contain domain logic or provider-specific implementation details. |

## Current Chat Flow

```text
FastAPI route
  -> ChatUseCase
  -> AgentPort
  -> LangChainAdapter
     -> MemoryMiddleware
        -> MemoryPort.save(completed turn)
        -> MemoryPort.retrieve(query)
        -> Mem0Adapter(infer=True)
  -> LiteLLM
```
