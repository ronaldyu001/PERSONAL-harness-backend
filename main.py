"""Root FastAPI entrypoint for the Harness API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine

from application.observability import PortObservability
from application.use_cases import (
    UseCaseChat,
    UseCaseListModels,
    UseCaseReadConversationHistory,
    UseCaseReadTraces,
)
from database.engines.engine_maia import create_engine, create_tables
from infrastructure.agent import AdapterLangChain
from infrastructure.conversation import AdapterPostgresConversation
from infrastructure.memory import AdapterMem0
from infrastructure.models import AdapterLiteLLM
from infrastructure.observability import (
    AdapterJsonlObservability,
    AdapterPostgresObservability,
)
from infrastructure.settings import (
    InfrastructureSettings,
    load_infrastructure_settings,
)
from presentation.api import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app-scoped dependencies."""
    # Wire infrastructure adapters into application use cases at the edge.
    settings: InfrastructureSettings = app.state.infrastructure_settings
    memory = AdapterMem0.from_config(settings.mem0)
    models = AdapterLiteLLM.from_config(settings.gateway)

    # Without a DSN the API runs exactly as before, minus the transcript.
    engine: AsyncEngine | None = None
    conversations: AdapterPostgresConversation | None = None
    if settings.postgres.dsn is not None:
        engine = create_engine(
            dsn=settings.postgres.dsn.get_secret_value(),
            pool_size=settings.postgres.pool_size,
            max_overflow=settings.postgres.max_overflow,
            pool_timeout_seconds=settings.postgres.pool_timeout_seconds,
            echo=settings.postgres.echo,
        )
        # A configured database that cannot be prepared is a startup failure,
        # not a transcript that silently goes missing.
        await create_tables(engine)
        conversations = AdapterPostgresConversation.from_engine(engine)

    # Traces go to the database when there is one, and to files otherwise,
    # so the agent is never running unobserved.
    observability: PortObservability
    if settings.observability.sink == "database" and engine is None:
        raise ValueError(
            "observability sink database requires POSTGRES_DSN"
        )
    if engine is not None and settings.observability.sink != "file":
        postgres_traces = AdapterPostgresObservability.from_engine(engine)
        if settings.observability.retention_days:
            # No scheduler here, and this process restarts often enough.
            await postgres_traces.purge_before(
                datetime.now(UTC)
                - timedelta(days=settings.observability.retention_days)
            )
        observability = postgres_traces
    else:
        observability = AdapterJsonlObservability.from_config(settings.logging)

    agent = AdapterLangChain.from_config(
        settings,
        memory=memory,
        conversations=conversations,
        observability=observability,
    )
    app.state.use_case_chat = UseCaseChat(agent)
    app.state.use_case_list_models = UseCaseListModels(models)
    app.state.use_case_read_conversation_history = (
        UseCaseReadConversationHistory(
            conversations,
            default_list_size=settings.conversation.default_list_size,
            max_list_size=settings.conversation.max_list_size,
        )
        if conversations is not None
        else None
    )
    app.state.use_case_read_traces = UseCaseReadTraces(
        observability,
        default_page_size=settings.observability.default_page_size,
        max_page_size=settings.observability.max_page_size,
    )

    try:
        yield
    finally:
        if engine is not None:
            await engine.dispose()


def create_app(
    settings: InfrastructureSettings | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or load_infrastructure_settings()
    app = FastAPI(
        title="Harness API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.infrastructure_settings = settings

    allowed_origins = list(settings.api.cors_allow_origins)
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    # Register API route modules.
    app.include_router(router)

    return app


# ASGI servers can target this object with: uvicorn main:app
app = create_app()


if __name__ == "__main__":
    # Local development runner; production should use an ASGI server command.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
