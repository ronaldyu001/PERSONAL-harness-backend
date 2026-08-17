"""Root FastAPI entrypoint for the Harness API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from application.use_cases.chat import ChatUseCase
from infrastructure.agent import LangChainAdapter
from infrastructure.memory.Mem0_adapter.Mem0_adapter import Mem0Adapter
from presentation.api.routes import router


def _cors_allow_origins() -> list[str]:
    """Return explicitly configured browser and desktop app origins."""
    return [
        origin.strip()
        for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
        if origin.strip()
    ]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize app-scoped dependencies."""
    # Wire infrastructure adapters into application use cases at the edge.
    memory = Mem0Adapter()
    agent = LangChainAdapter.from_env(memory=memory)
    app.state.chat_use_case = ChatUseCase(agent)

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Harness API",
        version="0.1.0",
        lifespan=lifespan,
    )

    allowed_origins = _cors_allow_origins()
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
