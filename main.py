"""Root FastAPI entrypoint for the Harness API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from application.use_cases.chat import ChatUseCase
from infrastructure.agent import LangChainAdapter
from infrastructure.memory.Mem0_adapter.Mem0_adapter import Mem0Adapter
from presentation.api.routes import router


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

    # Register API route modules.
    app.include_router(router)

    return app


# ASGI servers can target this object with: uvicorn main:app
app = create_app()


if __name__ == "__main__":
    # Local development runner; production should use an ASGI server command.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
