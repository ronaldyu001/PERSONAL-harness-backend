"""Async SQLAlchemy engine and session factory for Maia's database.

These are plain factories that take primitives, so the database package stays
independent of how configuration is resolved. The composition root owns the
engine's lifetime.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from database.models.Maia import Base


def create_engine(
    *,
    dsn: str,
    pool_size: int,
    max_overflow: int,
    pool_timeout_seconds: float,
    echo: bool = False,
) -> AsyncEngine:
    """Create the pooled async engine Maia's adapters share."""
    if not dsn.strip():
        raise ValueError("dsn is required")

    return create_async_engine(
        dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout_seconds,
        # Recover from connections the database dropped while idle.
        pool_pre_ping=True,
        echo=echo,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create the session factory bound to an engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_tables(engine: AsyncEngine) -> None:
    """Create Maia's tables when they do not already exist."""
    async with engine.begin() as connection:
        # ``create_all`` checks first, so this is safe on every startup.
        await connection.run_sync(Base.metadata.create_all)
