"""SQLAlchemy implementation of the application observability port."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from application.observability import (
    ModelContextTrace,
    ModelContextWriteRequest,
    ResponseGateTrace,
    ResponseGateWriteRequest,
    TraceReadRequest,
    TraceReadResult,
    TraceWriteResult,
)
from database.engines.engine_maia import create_session_factory
from database.models.model_maia import ModelContextEvent, ResponseGateEvent


class PostgresObservabilityAdapter:
    """Record and read agent traces in Postgres through SQLAlchemy."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    @classmethod
    def from_engine(cls, engine: AsyncEngine) -> PostgresObservabilityAdapter:
        """Build the adapter from an engine owned by the composition root."""
        return cls(create_session_factory(engine))

    async def record_model_context(
        self,
        request: ModelContextWriteRequest,
    ) -> TraceWriteResult:
        """Record one effective model request and its completion metadata."""
        trace = request.trace
        event_id = trace.event_id or str(uuid4())

        statement = insert(ModelContextEvent).values(
            event_id=event_id,
            occurred_at=trace.occurred_at,
            invocation_id=trace.invocation_id,
            session_id=trace.session_id,
            user_id=trace.user_id,
            model=trace.model,
            mode=trace.mode,
            model_call=trace.model_call,
            system_message=(
                dict(trace.system_message)
                if trace.system_message is not None
                else None
            ),
            messages=[dict(message) for message in trace.messages],
            tools=[dict(tool) for tool in trace.tools],
            status=trace.status,
            usage=dict(trace.usage) if trace.usage is not None else None,
        )

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    # A re-run of the same model call is already recorded.
                    statement.on_conflict_do_nothing(
                        index_elements=["invocation_id", "model_call"]
                    )
                )

        return TraceWriteResult(stream="model-context", event_id=event_id)

    async def record_response_gate(
        self,
        request: ResponseGateWriteRequest,
    ) -> TraceWriteResult:
        """Record one gate evaluation and routing decision."""
        trace = request.trace
        event_id = trace.event_id or str(uuid4())

        statement = insert(ResponseGateEvent).values(
            event_id=event_id,
            occurred_at=trace.occurred_at,
            invocation_id=trace.invocation_id,
            session_id=trace.session_id,
            user_id=trace.user_id,
            model=trace.model,
            mode=trace.mode,
            evaluation_call=trace.evaluation_call,
            repair_attempt=trace.repair_attempt,
            decision=trace.decision,
            passed=trace.passed,
            violations=list(trace.violations),
            feedback=trace.feedback,
            candidate_message_id=trace.candidate_message_id,
            candidate_characters=trace.candidate_characters,
            candidate=trace.candidate,
            available_tools=list(trace.available_tools),
            tools_used=list(trace.tools_used),
            usage=dict(trace.usage) if trace.usage is not None else None,
            error_type=trace.error_type,
            error_message=trace.error_message,
        )

        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    statement.on_conflict_do_nothing(
                        index_elements=["invocation_id", "evaluation_call"]
                    )
                )

        return TraceWriteResult(stream="response-gate", event_id=event_id)

    async def read_traces(self, request: TraceReadRequest) -> TraceReadResult:
        """Read one stream of traces back, most recent first."""
        model = (
            ModelContextEvent
            if request.stream == "model-context"
            else ResponseGateEvent
        )
        # Several rows in one invocation can land on the same timestamp, so the
        # call number is what keeps their order stable between reads.
        sequence = (
            ModelContextEvent.model_call
            if request.stream == "model-context"
            else ResponseGateEvent.evaluation_call
        )

        statement = select(model).where(model.user_id == request.user_id)
        if request.session_id is not None:
            statement = statement.where(model.session_id == request.session_id)
        if request.invocation_id is not None:
            statement = statement.where(
                model.invocation_id == request.invocation_id
            )
        # Filtering before the limit is the point: a window taken first would
        # hide records that belong to the session being inspected.
        statement = statement.order_by(
            model.occurred_at.desc(),
            sequence.desc(),
        ).limit(request.limit)

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()

        to_trace = (
            self._to_model_context
            if request.stream == "model-context"
            else self._to_response_gate
        )
        return TraceReadResult(
            stream=request.stream,
            records=tuple(to_trace(row) for row in rows),
        )

    async def purge_before(self, cutoff: datetime) -> int:
        """Delete traces older than the cutoff, returning how many went."""
        removed = 0
        async with self._session_factory() as session:
            async with session.begin():
                for model in (ModelContextEvent, ResponseGateEvent):
                    result = await session.execute(
                        delete(model).where(model.occurred_at < cutoff)
                    )
                    removed += result.rowcount or 0

        return removed

    @staticmethod
    def _to_model_context(row: ModelContextEvent) -> ModelContextTrace:
        """Map one stored model-context row onto its application schema."""
        return ModelContextTrace(
            invocation_id=row.invocation_id,
            occurred_at=row.occurred_at,
            model=row.model,
            mode=row.mode,
            model_call=row.model_call,
            session_id=row.session_id,
            user_id=row.user_id,
            system_message=row.system_message,
            messages=tuple(row.messages or ()),
            tools=tuple(row.tools or ()),
            status=row.status,
            usage=row.usage,
            event_id=row.event_id,
        )

    @staticmethod
    def _to_response_gate(row: ResponseGateEvent) -> ResponseGateTrace:
        """Map one stored response-gate row onto its application schema."""
        return ResponseGateTrace(
            invocation_id=row.invocation_id,
            occurred_at=row.occurred_at,
            model=row.model,
            mode=row.mode,
            evaluation_call=row.evaluation_call,
            repair_attempt=row.repair_attempt,
            decision=row.decision,
            candidate_characters=row.candidate_characters,
            passed=row.passed,
            session_id=row.session_id,
            user_id=row.user_id,
            violations=tuple(row.violations or ()),
            feedback=row.feedback,
            candidate_message_id=row.candidate_message_id,
            candidate=row.candidate,
            available_tools=tuple(row.available_tools or ()),
            tools_used=tuple(row.tools_used or ()),
            usage=row.usage,
            error_type=row.error_type,
            error_message=row.error_message,
            event_id=row.event_id,
        )
