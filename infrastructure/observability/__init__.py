"""Trace sinks implementing the application observability port.

Postgres is the wired sink; the JSON Lines one is the fallback used when no
database is configured. Both implement the whole port, so swapping them
changes where traces live and nothing else.
"""

from infrastructure.observability.adapter_jsonl import (
    AdapterJsonlObservability,
)
from infrastructure.observability.adapter_postgres import (
    AdapterPostgresObservability,
)


__all__ = (
    "AdapterJsonlObservability",
    "AdapterPostgresObservability",
)
