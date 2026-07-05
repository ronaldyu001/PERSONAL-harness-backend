"""Provider-agnostic context assembled before LLM rendering.

The central context schema stays intentionally small. Component-specific
builders produce typed payloads and wrap them in ContextBlock objects, while
renderers decide how those blocks become provider-agnostic ChatRequest models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, Mapping, Sequence, TypeVar


ContextComponent = Literal[
    "system",
    "conversation",
    "user",
    "memory",
    "retrieval",
    "governance",
]
ContextPriority = Literal["critical", "high", "medium", "low"]
ContextVisibility = Literal["model", "audit_only"]

ContextPayload = TypeVar("ContextPayload")


@dataclass(frozen=True, slots=True)
class ContextSource:
    """Where a context block came from."""

    source_type: str
    source_id: str | None = None
    title: str | None = None
    uri: str | None = None
    score: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextBlock(Generic[ContextPayload]):
    """A typed unit of context produced by one context builder."""

    component: ContextComponent
    context: ContextPayload
    title: str | None = None
    source: ContextSource | None = None
    priority: ContextPriority = "medium"
    visibility: ContextVisibility = "model"
    token_estimate: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_renderable(self) -> bool:
        """Return whether this block may be sent to an LLM renderer."""
        return self.visibility == "model" and self.context is not None


@dataclass(frozen=True, slots=True)
class TokenBudget:
    """Application-level token budget before provider-specific rendering."""

    max_input_tokens: int
    reserved_output_tokens: int
    target_context_tokens: int | None = None
    allocations: Mapping[ContextComponent, int] = field(default_factory=dict)

    @property
    def available_input_tokens(self) -> int:
        """Return input tokens left after reserving response capacity."""
        return max(self.max_input_tokens - self.reserved_output_tokens, 0)

    @property
    def context_limit(self) -> int:
        """Return the effective context limit for trimming/rendering."""
        if self.target_context_tokens is None:
            return self.available_input_tokens
        return min(self.target_context_tokens, self.available_input_tokens)


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Complete structured context for one assistant turn."""

    blocks: Sequence[ContextBlock[Any]] = ()
    token_budget: TokenBudget | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def blocks_for(self, component: ContextComponent) -> tuple[ContextBlock[Any], ...]:
        """Return all blocks produced for a context component."""
        return tuple(block for block in self.blocks if block.component == component)

    def renderable_blocks(self) -> tuple[ContextBlock[Any], ...]:
        """Return all blocks eligible for model context."""
        return tuple(block for block in self.blocks if block.is_renderable())

    def estimated_context_tokens(self) -> int:
        """Return the sum of known token estimates for renderable blocks."""
        return sum(block.token_estimate or 0 for block in self.renderable_blocks())

    def with_block(self, block: ContextBlock[Any]) -> ApplicationContext:
        """Return a copy of this context with one block appended."""
        return replace(self, blocks=(*self.blocks, block))

    def with_blocks(self, blocks: Sequence[ContextBlock[Any]]) -> ApplicationContext:
        """Return a copy of this context with multiple blocks appended."""
        return replace(self, blocks=(*self.blocks, *blocks))

    def with_token_budget(self, token_budget: TokenBudget) -> ApplicationContext:
        """Return a copy of this context with an updated token budget."""
        return replace(self, token_budget=token_budget)

    def audit_attributes(self) -> dict[str, Any]:
        """Return compact metadata useful for logs and traces."""
        return {
            "block_count": len(self.blocks),
            "renderable_blocks": len(self.renderable_blocks()),
            "estimated_context_tokens": self.estimated_context_tokens(),
            "components": tuple(sorted({block.component for block in self.blocks})),
        }
