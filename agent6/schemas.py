"""
schemas.py — Pydantic v2 contracts for every boundary in the Session 6 agent.

Every inter-layer hand-off between Memory, Perception, Decision and Action is
typed by one of the models below. No free-form dicts cross a layer boundary,
no regex is applied to LLM output.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ──────────────────────────────────────────────────────────────────────────
# MEMORY
# ──────────────────────────────────────────────────────────────────────────

MemoryKind = Literal["fact", "preference", "tool_outcome", "artifact"]


class MemoryItem(BaseModel):
    """One row in the durable memory store (state/memory.json)."""
    id: str
    kind: MemoryKind
    text: str                                       # human-readable summary
    keywords: list[str] = Field(default_factory=list)
    value: dict[str, Any] = Field(default_factory=dict)   # structured payload
    run_id: str
    created_at: float
    artifact_id: Optional[str] = None               # set when item references an artifact

    model_config = ConfigDict(extra="allow")


class MemoryClassification(BaseModel):
    """Structured-output schema for the memory classifier LLM call."""
    is_memorable: bool
    kind: MemoryKind = "fact"
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    value: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# PERCEPTION
# ──────────────────────────────────────────────────────────────────────────

GoalStatus = Literal["open", "done"]


class Goal(BaseModel):
    """One decomposed step that the loop tracks across iterations."""
    id: str                                         # stable id like "g1", "g2"
    description: str
    status: GoalStatus = "open"
    # When set, the orchestrator loads the artifact's raw bytes and pastes them
    # into Decision's prompt under ATTACHED ARTIFACTS for this goal.
    attach_artifact_id: Optional[str] = None
    # One-word tag from a closed set: lookup / search / fetch / extraction /
    # synthesis / comparison / tool_action / memory_recall. Set by Perception.
    reasoning_type: Optional[str] = None


class PerceptionOutput(BaseModel):
    """Structured-output schema returned by Perception each iteration."""
    goals: list[Goal] = Field(default_factory=list)
    # The model's own running summary of what it has done and what is left.
    notes: str = ""

    @property
    def all_done(self) -> bool:
        return bool(self.goals) and all(g.status == "done" for g in self.goals)

    def next_unfinished(self) -> Optional[Goal]:
        for g in self.goals:
            if g.status == "open":
                return g
        return None


# ──────────────────────────────────────────────────────────────────────────
# DECISION
# ──────────────────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """A single typed MCP tool call selected by Decision."""
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Decision's typed return: either an answer or exactly one tool call."""
    answer: Optional[str] = None
    tool_call: Optional[ToolCall] = None

    @property
    def is_answer(self) -> bool:
        return self.tool_call is None and (self.answer is not None)


# ──────────────────────────────────────────────────────────────────────────
# HISTORY EVENTS
# ──────────────────────────────────────────────────────────────────────────

class HistoryEvent(BaseModel):
    """One row in the per-run history list."""
    iter: int
    kind: Literal["action", "answer"]
    goal_id: str
    text: Optional[str] = None
    tool: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None
    result_descriptor: Optional[str] = None
    artifact_id: Optional[str] = None
