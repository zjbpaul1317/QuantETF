from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionTarget:
    symbol: str
    target_weight: float
    rank: int | None = None
    score: float | None = None
    selection_reason: str | None = None
