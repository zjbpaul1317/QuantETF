from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataLoadRequest:
    """Query options for ETF historical bars."""

    symbols: list[str]
    start_date: str | None = None
    end_date: str | None = None
    columns: list[str] | None = None
    force_reload: bool = False
    filters: dict[str, object] = field(default_factory=dict)
