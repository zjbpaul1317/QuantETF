from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSignalFilter(ABC):
    """Base interface for portfolio selection filters."""

    @abstractmethod
    def apply(self, signal_snapshot: pd.DataFrame, current_holdings: pd.DataFrame | None = None) -> pd.DataFrame:
        """Filter a signal snapshot and return an annotated frame."""
