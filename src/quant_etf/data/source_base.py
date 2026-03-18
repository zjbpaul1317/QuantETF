from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .models import DataLoadRequest


class ETFHistoryDataSource(ABC):
    """Abstract ETF history data source."""

    @abstractmethod
    def load_bars(self, request: DataLoadRequest) -> pd.DataFrame:
        """Load raw ETF bars for the requested symbols and date range."""
