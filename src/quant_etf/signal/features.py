from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_etf.config.schema import AppConfig


class IndicatorCalculator:
    """Calculate daily strategy indicators from ETF bar history."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.return_windows = tuple(config.strategy.lookback_windows)
        self.ma_window = config.strategy.ma_window
        self.volatility_window = 20
        self.turnover_window = config.universe.liquidity_lookback

    def calculate(self, history: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_history(history)
        grouped = frame.groupby(level="symbol", sort=False)

        for window in self.return_windows:
            frame[f"r{window}"] = grouped["close"].pct_change(periods=window)

        frame[f"ma{self.ma_window}"] = grouped["close"].transform(
            lambda series: series.rolling(self.ma_window, min_periods=self.ma_window).mean()
        )
        frame["volatility_20"] = grouped["return_1d"].transform(
            lambda series: series.rolling(self.volatility_window, min_periods=self.volatility_window).std(ddof=0)
        )
        frame["avg_turnover"] = grouped["amount"].transform(
            lambda series: series.rolling(self.turnover_window, min_periods=self.turnover_window).mean()
        )

        if "listed_days" not in frame.columns:
            frame["listed_days"] = grouped.cumcount() + 1

        return frame.sort_index()

    def _prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        frame = history.copy()

        if not isinstance(frame.index, pd.MultiIndex):
            if {"trade_date", "symbol"}.issubset(frame.columns):
                frame = frame.set_index(["trade_date", "symbol"])
            else:
                raise ValueError("history must have a MultiIndex ['trade_date', 'symbol'] or matching columns")

        if frame.index.names != ["trade_date", "symbol"]:
            frame.index = frame.index.set_names(["trade_date", "symbol"])

        frame = frame.sort_index()
        frame = self._ensure_columns(frame, required_columns=("close", "amount"))

        if "return_1d" not in frame.columns:
            grouped = frame.groupby(level="symbol", sort=False)
            frame["return_1d"] = grouped["close"].pct_change()

        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
        frame["return_1d"] = pd.to_numeric(frame["return_1d"], errors="coerce")
        return frame

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"history is missing required columns: {missing_text}")
        return frame
