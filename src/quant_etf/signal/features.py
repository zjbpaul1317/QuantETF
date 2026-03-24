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
        self.bias_ma_window = config.strategy.bias_ma_window
        self.bias_regression_window = config.strategy.bias_regression_window
        self.slope_window = config.strategy.slope_window
        self.efficiency_window = config.strategy.efficiency_window
        self.volatility_window = 20
        self.turnover_window = config.universe.liquidity_lookback

    def calculate(self, history: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_history(history)
        grouped = frame.groupby(level="symbol", sort=False)

        for window in self.return_windows:
            frame[f"r{window}"] = grouped["close"].pct_change(periods=window)

        frame["pivot"] = (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
        frame[f"ma{self.ma_window}"] = grouped["close"].transform(
            lambda series: series.rolling(self.ma_window, min_periods=self.ma_window).mean()
        )
        frame["bias"] = grouped["close"].transform(
            lambda series: series / series.rolling(self.bias_ma_window, min_periods=self.bias_ma_window).mean()
        )
        frame["bias_factor"] = grouped["bias"].transform(
            lambda series: series.rolling(self.bias_regression_window, min_periods=self.bias_regression_window)
            .apply(self._calc_bias_score, raw=True)
        )
        frame["slope_factor"] = grouped["close"].transform(
            lambda series: series.rolling(self.slope_window, min_periods=self.slope_window)
            .apply(self._calc_slope_score, raw=True)
        )
        frame["efficiency_factor"] = grouped["pivot"].transform(
            lambda series: series.rolling(self.efficiency_window, min_periods=self.efficiency_window)
            .apply(self._calc_efficiency_score, raw=True)
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
        frame = self._ensure_columns(frame, required_columns=("open", "high", "low", "close", "amount"))

        if "return_1d" not in frame.columns:
            grouped = frame.groupby(level="symbol", sort=False)
            frame["return_1d"] = grouped["close"].pct_change()

        for column in ("open", "high", "low", "close", "amount"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
        frame["return_1d"] = pd.to_numeric(frame["return_1d"], errors="coerce")
        return frame

    @staticmethod
    def _calc_bias_score(values: np.ndarray) -> float:
        series = IndicatorCalculator._clean_window(values)
        if series is None:
            return np.nan
        normalized = series / series[0]
        slope, _ = IndicatorCalculator._linear_regression_stats(normalized)
        return float(slope * 10_000.0)

    @staticmethod
    def _calc_slope_score(values: np.ndarray) -> float:
        series = IndicatorCalculator._clean_window(values)
        if series is None:
            return np.nan
        normalized = series / series[0]
        slope, r_squared = IndicatorCalculator._linear_regression_stats(normalized)
        return float(slope * r_squared * 10_000.0)

    @staticmethod
    def _calc_efficiency_score(values: np.ndarray) -> float:
        series = IndicatorCalculator._clean_window(values)
        if series is None:
            return np.nan

        log_prices = np.log(series)
        momentum = 100.0 * (log_prices[-1] - log_prices[0])
        direction = abs(log_prices[-1] - log_prices[0])
        volatility = np.abs(np.diff(log_prices)).sum()
        if volatility <= 0:
            return 0.0
        return float(momentum * (direction / volatility))

    @staticmethod
    def _clean_window(values: np.ndarray) -> np.ndarray | None:
        series = np.asarray(values, dtype="float64")
        if series.size < 2 or np.isnan(series).any() or np.any(series <= 0):
            return None
        return series

    @staticmethod
    def _linear_regression_stats(values: np.ndarray) -> tuple[float, float]:
        x = np.arange(values.size, dtype="float64")
        x_mean = float(x.mean())
        y_mean = float(values.mean())
        denom = float(np.square(x - x_mean).sum())
        if denom <= 0:
            return 0.0, 0.0

        slope = float(((x - x_mean) * (values - y_mean)).sum() / denom)
        intercept = y_mean - slope * x_mean
        fitted = slope * x + intercept
        ss_tot = float(np.square(values - y_mean).sum())
        ss_res = float(np.square(values - fitted).sum())
        if ss_tot <= 0:
            return slope, 0.0
        r_squared = max(0.0, 1.0 - ss_res / ss_tot)
        return slope, r_squared

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"history is missing required columns: {missing_text}")
        return frame
