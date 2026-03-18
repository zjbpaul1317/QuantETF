from __future__ import annotations

import numpy as np
import pandas as pd

from quant_etf.config.schema import DataConfig


class ETFDataPreprocessor:
    """Apply adjustment, derive helper fields and return a canonical bar table."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def preprocess(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return bars.copy()

        frame = bars.copy()
        frame = self._add_default_adj_factor(frame)
        frame = self._preserve_raw_prices(frame)
        frame = self._apply_adjustment(frame)
        frame = self._add_derived_fields(frame)
        frame = self._set_index(frame)
        return frame

    @staticmethod
    def _add_default_adj_factor(frame: pd.DataFrame) -> pd.DataFrame:
        if "adj_factor" not in frame.columns:
            frame["adj_factor"] = 1.0
        frame["adj_factor"] = frame["adj_factor"].fillna(1.0)
        return frame

    @staticmethod
    def _preserve_raw_prices(frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("open", "high", "low", "close"):
            frame[f"raw_{column}"] = frame[column]
        return frame

    def _apply_adjustment(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.config.use_adjusted_price or self.config.adjustment == "none":
            return frame

        adjusted_frames: list[pd.DataFrame] = []
        for _, group in frame.groupby("symbol", sort=False):
            adjusted_frames.append(self._adjust_group(group.copy()))
        return pd.concat(adjusted_frames, ignore_index=True)

    def _adjust_group(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("trade_date").copy()
        factor = group["adj_factor"].replace(0, np.nan).ffill().bfill().fillna(1.0)
        if self.config.adjustment == "qfq":
            normalized = factor / factor.iloc[-1]
        elif self.config.adjustment == "hfq":
            normalized = factor / factor.iloc[0]
        else:
            normalized = pd.Series(1.0, index=group.index)

        for column in ("open", "high", "low", "close"):
            group[column] = group[column] * normalized
        return group

    @staticmethod
    def _add_derived_fields(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        grouped = frame.groupby("symbol", sort=False)
        frame["prev_close"] = grouped["close"].shift(1)
        frame["return_1d"] = grouped["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        frame["vwap"] = np.where(frame["volume"] > 0, frame["amount"] / frame["volume"], frame["close"])
        frame["listed_days"] = grouped.cumcount() + 1
        frame["is_suspended"] = frame["volume"].fillna(0) <= 0
        frame["is_tradeable"] = (~frame["is_suspended"]) & frame["close"].notna()
        return frame

    @staticmethod
    def _set_index(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.set_index(["trade_date", "symbol"]).sort_index()
