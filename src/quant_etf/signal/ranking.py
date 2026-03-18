from __future__ import annotations

import pandas as pd

from quant_etf.config.schema import AppConfig


class SignalRanker:
    """Apply eligibility rules and cross-sectional score ranking."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ma_column = f"ma{config.strategy.ma_window}"
        self.score_threshold = config.strategy.score_threshold
        self.buy_top_n = config.strategy.buy_top_n
        self.hold_buffer_n = config.strategy.hold_buffer_n

    def rank(self, scored_features: pd.DataFrame) -> pd.DataFrame:
        frame = scored_features.copy()
        self._ensure_required_columns(frame)

        frame["passed_listing"] = frame["listed_days"] >= self.config.universe.min_listed_days
        frame["passed_liquidity"] = frame["avg_turnover"] >= self.config.universe.min_avg_turnover
        frame["passed_trend"] = frame["close"] > frame[self.ma_column]
        frame["passed_score"] = frame["score"] > self.score_threshold
        frame["eligible"] = (
            frame["passed_listing"]
            & frame["passed_liquidity"]
            & frame["passed_trend"]
            & frame["passed_score"]
        )

        frame["rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
        eligible = frame["eligible"]
        if eligible.any():
            eligible_ranks = frame.loc[eligible].groupby(level="trade_date")["score"].rank(
                ascending=False,
                method="first",
            )
            frame.loc[eligible, "rank"] = eligible_ranks.astype("Int64")

        frame["hold_signal"] = frame["eligible"] & frame["rank"].le(self.hold_buffer_n).fillna(False)
        frame["buy_signal"] = frame["eligible"] & frame["rank"].le(self.buy_top_n).fillna(False)
        return frame

    def _ensure_required_columns(self, frame: pd.DataFrame) -> None:
        required = {"close", "listed_days", "avg_turnover", "score", self.ma_column}
        missing = sorted(required.difference(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"scored feature frame is missing required columns: {missing_text}")
