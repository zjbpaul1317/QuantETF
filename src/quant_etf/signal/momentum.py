from __future__ import annotations

import pandas as pd

from quant_etf.config.schema import AppConfig


class MomentumScorer:
    """Calculate cross-sectional momentum score from return features."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.windows = tuple(config.strategy.lookback_windows)
        self.weights = tuple(config.strategy.score_weights)

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = features.copy()
        required_columns = [f"r{window}" for window in self.windows]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"feature frame is missing return columns: {missing_text}")

        score = pd.Series(0.0, index=frame.index, dtype="float64")
        for window, weight in zip(self.windows, self.weights, strict=True):
            score = score.add(frame[f"r{window}"] * weight, fill_value=0.0)
        frame["score"] = score
        return frame
