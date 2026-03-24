from __future__ import annotations

import pandas as pd

from quant_etf.config.schema import AppConfig


class MomentumScorer:
    """Calculate cross-sectional momentum score from standardized factor signals."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.factor_columns = ("bias_factor", "slope_factor", "efficiency_factor")
        self.weights = tuple(config.strategy.score_weights)

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        frame = features.copy()
        required_columns = list(self.factor_columns)
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"feature frame is missing factor columns: {missing_text}")

        score = pd.Series(0.0, index=frame.index, dtype="float64")
        grouped = frame.groupby(level="trade_date", sort=False)
        for column, weight in zip(self.factor_columns, self.weights, strict=True):
            z_column = f"{column}_z"
            frame[z_column] = grouped[column].transform(self._zscore)
            score = score.add(frame[z_column] * weight, fill_value=0.0)
        frame["score"] = score
        return frame

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        mean = numeric.mean()
        std = numeric.std(ddof=0)
        if pd.isna(std) or std <= 1e-12:
            return pd.Series(0.0, index=series.index, dtype="float64")
        return ((numeric - mean) / std).fillna(0.0)
