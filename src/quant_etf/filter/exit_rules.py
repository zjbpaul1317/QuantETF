from __future__ import annotations

import numpy as np
import pandas as pd

from quant_etf.config.schema import AppConfig

from .base import BaseSignalFilter


class HoldingExitFilter(BaseSignalFilter):
    """Evaluate whether current holdings should be sold on a rebalance date."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.stoploss_ratio = config.strategy.stoploss_ma_ratio
        self.score_threshold = config.strategy.score_threshold
        self.hold_buffer_n = (
            config.strategy.hold_buffer_n
            if config.strategy.enable_buffer_hold
            else config.strategy.buy_top_n
        )

    def apply(self, signal_snapshot: pd.DataFrame, current_holdings: pd.DataFrame | None = None) -> pd.DataFrame:
        holdings = self._normalize_holdings(current_holdings)
        if holdings.empty:
            return self._empty_result()

        snapshot = self._normalize_snapshot(signal_snapshot)
        merged = holdings.merge(snapshot, on="symbol", how="left")

        merged["trigger_missing_signal"] = merged["date"].isna()
        merged["trigger_stoploss"] = (
            merged["close"].notna()
            & merged["ma60"].notna()
            & (merged["close"] < self.stoploss_ratio * merged["ma60"])
        )
        merged["trigger_negative_score"] = merged["score"].fillna(-np.inf) <= self.score_threshold
        merged["trigger_rank_out"] = merged["rank"].isna() | (merged["rank"] > self.hold_buffer_n)
        merged["should_sell"] = (
            merged["trigger_missing_signal"]
            | merged["trigger_stoploss"]
            | merged["trigger_negative_score"]
            | merged["trigger_rank_out"]
        )
        merged["should_keep"] = ~merged["should_sell"]
        merged["exit_reason"] = merged.apply(self._join_reasons, axis=1)
        return merged.sort_values(["should_sell", "symbol"], ascending=[False, True]).reset_index(drop=True)

    @staticmethod
    def _normalize_holdings(current_holdings: pd.DataFrame | None) -> pd.DataFrame:
        if current_holdings is None or current_holdings.empty:
            return pd.DataFrame(columns=["symbol", "current_weight", "quantity", "market_value"])

        frame = current_holdings.copy()
        if "symbol" not in frame.columns:
            raise ValueError("current_holdings must contain a 'symbol' column")

        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        for column in ("current_weight", "quantity", "market_value"):
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if frame["current_weight"].isna().all():
            if frame["market_value"].notna().any() and frame["market_value"].sum() > 0:
                total_market_value = frame["market_value"].sum()
                frame["current_weight"] = frame["market_value"] / total_market_value
            else:
                frame["current_weight"] = 1.0 / len(frame)

        frame = frame.groupby("symbol", as_index=False).agg(
            current_weight=("current_weight", "sum"),
            quantity=("quantity", "sum"),
            market_value=("market_value", "sum"),
        )
        return frame

    @staticmethod
    def _normalize_snapshot(signal_snapshot: pd.DataFrame) -> pd.DataFrame:
        frame = signal_snapshot.copy()
        required = {"date", "symbol", "close", "ma60", "score", "rank"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"signal snapshot is missing required columns: {missing_text}")

        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        if "eligible" not in frame.columns:
            frame["eligible"] = False
        if "hold_signal" not in frame.columns:
            frame["hold_signal"] = frame["rank"].le(5).fillna(False)
        return frame.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)

    @staticmethod
    def _join_reasons(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["trigger_missing_signal"]):
            reasons.append("missing_signal")
        if bool(row["trigger_stoploss"]):
            reasons.append("stoploss")
        if bool(row["trigger_negative_score"]):
            reasons.append("score_non_positive")
        if bool(row["trigger_rank_out"]):
            reasons.append("rank_out_of_buffer")
        return "|".join(reasons) if reasons else "keep"

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "current_weight",
                "quantity",
                "market_value",
                "date",
                "close",
                "ma60",
                "score",
                "rank",
                "eligible",
                "hold_signal",
                "trigger_missing_signal",
                "trigger_stoploss",
                "trigger_negative_score",
                "trigger_rank_out",
                "should_sell",
                "should_keep",
                "exit_reason",
            ]
        )
