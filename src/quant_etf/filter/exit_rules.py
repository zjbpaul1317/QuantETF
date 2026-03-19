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
        self.confirm_weeks = int(config.strategy.exit_confirm_weeks)

    def apply(
        self,
        signal_table: pd.DataFrame,
        current_holdings: pd.DataFrame | None = None,
        as_of_date: str | pd.Timestamp | None = None,
        market_regime_on: bool | None = None,
    ) -> pd.DataFrame:
        holdings = self._normalize_holdings(current_holdings)
        if holdings.empty:
            return self._empty_result()

        frame = self._normalize_signal_table(signal_table)
        target_date = pd.Timestamp(as_of_date) if as_of_date is not None else frame["date"].max()
        snapshot = frame.loc[frame["date"] == target_date].copy()
        if snapshot.empty:
            raise ValueError(f"No signal snapshot found for date: {target_date.date()}")

        regime_on = self._resolve_market_regime(snapshot, market_regime_on)
        merged = holdings.merge(snapshot, on="symbol", how="left")

        merged["trigger_missing_signal"] = merged["date"].isna()
        merged["trigger_stoploss"] = (
            merged["close"].notna()
            & merged["ma60"].notna()
            & (merged["close"] < self.stoploss_ratio * merged["ma60"])
        )
        merged["trigger_regime_off"] = not regime_on

        score_streak_map = self._build_consecutive_trigger_map(
            frame=frame,
            target_date=target_date,
            symbols=merged["symbol"].astype(str).tolist(),
            column="score",
            condition=lambda series: series.fillna(-np.inf) <= self.score_threshold,
        )
        rank_streak_map = self._build_consecutive_trigger_map(
            frame=frame,
            target_date=target_date,
            symbols=merged["symbol"].astype(str).tolist(),
            column="rank",
            condition=lambda series: series.isna() | (series > self.hold_buffer_n),
        )

        merged["trigger_negative_score"] = merged["symbol"].map(score_streak_map).fillna(False)
        merged["trigger_rank_out"] = merged["symbol"].map(rank_streak_map).fillna(False)
        merged["force_sell"] = (
            merged["trigger_missing_signal"]
            | merged["trigger_stoploss"]
            | merged["trigger_regime_off"]
        )
        merged["weak_sell"] = merged["trigger_negative_score"] | merged["trigger_rank_out"]
        merged["should_sell"] = merged["force_sell"] | merged["weak_sell"]
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
    def _normalize_signal_table(signal_table: pd.DataFrame) -> pd.DataFrame:
        frame = signal_table.copy()
        required = {"date", "symbol", "close", "ma60", "score", "rank", "market_regime_on"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"signal_table is missing required columns: {missing_text}")

        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        if "eligible" not in frame.columns:
            frame["eligible"] = False
        if "hold_signal" not in frame.columns:
            frame["hold_signal"] = frame["rank"].le(6).fillna(False)
        return frame.sort_values(["date", "symbol"]).drop_duplicates(subset=["date", "symbol"], keep="last").reset_index(drop=True)

    def _build_consecutive_trigger_map(
        self,
        frame: pd.DataFrame,
        target_date: pd.Timestamp,
        symbols: list[str],
        column: str,
        condition,
    ) -> dict[str, bool]:
        history = frame.loc[frame["date"] <= target_date].copy()
        recent_dates = sorted(history["date"].dropna().unique())[-self.confirm_weeks :]
        if len(recent_dates) < self.confirm_weeks:
            return {symbol: False for symbol in symbols}

        recent = history.loc[history["date"].isin(recent_dates)].copy()
        trigger_map: dict[str, bool] = {}
        for symbol in symbols:
            series = (
                recent.loc[recent["symbol"] == symbol]
                .sort_values("date")[column]
            )
            if len(series) < self.confirm_weeks:
                trigger_map[symbol] = False
                continue
            trigger_map[symbol] = bool(condition(series).all())
        return trigger_map

    @staticmethod
    def _resolve_market_regime(snapshot: pd.DataFrame, market_regime_on: bool | None) -> bool:
        if market_regime_on is not None:
            return bool(market_regime_on)
        series = snapshot["market_regime_on"].dropna()
        if series.empty:
            raise ValueError("signal snapshot contains no valid 'market_regime_on' values")
        return bool(series.iloc[0])

    @staticmethod
    def _join_reasons(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["trigger_missing_signal"]):
            reasons.append("missing_signal")
        if bool(row["trigger_regime_off"]):
            reasons.append("regime_off")
        if bool(row["trigger_stoploss"]):
            reasons.append("stoploss")
        if bool(row["trigger_negative_score"]):
            reasons.append("score_non_positive_2w")
        if bool(row["trigger_rank_out"]):
            reasons.append("rank_out_of_buffer_2w")
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
                "trigger_regime_off",
                "trigger_negative_score",
                "trigger_rank_out",
                "force_sell",
                "weak_sell",
                "should_sell",
                "should_keep",
                "exit_reason",
            ]
        )
