from __future__ import annotations

import numpy as np
import pandas as pd

from quant_etf.config.schema import AppConfig
from quant_etf.filter import HoldingExitFilter

from .allocator import EqualWeightAllocator
from .holdings import normalize_holdings


class TargetPortfolioBuilder:
    """Build target portfolio weights from weekly signal snapshots."""

    REQUIRED_COLUMNS = {"date", "symbol", "score", "rank", "buy_signal", "market_regime_on"}
    SINGLE_SYMBOL_MAX_EXPOSURE = 0.5

    def __init__(
        self,
        config: AppConfig,
        exit_filter: HoldingExitFilter | None = None,
        allocator: EqualWeightAllocator | None = None,
    ) -> None:
        self.config = config
        self.exit_filter = exit_filter or HoldingExitFilter(config)
        self.allocator = allocator or EqualWeightAllocator(config)
        self.min_weight_delta = float(config.strategy.min_rebalance_weight_delta)
        self.min_score_upgrade = float(config.strategy.min_score_upgrade)
        self.min_score_challenge_ratio = float(config.strategy.min_score_challenge_ratio)

    def build(
        self,
        signal_table: pd.DataFrame,
        current_holdings: pd.DataFrame | None = None,
        as_of_date: str | pd.Timestamp | None = None,
        market_regime_on: bool | None = None,
    ) -> pd.DataFrame:
        frame = self._normalize_signal_table(signal_table)
        snapshot = self.select_snapshot(frame, as_of_date=as_of_date)
        self._ensure_required_columns(snapshot)
        holdings = normalize_holdings(current_holdings)

        regime_on = self._resolve_market_regime(snapshot, market_regime_on)
        target_symbols: list[str]
        hold_reason_map: dict[str, str]
        exit_evaluation = self.exit_filter.apply(
            frame,
            holdings,
            as_of_date=pd.Timestamp(snapshot["date"].iloc[0]),
            market_regime_on=regime_on,
        )

        if not regime_on:
            if self.config.market_regime.risk_off_action == "flat" or self.config.market_regime.risk_off_exposure <= 0:
                return self._empty_target(snapshot, hold_reason="risk_off_flat")

            target_symbols, hold_reason_map = self._select_symbols(
                snapshot,
                holdings,
                exit_evaluation,
                apply_buffer_hold=False,
            )
            target_exposure = min(
                self._determine_target_exposure(len(target_symbols)),
                self.config.market_regime.risk_off_exposure,
            )
            for symbol in target_symbols:
                hold_reason_map[symbol] = "risk_off_scaled"
        else:
            target_symbols, hold_reason_map = self._select_symbols(
                snapshot,
                holdings,
                exit_evaluation,
                apply_buffer_hold=self.config.strategy.enable_buffer_hold,
            )
            target_exposure = self._determine_target_exposure(len(target_symbols))

        target_weights = self.allocator.allocate(
            target_symbols,
            total_exposure=target_exposure,
            signal_snapshot=snapshot.loc[snapshot["symbol"].isin(target_symbols)].copy(),
        )
        snapshot_columns = ["symbol", "date", "score", "rank"]
        if "volatility_20" in snapshot.columns:
            snapshot_columns.append("volatility_20")
        target = target_weights.merge(
            snapshot[snapshot_columns],
            on="symbol",
            how="left",
        )
        target = target.rename(columns={"date": "rebalance_date"})
        target["hold_reason"] = target["symbol"].map(hold_reason_map)
        target = self._apply_weight_turnover_guard(target, holdings)
        return target.sort_values(["rank", "symbol"], na_position="last").reset_index(drop=True)

    def build_all(
        self,
        signal_table: pd.DataFrame,
        initial_holdings: pd.DataFrame | None = None,
        market_regime_overrides: dict[pd.Timestamp, bool] | None = None,
    ) -> pd.DataFrame:
        frame = signal_table.copy()
        if frame.empty:
            return self._empty_target_table()

        frame["date"] = pd.to_datetime(frame["date"])
        rebalance_dates = sorted(frame["date"].dropna().unique())[:: self.config.strategy.rebalance_interval_weeks]
        current_holdings = normalize_holdings(initial_holdings)
        all_targets: list[pd.DataFrame] = []

        for rebalance_date in rebalance_dates:
            regime_on = None
            if market_regime_overrides is not None:
                regime_on = market_regime_overrides.get(pd.Timestamp(rebalance_date))

            target = self.build(
                signal_table=frame,
                current_holdings=current_holdings,
                as_of_date=pd.Timestamp(rebalance_date),
                market_regime_on=regime_on,
            )
            all_targets.append(target)
            current_holdings = self._target_to_holdings(target)

        if not all_targets:
            return self._empty_target_table()
        return pd.concat(all_targets, ignore_index=True, sort=False)

    @staticmethod
    def select_snapshot(signal_table: pd.DataFrame, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        if signal_table.empty:
            raise ValueError("signal_table must not be empty")

        frame = signal_table.copy()
        if "date" not in frame.columns or "symbol" not in frame.columns:
            raise ValueError("signal_table must contain 'date' and 'symbol' columns")

        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        target_date = pd.Timestamp(as_of_date) if as_of_date is not None else frame["date"].max()
        snapshot = frame.loc[frame["date"] == target_date].copy()
        if snapshot.empty:
            raise ValueError(f"No signal snapshot found for date: {target_date.date()}")
        return snapshot.sort_values(["rank", "symbol"], na_position="last").reset_index(drop=True)

    @staticmethod
    def _normalize_signal_table(signal_table: pd.DataFrame) -> pd.DataFrame:
        frame = signal_table.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        return frame.sort_values(["date", "rank", "symbol"], na_position="last").reset_index(drop=True)

    def _select_symbols(
        self,
        snapshot: pd.DataFrame,
        holdings: pd.DataFrame,
        exit_evaluation: pd.DataFrame,
        apply_buffer_hold: bool,
    ) -> tuple[list[str], dict[str, str]]:
        selected: list[str] = []
        hold_reason_map: dict[str, str] = {}
        replacement_pool = pd.DataFrame(columns=["symbol", "rank", "score"])

        if apply_buffer_hold and not holdings.empty:
            kept = exit_evaluation.loc[exit_evaluation["should_keep"], ["symbol"]].merge(
                snapshot[["symbol", "rank", "score"]],
                on="symbol",
                how="left",
            )
            kept = kept.sort_values(["rank", "score", "symbol"], ascending=[True, False, True], na_position="last")
            for row in kept.itertuples(index=False):
                if len(selected) >= self.config.strategy.buy_top_n:
                    break
                selected.append(row.symbol)
                if pd.notna(row.rank) and int(row.rank) <= self.config.strategy.buy_top_n:
                    hold_reason_map[row.symbol] = "keep_top_buy"
                else:
                    hold_reason_map[row.symbol] = "keep_buffer"

            replacement_pool = (
                exit_evaluation.loc[(~exit_evaluation["should_keep"]) & (~exit_evaluation["force_sell"]), ["symbol"]]
                .merge(snapshot[["symbol", "rank", "score"]], on="symbol", how="left")
                .sort_values(["rank", "score", "symbol"], ascending=[True, False, True], na_position="last")
                .reset_index(drop=True)
            )

        buy_candidates = snapshot.loc[snapshot["buy_signal"]].sort_values(["rank", "symbol"])
        buy_rows = list(buy_candidates.itertuples(index=False))
        buy_index = 0
        for row in replacement_pool.itertuples(index=False):
            if len(selected) >= self.config.strategy.buy_top_n:
                break
            challenger = None
            while buy_index < len(buy_rows):
                candidate = buy_rows[buy_index]
                buy_index += 1
                if candidate.symbol not in selected:
                    challenger = candidate
                    break

            if challenger is None:
                selected.append(row.symbol)
                hold_reason_map[row.symbol] = "keep_small_edge"
                continue

            challenger_score = float(challenger.score) if pd.notna(challenger.score) else -np.inf
            incumbent_score = float(row.score) if pd.notna(row.score) else -np.inf
            if not self._is_challenger_strong_enough(challenger_score, incumbent_score):
                selected.append(row.symbol)
                hold_reason_map[row.symbol] = "keep_small_edge"
                buy_index -= 1
                continue

            if challenger.symbol not in selected:
                selected.append(challenger.symbol)
                hold_reason_map[challenger.symbol] = "new_buy"

        for idx in range(buy_index, len(buy_rows)):
            row = buy_rows[idx]
            if len(selected) >= self.config.strategy.buy_top_n:
                break
            if row.symbol in selected:
                continue
            selected.append(row.symbol)
            hold_reason_map[row.symbol] = "new_buy"

        if len(selected) > self.config.strategy.buy_top_n:
            selected = selected[: self.config.strategy.buy_top_n]

        return selected, hold_reason_map

    def _is_challenger_strong_enough(self, challenger_score: float, incumbent_score: float) -> bool:
        if not np.isfinite(challenger_score):
            return False
        if not np.isfinite(incumbent_score):
            return True
        if incumbent_score > 0:
            return challenger_score >= incumbent_score * self.min_score_challenge_ratio
        return challenger_score >= incumbent_score + self.min_score_upgrade

    def _determine_target_exposure(self, selected_count: int) -> float:
        if selected_count <= 0:
            return 0.0
        if selected_count == 1:
            return self.SINGLE_SYMBOL_MAX_EXPOSURE
        return 1.0 - self.config.trading.cash_reserve_ratio

    def _apply_weight_turnover_guard(
        self,
        target: pd.DataFrame,
        holdings: pd.DataFrame,
    ) -> pd.DataFrame:
        if target.empty or holdings.empty:
            return target

        current_weights = holdings[["symbol", "current_weight"]].copy()
        current_weights["symbol"] = current_weights["symbol"].astype(str).str.upper()
        guarded = target.merge(current_weights, on="symbol", how="left")
        reusable = guarded["current_weight"].notna() & (
            (guarded["target_weight"] - guarded["current_weight"]).abs() < self.min_weight_delta
        )
        guarded.loc[reusable, "target_weight"] = guarded.loc[reusable, "current_weight"]
        guarded.loc[reusable, "hold_reason"] = guarded.loc[reusable, "hold_reason"].fillna("keep_weight_threshold")
        guarded.loc[reusable, "hold_reason"] = guarded.loc[reusable, "hold_reason"].replace("new_buy", "keep_weight_threshold")
        return guarded.drop(columns=["current_weight"])

    def _resolve_market_regime(self, snapshot: pd.DataFrame, market_regime_on: bool | None) -> bool:
        if "market_regime_on" not in snapshot.columns:
            raise ValueError("signal snapshot must contain 'market_regime_on' before building targets")

        if market_regime_on is not None:
            return bool(market_regime_on)

        series = snapshot["market_regime_on"].dropna()
        if series.empty:
            raise ValueError("signal snapshot contains no valid 'market_regime_on' values")
        return bool(series.iloc[0])

    def _ensure_required_columns(self, snapshot: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(snapshot.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"signal snapshot is missing required columns: {missing_text}")

    @staticmethod
    def _empty_target(snapshot: pd.DataFrame, hold_reason: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rebalance_date": pd.Series(dtype="datetime64[ns]"),
                "symbol": pd.Series(dtype="object"),
                "target_weight": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "rank": pd.Series(dtype="Int64"),
                "volatility_20": pd.Series(dtype="float64"),
                "hold_reason": pd.Series(dtype="object"),
            }
        )

    @staticmethod
    def _empty_target_table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rebalance_date": pd.Series(dtype="datetime64[ns]"),
                "symbol": pd.Series(dtype="object"),
                "target_weight": pd.Series(dtype="float64"),
                "score": pd.Series(dtype="float64"),
                "rank": pd.Series(dtype="Int64"),
                "volatility_20": pd.Series(dtype="float64"),
                "hold_reason": pd.Series(dtype="object"),
            }
        )

    @staticmethod
    def _target_to_holdings(target: pd.DataFrame) -> pd.DataFrame:
        if target.empty:
            return pd.DataFrame(columns=["symbol", "current_weight", "quantity", "market_value"])

        return target[["symbol", "target_weight"]].rename(columns={"target_weight": "current_weight"}).copy()
