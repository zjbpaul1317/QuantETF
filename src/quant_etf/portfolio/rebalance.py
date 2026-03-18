from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_etf.config.schema import AppConfig
from quant_etf.filter import HoldingExitFilter

from .allocator import EqualWeightAllocator
from .holdings import normalize_holdings
from .target_builder import TargetPortfolioBuilder


@dataclass(frozen=True)
class RebalanceResult:
    signal_date: pd.Timestamp
    target_portfolio: pd.DataFrame
    rebalance_plan: pd.DataFrame
    exit_evaluation: pd.DataFrame


class RebalancePlanner:
    """Turn weekly signals and current holdings into target weights and trade actions."""

    def __init__(
        self,
        config: AppConfig,
        exit_filter: HoldingExitFilter | None = None,
        allocator: EqualWeightAllocator | None = None,
    ) -> None:
        self.config = config
        self.exit_filter = exit_filter or HoldingExitFilter(config)
        self.allocator = allocator or EqualWeightAllocator(config)
        self.target_builder = TargetPortfolioBuilder(
            config=config,
            exit_filter=self.exit_filter,
            allocator=self.allocator,
        )
        self.weight_tolerance = 1e-8

    def plan(
        self,
        weekly_signals: pd.DataFrame,
        current_holdings: pd.DataFrame | None = None,
        as_of_date: str | pd.Timestamp | None = None,
        total_exposure: float | None = None,
    ) -> RebalanceResult:
        snapshot = self.target_builder.select_snapshot(weekly_signals, as_of_date=as_of_date)
        holdings = normalize_holdings(current_holdings)
        exit_evaluation = self.exit_filter.apply(snapshot, holdings)
        target_portfolio = self.target_builder.build(
            weekly_signals,
            current_holdings=holdings,
            as_of_date=as_of_date,
        )
        if total_exposure is not None and not target_portfolio.empty:
            scaled = self.allocator.allocate(target_portfolio["symbol"].tolist(), total_exposure=total_exposure)
            target_portfolio = target_portfolio.drop(columns=["target_weight"]).merge(scaled, on="symbol", how="left")
        rebalance_plan = self._build_rebalance_plan(snapshot, holdings, target_portfolio, exit_evaluation)
        return RebalanceResult(
            signal_date=pd.Timestamp(snapshot["date"].iloc[0]),
            target_portfolio=target_portfolio,
            rebalance_plan=rebalance_plan,
            exit_evaluation=exit_evaluation,
        )

    def _build_rebalance_plan(
        self,
        snapshot: pd.DataFrame,
        holdings: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        exit_evaluation: pd.DataFrame,
    ) -> pd.DataFrame:
        current = holdings[["symbol", "current_weight", "quantity", "market_value"]].copy()
        target = target_portfolio[["symbol", "target_weight", "rank", "score", "hold_reason"]].copy()
        universe = pd.DataFrame(
            {
                "symbol": sorted(set(current["symbol"]).union(target["symbol"])),
            }
        )

        plan = universe.merge(current, on="symbol", how="left").merge(target, on="symbol", how="left")
        plan = plan.merge(
            snapshot[["symbol", "date", "close", "ma60", "eligible", "hold_signal", "buy_signal"]],
            on="symbol",
            how="left",
        )
        plan = plan.merge(
            exit_evaluation[["symbol", "should_sell", "exit_reason"]],
            on="symbol",
            how="left",
        )

        plan["current_weight"] = plan["current_weight"].fillna(0.0)
        plan["target_weight"] = plan["target_weight"].fillna(0.0)
        plan["delta_weight"] = plan["target_weight"] - plan["current_weight"]
        plan["should_sell"] = plan["should_sell"].fillna(False)
        plan["exit_reason"] = plan["exit_reason"].fillna("")
        plan["action"] = plan.apply(self._resolve_action, axis=1)
        plan["trade_reason"] = plan.apply(self._resolve_trade_reason, axis=1)

        return plan.sort_values(
            ["action", "rank", "symbol"],
            key=lambda series: self._action_sort_key(series) if series.name == "action" else series,
            na_position="last",
        ).reset_index(drop=True)

    def _resolve_action(self, row: pd.Series) -> str:
        current_weight = float(row["current_weight"])
        target_weight = float(row["target_weight"])
        delta_weight = float(row["delta_weight"])

        if current_weight <= self.weight_tolerance and target_weight > self.weight_tolerance:
            return "buy"
        if current_weight > self.weight_tolerance and target_weight <= self.weight_tolerance:
            return "sell"
        if abs(delta_weight) <= self.weight_tolerance:
            return "hold"
        if delta_weight > 0:
            return "increase"
        return "reduce"

    @staticmethod
    def _resolve_trade_reason(row: pd.Series) -> str:
        if row["action"] == "sell":
            return row["exit_reason"] or "removed_from_target"
        if row["action"] == "buy":
            return row["hold_reason"] or "new_target"
        if row["action"] in {"increase", "reduce"}:
            return row["hold_reason"] or "rebalance"
        return "keep"

    @staticmethod
    def _action_sort_key(series: pd.Series) -> pd.Series:
        order = {
            "sell": 0,
            "reduce": 1,
            "hold": 2,
            "increase": 3,
            "buy": 4,
        }
        return series.map(order).fillna(99)
