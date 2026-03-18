from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    cumulative_return: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    average_holding_period: float
    annual_turnover_rate: float
    turnover_rate: float
    total_return: float

    def to_dict(self) -> dict[str, float]:
        return {
            "cumulative_return": self.cumulative_return,
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "average_holding_period": self.average_holding_period,
            "annual_turnover_rate": self.annual_turnover_rate,
            "turnover_rate": self.turnover_rate,
            "total_return": self.total_return,
        }


def calculate_metrics(
    daily_nav: pd.DataFrame,
    trades: pd.DataFrame,
    daily_holdings: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestMetrics:
    if daily_nav.empty:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    nav = daily_nav["nav"].astype(float)
    daily_return = daily_nav["daily_return"].fillna(0.0).astype(float)
    cumulative_return = float(nav.iloc[-1] - 1.0)
    total_return = cumulative_return

    periods = max(len(daily_nav), 1)
    annual_return = float(np.power(nav.iloc[-1], 252 / periods) - 1.0)

    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    excess_daily_return = daily_return - risk_free_rate / 252
    daily_volatility = float(excess_daily_return.std(ddof=0))
    annual_volatility = float(daily_return.std(ddof=0) * np.sqrt(252))
    sharpe_ratio = 0.0 if daily_volatility == 0.0 else float(np.sqrt(252) * excess_daily_return.mean() / daily_volatility)
    calmar_ratio = 0.0 if max_drawdown == 0.0 else float(annual_return / abs(max_drawdown))

    sell_trades = trades.loc[trades["side"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    if sell_trades.empty:
        win_rate = 0.0
    else:
        win_rate = float((sell_trades["realized_pnl"] > 0).mean())

    turnover_rate = float(daily_nav["turnover"].fillna(0.0).sum())
    years = max(periods / 252, 1 / 252)
    annual_turnover_rate = float(turnover_rate / years)
    average_holding_period = _calculate_average_holding_period(trades)

    return BacktestMetrics(
        cumulative_return=cumulative_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        average_holding_period=average_holding_period,
        annual_turnover_rate=annual_turnover_rate,
        turnover_rate=turnover_rate,
        total_return=total_return,
    )


def _calculate_average_holding_period(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0

    frame = trades.copy()
    required = {"trade_date", "symbol", "side", "quantity"}
    if not required.issubset(frame.columns):
        return 0.0

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    queues: dict[str, list[list[object]]] = {}
    total_holding_days = 0.0
    matched_quantity = 0

    for row in frame.itertuples(index=False):
        symbol = str(row.symbol)
        side = str(row.side).upper()
        quantity = int(row.quantity)
        trade_date = pd.Timestamp(row.trade_date)

        if quantity <= 0:
            continue

        if side == "BUY":
            queues.setdefault(symbol, []).append([trade_date, quantity])
            continue

        if side != "SELL":
            continue

        queue = queues.setdefault(symbol, [])
        remaining = quantity
        while remaining > 0 and queue:
            buy_date, buy_quantity = queue[0]
            matched = min(remaining, int(buy_quantity))
            total_holding_days += max((trade_date - pd.Timestamp(buy_date)).days, 0) * matched
            matched_quantity += matched
            remaining -= matched
            buy_quantity -= matched
            if buy_quantity <= 0:
                queue.pop(0)
            else:
                queue[0][1] = buy_quantity

    if matched_quantity == 0:
        return 0.0
    return float(total_holding_days / matched_quantity)
