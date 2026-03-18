from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_etf.config.schema import AppConfig

from .metrics import BacktestMetrics, calculate_metrics


@dataclass(frozen=True)
class BacktestResult:
    daily_holdings: pd.DataFrame
    daily_nav: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


class BacktestEngine:
    """A concise weekly-rebalance backtest engine for ETF rotation strategies."""

    REQUIRED_PRICE_COLUMNS = {"open", "close"}

    def __init__(
        self,
        config: AppConfig,
        cost_params: dict[str, float] | None = None,
        slippage_rate: float | None = None,
    ) -> None:
        self.config = config
        self.commission_rate = config.cost.commission_rate
        self.stamp_duty_rate = config.cost.stamp_duty_rate
        self.min_commission = config.cost.min_commission
        self.slippage_rate = config.cost.slippage_rate
        if cost_params:
            self.commission_rate = float(cost_params.get("commission_rate", self.commission_rate))
            self.stamp_duty_rate = float(cost_params.get("stamp_duty_rate", self.stamp_duty_rate))
            self.min_commission = float(cost_params.get("min_commission", self.min_commission))
        if slippage_rate is not None:
            self.slippage_rate = float(slippage_rate)

        self.lot_size = config.trading.lot_size
        self.risk_free_rate = config.backtest.risk_free_rate

    def run(
        self,
        history: pd.DataFrame,
        target_portfolio: pd.DataFrame,
        rebalance_dates: list[str | pd.Timestamp] | None = None,
        initial_capital: float | None = None,
    ) -> BacktestResult:
        price_data = self._prepare_history(history)
        targets = self._prepare_targets(target_portfolio)
        trading_dates = price_data["trading_dates"]
        execution_schedule = self._build_execution_schedule(trading_dates, targets, rebalance_dates)

        cash = float(initial_capital if initial_capital is not None else self.config.backtest.initial_capital)
        positions: dict[str, dict[str, float]] = {}
        trades: list[dict[str, object]] = []
        nav_records: list[dict[str, object]] = []
        holding_records: list[dict[str, object]] = []
        previous_nav = cash

        for trade_date in trading_dates:
            traded_value = 0.0
            if trade_date in execution_schedule:
                signal_date = execution_schedule[trade_date]
                target_snapshot = targets.get(signal_date, self._empty_target_snapshot())
                cash, traded_value, execution_trades = self._execute_rebalance(
                    trade_date=trade_date,
                    signal_date=signal_date,
                    target_snapshot=target_snapshot,
                    positions=positions,
                    cash=cash,
                    price_data=price_data,
                )
                trades.extend(execution_trades)

            holdings_value, holding_rows = self._mark_to_market(
                trade_date=trade_date,
                positions=positions,
                close_prices=price_data["close_prices"],
                cash=cash,
            )
            portfolio_value = cash + holdings_value
            nav = portfolio_value / float(initial_capital if initial_capital is not None else self.config.backtest.initial_capital)
            daily_return = 0.0 if not nav_records else portfolio_value / previous_nav - 1.0
            turnover = 0.0 if previous_nav == 0 else traded_value / previous_nav

            nav_records.append(
                {
                    "date": trade_date,
                    "cash": cash,
                    "holdings_value": holdings_value,
                    "portfolio_value": portfolio_value,
                    "nav": nav,
                    "daily_return": daily_return,
                    "turnover": turnover,
                }
            )
            holding_records.extend(holding_rows)
            previous_nav = portfolio_value

        daily_holdings = pd.DataFrame(holding_records)
        daily_nav = pd.DataFrame(nav_records)
        trades_df = pd.DataFrame(trades)
        metrics = calculate_metrics(
            daily_nav,
            trades_df,
            daily_holdings=daily_holdings,
            risk_free_rate=self.risk_free_rate,
        ).to_dict()
        return BacktestResult(
            daily_holdings=daily_holdings,
            daily_nav=daily_nav,
            trades=trades_df,
            metrics=metrics,
        )

    def _prepare_history(self, history: pd.DataFrame) -> dict[str, object]:
        frame = history.copy()
        if not isinstance(frame.index, pd.MultiIndex):
            if {"trade_date", "symbol"}.issubset(frame.columns):
                frame = frame.set_index(["trade_date", "symbol"])
            else:
                raise ValueError("history must have a MultiIndex ['trade_date', 'symbol'] or matching columns")

        frame.index = frame.index.set_names(["trade_date", "symbol"])
        frame = frame.sort_index().reset_index()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])

        missing = sorted(self.REQUIRED_PRICE_COLUMNS.difference(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"history is missing required price columns: {missing_text}")

        open_prices = frame.pivot(index="trade_date", columns="symbol", values="open").sort_index()
        close_prices = frame.pivot(index="trade_date", columns="symbol", values="close").sort_index().ffill()
        prev_close_prices = close_prices.shift(1)
        trading_dates = list(close_prices.index)
        return {
            "open_prices": open_prices,
            "close_prices": close_prices,
            "prev_close_prices": prev_close_prices,
            "trading_dates": trading_dates,
        }

    @staticmethod
    def _prepare_targets(target_portfolio: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
        if target_portfolio.empty:
            return {}

        frame = target_portfolio.copy()
        required = {"rebalance_date", "symbol", "target_weight"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"target_portfolio is missing required columns: {missing_text}")

        frame["rebalance_date"] = pd.to_datetime(frame["rebalance_date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)

        return {
            rebalance_date: snapshot.sort_values("symbol").reset_index(drop=True)
            for rebalance_date, snapshot in frame.groupby("rebalance_date", sort=True)
        }

    @staticmethod
    def _build_execution_schedule(
        trading_dates: list[pd.Timestamp],
        targets: dict[pd.Timestamp, pd.DataFrame],
        rebalance_dates: list[str | pd.Timestamp] | None = None,
    ) -> dict[pd.Timestamp, pd.Timestamp]:
        if rebalance_dates is None:
            signal_dates = sorted(targets.keys())
        else:
            signal_dates = sorted(pd.Timestamp(date) for date in rebalance_dates)

        execution_schedule: dict[pd.Timestamp, pd.Timestamp] = {}
        trading_index = pd.Index(trading_dates)
        for signal_date in signal_dates:
            loc = trading_index.searchsorted(signal_date, side="right")
            if loc >= len(trading_index):
                continue
            execution_schedule[trading_index[loc]] = signal_date
        return execution_schedule

    def _execute_rebalance(
        self,
        trade_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        target_snapshot: pd.DataFrame,
        positions: dict[str, dict[str, float]],
        cash: float,
        price_data: dict[str, object],
    ) -> tuple[float, float, list[dict[str, object]]]:
        open_prices = price_data["open_prices"].loc[trade_date]
        prev_close_prices = price_data["prev_close_prices"].loc[trade_date]
        valuation_prices = open_prices.combine_first(prev_close_prices)

        portfolio_value = cash
        for symbol, state in positions.items():
            if symbol in valuation_prices and pd.notna(valuation_prices[symbol]):
                portfolio_value += state["quantity"] * float(valuation_prices[symbol])

        target_quantity_map: dict[str, int] = {}
        for row in target_snapshot.itertuples(index=False):
            price = open_prices.get(row.symbol)
            if pd.isna(price) or price <= 0:
                continue
            fill_price = float(price) * (1.0 + self.slippage_rate)
            target_value = portfolio_value * float(row.target_weight)
            target_quantity_map[row.symbol] = self._round_lot_down(target_value / fill_price)

        trades: list[dict[str, object]] = []
        traded_value = 0.0

        current_symbols = sorted(set(positions).union(target_quantity_map))

        for symbol in current_symbols:
            current_quantity = int(positions.get(symbol, {}).get("quantity", 0))
            target_quantity = int(target_quantity_map.get(symbol, 0))
            sell_quantity = current_quantity - target_quantity
            if sell_quantity <= 0:
                continue

            price = open_prices.get(symbol)
            if pd.isna(price) or price <= 0:
                continue
            fill_price = float(price) * (1.0 - self.slippage_rate)
            gross_amount = sell_quantity * fill_price
            commission = self._commission(gross_amount)
            stamp_duty = gross_amount * self.stamp_duty_rate
            net_amount = gross_amount - commission - stamp_duty
            avg_cost = float(positions[symbol]["avg_cost"])
            realized_pnl = net_amount - sell_quantity * avg_cost

            positions[symbol]["quantity"] = current_quantity - sell_quantity
            if positions[symbol]["quantity"] <= 0:
                del positions[symbol]

            cash += net_amount
            traded_value += gross_amount
            trades.append(
                self._build_trade_record(
                    trade_date=trade_date,
                    signal_date=signal_date,
                    symbol=symbol,
                    side="SELL",
                    quantity=sell_quantity,
                    price=fill_price,
                    gross_amount=gross_amount,
                    commission=commission,
                    stamp_duty=stamp_duty,
                    realized_pnl=realized_pnl,
                )
            )

        for symbol in current_symbols:
            target_quantity = int(target_quantity_map.get(symbol, 0))
            current_quantity = int(positions.get(symbol, {}).get("quantity", 0))
            buy_quantity = target_quantity - current_quantity
            if buy_quantity <= 0:
                continue

            price = open_prices.get(symbol)
            if pd.isna(price) or price <= 0:
                continue

            fill_price = float(price) * (1.0 + self.slippage_rate)
            buy_quantity = self._cap_buy_quantity_to_cash(fill_price, buy_quantity, cash)
            if buy_quantity <= 0:
                continue

            gross_amount = buy_quantity * fill_price
            commission = self._commission(gross_amount)
            total_cost = gross_amount + commission
            cash -= total_cost

            previous_quantity = int(positions.get(symbol, {}).get("quantity", 0))
            previous_cost = float(positions.get(symbol, {}).get("avg_cost", 0.0))
            new_quantity = previous_quantity + buy_quantity
            new_avg_cost = ((previous_quantity * previous_cost) + total_cost) / new_quantity
            positions[symbol] = {"quantity": new_quantity, "avg_cost": new_avg_cost}

            traded_value += gross_amount
            trades.append(
                self._build_trade_record(
                    trade_date=trade_date,
                    signal_date=signal_date,
                    symbol=symbol,
                    side="BUY",
                    quantity=buy_quantity,
                    price=fill_price,
                    gross_amount=gross_amount,
                    commission=commission,
                    stamp_duty=0.0,
                    realized_pnl=0.0,
                )
            )

        return cash, traded_value, trades

    def _mark_to_market(
        self,
        trade_date: pd.Timestamp,
        positions: dict[str, dict[str, float]],
        close_prices: pd.DataFrame,
        cash: float,
    ) -> tuple[float, list[dict[str, object]]]:
        if not positions:
            return 0.0, []

        prices = close_prices.loc[trade_date]
        holdings_value = 0.0
        rows: list[dict[str, object]] = []

        for symbol, state in positions.items():
            price = prices.get(symbol)
            if pd.isna(price):
                continue
            market_value = state["quantity"] * float(price)
            holdings_value += market_value
            rows.append(
                {
                    "date": trade_date,
                    "symbol": symbol,
                    "quantity": int(state["quantity"]),
                    "close": float(price),
                    "market_value": market_value,
                    "avg_cost": float(state["avg_cost"]),
                }
            )

        portfolio_value = cash + holdings_value
        for row in rows:
            row["weight"] = 0.0 if portfolio_value == 0 else row["market_value"] / portfolio_value
        return holdings_value, rows

    def _cap_buy_quantity_to_cash(self, fill_price: float, desired_quantity: int, cash: float) -> int:
        quantity = self._round_lot_down(desired_quantity)
        while quantity > 0:
            gross_amount = quantity * fill_price
            total_cost = gross_amount + self._commission(gross_amount)
            if total_cost <= cash:
                return quantity
            quantity -= self.lot_size
        return 0

    def _commission(self, gross_amount: float) -> float:
        if gross_amount <= 0:
            return 0.0
        return max(gross_amount * self.commission_rate, self.min_commission)

    def _round_lot_down(self, raw_quantity: float) -> int:
        if raw_quantity <= 0:
            return 0
        return int(raw_quantity // self.lot_size) * self.lot_size

    @staticmethod
    def _build_trade_record(
        trade_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        gross_amount: float,
        commission: float,
        stamp_duty: float,
        realized_pnl: float,
    ) -> dict[str, object]:
        return {
            "trade_date": trade_date,
            "signal_date": signal_date,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "gross_amount": gross_amount,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "net_amount": gross_amount - commission - stamp_duty if side == "SELL" else gross_amount + commission,
            "realized_pnl": realized_pnl,
        }

    @staticmethod
    def _empty_target_snapshot() -> pd.DataFrame:
        return pd.DataFrame(columns=["rebalance_date", "symbol", "target_weight", "score", "rank", "hold_reason"])
