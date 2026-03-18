from dataclasses import replace

import numpy as np
import pandas as pd

from quant_etf.backtest import BacktestEngine
from quant_etf.config import load_app_config


def _build_backtest_config():
    config = load_app_config("configs", env_prefix=None)
    return replace(
        config,
        backtest=replace(config.backtest, initial_capital=100_000.0, risk_free_rate=0.0),
        trading=replace(config.trading, lot_size=100),
        cost=replace(config.cost, commission_rate=0.0, stamp_duty_rate=0.0, min_commission=0.0, slippage_rate=0.0),
    )


def _build_backtest_history() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    symbols = {
        "510300.SH": np.linspace(1.00, 1.20, len(dates)),
        "510500.SH": np.linspace(1.00, 0.90, len(dates)),
        "159915.SZ": np.linspace(1.00, 1.15, len(dates)),
    }

    frames: list[pd.DataFrame] = []
    for symbol, close in symbols.items():
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "symbol": symbol,
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": 1_000_000.0,
                "amount": close * 1_000_000,
                "adj_factor": 1.0,
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).set_index(["trade_date", "symbol"]).sort_index()


def _build_target_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rebalance_date": [
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-12"),
                pd.Timestamp("2024-01-12"),
            ],
            "symbol": ["510300.SH", "510500.SH", "510300.SH", "159915.SZ"],
            "target_weight": [0.5, 0.5, 0.5, 0.5],
            "score": [0.10, 0.09, 0.12, 0.11],
            "rank": [1, 2, 1, 2],
            "hold_reason": ["new_buy", "new_buy", "keep_top_buy", "new_buy"],
        }
    )


def test_backtest_engine_executes_on_next_trading_day() -> None:
    config = _build_backtest_config()
    history = _build_backtest_history()
    targets = _build_target_portfolio()

    result = BacktestEngine(config).run(history, targets)

    assert not result.daily_nav.empty
    assert not result.trades.empty
    assert set(result.trades["trade_date"].dt.strftime("%Y-%m-%d")) == {"2024-01-08", "2024-01-15"}
    assert {"annual_return", "max_drawdown", "sharpe_ratio", "win_rate", "turnover_rate"}.issubset(result.metrics.keys())


def test_backtest_engine_costs_reduce_final_nav() -> None:
    base_config = _build_backtest_config()
    cost_config = replace(
        base_config,
        cost=replace(base_config.cost, commission_rate=0.001, stamp_duty_rate=0.0, min_commission=1.0, slippage_rate=0.001),
    )
    history = _build_backtest_history()
    targets = _build_target_portfolio()

    no_cost_result = BacktestEngine(base_config).run(history, targets)
    with_cost_result = BacktestEngine(cost_config).run(history, targets)

    assert with_cost_result.daily_nav["nav"].iloc[-1] < no_cost_result.daily_nav["nav"].iloc[-1]
    assert with_cost_result.trades["commission"].sum() > 0
