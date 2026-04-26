from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant_etf.backtest import BacktestEngine, BacktestResult
from quant_etf.config.schema import AppConfig
from quant_etf.data import ETFDataRepository
from quant_etf.report import export_backtest_bundle

# Default universe covers broad market, overseas equity, gold and major sectors.
DEFAULT_ROTATION_SYMBOLS: tuple[str, ...] = (
    "513100.SH",  # Nasdaq 100 ETF
    "518880.SH",  # Gold ETF
    "159915.SZ",  # ChiNext ETF
    "159928.SZ",  # Consumer ETF
    "515000.SH",  # Tech ETF
    "512010.SH",  # Pharma ETF
    "516160.SH",  # New energy ETF
    "512880.SH",  # Securities ETF
    "512660.SH",  # Military ETF
    "588000.SH",  # STAR 50 ETF
    "510300.SH",  # CSI 300 ETF
    "510500.SH",  # CSI 500 ETF
)


@dataclass(frozen=True)
class MomentumRotationRunResult:
    symbols_used: list[str]
    history_rows: int
    weekly_signals: pd.DataFrame
    target_portfolio: pd.DataFrame
    backtest: BacktestResult
    output_paths: dict[str, Path]


def run_momentum_rotation_backtest(
    config: AppConfig,
    symbols: Iterable[str] | None = None,
    lookback_days: int = 90,
    top_n: int = 4,
    output_dir: str | Path | None = None,
) -> MomentumRotationRunResult:
    selected_symbols = [symbol.upper() for symbol in (symbols or DEFAULT_ROTATION_SYMBOLS)]
    if lookback_days < 20:
        raise ValueError("lookback_days must be >= 20")
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    history = ETFDataRepository(config).load_history(symbols=selected_symbols)
    weekly_signals, target_portfolio = build_weekly_momentum_targets(
        history=history,
        lookback_days=lookback_days,
        top_n=top_n,
    )
    if target_portfolio.empty:
        raise RuntimeError("No valid rebalance targets were generated. Please check symbols or lookback window.")

    rebalance_dates = sorted(target_portfolio["rebalance_date"].dropna().unique())
    backtest_result = BacktestEngine(config).run(
        history=history,
        target_portfolio=target_portfolio,
        rebalance_dates=rebalance_dates,
    )
    export_dir = Path(output_dir).resolve() if output_dir is not None else config.report.output_dir
    output_paths = export_backtest_bundle(
        output_dir=export_dir,
        result=backtest_result,
        weekly_signals=weekly_signals,
        target_portfolio=target_portfolio,
        risk_free_rate=config.backtest.risk_free_rate,
    )
    return MomentumRotationRunResult(
        symbols_used=sorted(history.reset_index()["symbol"].astype(str).str.upper().unique().tolist()),
        history_rows=len(history),
        weekly_signals=weekly_signals,
        target_portfolio=target_portfolio,
        backtest=backtest_result,
        output_paths=output_paths,
    )


def build_weekly_momentum_targets(
    history: pd.DataFrame,
    lookback_days: int,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_prices = _build_close_matrix(history)
    rebalance_dates = _weekly_rebalance_dates(close_prices.index)

    signal_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []

    for rebalance_date in rebalance_dates:
        snapshot = close_prices.loc[:rebalance_date]
        momentum = {
            symbol: calculate_momentum_score(snapshot[symbol], lookback_days=lookback_days)
            for symbol in close_prices.columns
        }
        ranked = (
            pd.Series(momentum, name="momentum_score")
            .dropna()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "symbol"})
        )
        if ranked.empty:
            continue

        ranked["date"] = pd.Timestamp(rebalance_date)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        ranked["buy_signal"] = ranked["rank"] <= top_n
        selected_count = int(ranked["buy_signal"].sum())
        ranked["target_weight"] = np.where(ranked["buy_signal"], 1.0 / max(selected_count, 1), 0.0)
        signal_rows.extend(ranked.to_dict(orient="records"))

        selected = ranked.loc[ranked["buy_signal"]].copy()
        if selected.empty:
            continue
        target_rows.extend(
            {
                "rebalance_date": row.date,
                "symbol": row.symbol,
                "target_weight": float(row.target_weight),
                "rank": int(row.rank),
                "momentum_score": float(row.momentum_score),
            }
            for row in selected.itertuples(index=False)
        )

    weekly_signals = pd.DataFrame(signal_rows)
    target_portfolio = pd.DataFrame(target_rows)
    if not weekly_signals.empty:
        weekly_signals["date"] = pd.to_datetime(weekly_signals["date"])
        weekly_signals = weekly_signals.sort_values(["date", "rank", "symbol"]).reset_index(drop=True)
    if not target_portfolio.empty:
        target_portfolio["rebalance_date"] = pd.to_datetime(target_portfolio["rebalance_date"])
        target_portfolio = target_portfolio.sort_values(["rebalance_date", "rank", "symbol"]).reset_index(drop=True)
    return weekly_signals, target_portfolio


def calculate_momentum_score(close_series: pd.Series, lookback_days: int) -> float:
    """Momentum factor = annualized slope * R^2 using linear regression on log close."""
    window = pd.Series(close_series).dropna().tail(lookback_days)
    if len(window) < lookback_days:
        return float("nan")

    y = np.log(window.to_numpy(dtype=float))
    x = np.arange(len(y), dtype=float)

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_var = float(np.sum(x_centered**2))
    if x_var == 0:
        return float("nan")

    slope = float(np.sum(x_centered * y_centered) / x_var)
    fitted = y.mean() + slope * x_centered
    ss_tot = float(np.sum(y_centered**2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((y - fitted) ** 2))
    r_squared = max(0.0, 1.0 - ss_res / ss_tot)

    annualized_slope = float(np.exp(slope * 252.0) - 1.0)
    return annualized_slope * r_squared


def _build_close_matrix(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    required = {"trade_date", "symbol", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"history is missing required columns: {', '.join(missing)}")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    close_prices = frame.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    return close_prices.ffill()


def _weekly_rebalance_dates(trading_dates: pd.Index) -> list[pd.Timestamp]:
    if len(trading_dates) == 0:
        return []
    frame = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    week_last = frame.groupby("week", sort=True)["date"].max()
    return week_last.tolist()
