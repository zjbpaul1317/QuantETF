from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from quant_etf.config import load_app_config
from quant_etf.main import run_backtest_with_config
from quant_etf.portfolio import TargetPortfolioBuilder


def _build_pipeline_config(tmp_path: Path):
    config = load_app_config("configs", env_prefix=None)
    return replace(
        config,
        data=replace(
            config.data,
            raw_dir=Path("tests/fixtures/pipeline_data").resolve(),
            provider="local",
            file_format="csv",
            use_adjusted_price=False,
        ),
        universe=replace(
            config.universe,
            symbols=["510300.SH", "159915.SZ", "510500.SH", "512100.SH"],
            min_listed_days=5,
            liquidity_lookback=5,
            min_avg_turnover=100_000.0,
        ),
        strategy=replace(
            config.strategy,
            lookback_windows=(3, 5, 8),
            score_weights=(0.4, 0.4, 0.2),
            ma_window=5,
            buy_top_n=2,
            hold_buffer_n=3,
            signal_weekday=4,
        ),
        backtest=replace(
            config.backtest,
            start_date="2024-01-01",
            end_date="2024-03-31",
            initial_capital=100_000.0,
            risk_free_rate=0.0,
        ),
        cost=replace(
            config.cost,
            commission_rate=0.0,
            stamp_duty_rate=0.0,
            min_commission=0.0,
            slippage_rate=0.0,
        ),
        report=replace(config.report, output_dir=tmp_path.resolve()),
    )


def test_target_builder_build_all_tracks_rebalance_dates() -> None:
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-12")],
            "symbol": ["A.SH", "B.SH", "A.SH", "C.SH"],
            "score": [0.10, 0.09, 0.11, 0.10],
            "rank": [1, 2, 1, 2],
            "buy_signal": [True, True, True, True],
            "close": [1.0, 1.0, 1.0, 1.0],
            "ma60": [0.9, 0.9, 0.9, 0.9],
        }
    )
    config = load_app_config("configs", env_prefix=None)
    config = replace(
        config,
        strategy=replace(
            config.strategy,
            buy_top_n=2,
            hold_buffer_n=3,
            enable_buffer_hold=True,
            rebalance_interval_weeks=1,
            weight_method="equal",
        ),
        universe=replace(config.universe, min_listed_days=1, min_avg_turnover=0.0),
    )
    signals["hold_signal"] = True
    signals["eligible"] = True
    signals["market_regime_on"] = True
    signals["risk_off"] = False

    targets = TargetPortfolioBuilder(config).build_all(signals)

    assert sorted(targets["rebalance_date"].dt.strftime("%Y-%m-%d").unique().tolist()) == ["2024-01-05", "2024-01-12"]


def test_target_builder_build_all_respects_rebalance_interval() -> None:
    signals = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-12"),
                pd.Timestamp("2024-01-12"),
                pd.Timestamp("2024-01-19"),
                pd.Timestamp("2024-01-19"),
            ],
            "symbol": ["A.SH", "B.SH", "A.SH", "C.SH", "B.SH", "C.SH"],
            "score": [0.10, 0.09, 0.11, 0.10, 0.12, 0.11],
            "rank": [1, 2, 1, 2, 1, 2],
            "buy_signal": [True, True, True, True, True, True],
            "close": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ma60": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            "volatility_20": [0.10, 0.15, 0.10, 0.15, 0.10, 0.15],
            "hold_signal": [True, True, True, True, True, True],
            "eligible": [True, True, True, True, True, True],
            "market_regime_on": [True, True, True, True, True, True],
            "risk_off": [False, False, False, False, False, False],
        }
    )
    config = load_app_config("configs", env_prefix=None)
    config = replace(
        config,
        strategy=replace(
            config.strategy,
            buy_top_n=2,
            hold_buffer_n=3,
            enable_buffer_hold=True,
            rebalance_interval_weeks=2,
            weight_method="equal",
        ),
        universe=replace(config.universe, min_listed_days=1, min_avg_turnover=0.0),
    )

    targets = TargetPortfolioBuilder(config).build_all(signals)

    assert sorted(targets["rebalance_date"].dt.strftime("%Y-%m-%d").unique().tolist()) == ["2024-01-05", "2024-01-19"]


def test_run_backtest_with_config_exports_full_bundle(tmp_path: Path) -> None:
    config = _build_pipeline_config(tmp_path)

    result = run_backtest_with_config(config, output_dir=tmp_path)

    assert result.history_rows > 0
    assert not result.weekly_signals.empty
    assert not result.backtest.daily_nav.empty
    assert (tmp_path / "daily_nav.csv").exists()
    assert (tmp_path / "daily_holdings.csv").exists()
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "weekly_signals.csv").exists()
    assert (tmp_path / "target_portfolio.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "analysis.json").exists()
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "summary.txt").exists()

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert {
        "cumulative_return",
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "average_holding_period",
        "annual_turnover_rate",
    }.issubset(metrics.keys())
