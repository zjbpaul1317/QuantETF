from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_etf.config import load_app_config
from quant_etf.filter import HoldingExitFilter
from quant_etf.portfolio import RebalancePlanner, TargetPortfolioBuilder
from quant_etf.signal import MarketRegimeAssessor, SignalEngine


def _build_portfolio_test_config():
    config = load_app_config("configs", env_prefix=None)
    return replace(
        config,
        strategy=replace(
            config.strategy,
            ma_window=60,
            lookback_windows=(20, 60, 120),
            score_weights=(0.4, 0.4, 0.2),
            bias_ma_window=60,
            bias_regression_window=20,
            slope_window=20,
            efficiency_window=20,
            buy_top_n=3,
            hold_buffer_n=6,
            enable_buffer_hold=True,
            stoploss_ma_ratio=0.98,
            signal_weekday=4,
            weight_method="equal",
        ),
        universe=replace(
            config.universe,
            symbols=["510300.SH", "159915.SZ", "510500.SH", "516160.SH", "512100.SH", "512880.SH"],
            min_listed_days=120,
            liquidity_lookback=20,
            min_avg_turnover=10_000_000.0,
        ),
        market_regime=replace(config.market_regime, enabled=False),
    )


def _build_portfolio_history() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", "2024-07-05")
    symbols = {
        "510300.SH": {"start": 3.0, "slope": 0.0030, "volume": 12_000_000},
        "159915.SZ": {"start": 1.0, "slope": 0.0024, "volume": 20_000_000},
        "510500.SH": {"start": 5.0, "slope": 0.0016, "volume": 8_000_000},
        "516160.SH": {"start": 1.5, "slope": 0.0011, "volume": 9_000_000},
        "512100.SH": {"start": 0.9, "slope": -0.0010, "volume": 18_000_000},
        "512880.SH": {"start": 1.2, "slope": 0.0028, "volume": 30_000},
    }

    frames: list[pd.DataFrame] = []
    for symbol, params in symbols.items():
        close = params["start"] * np.power(1 + params["slope"], np.arange(len(dates)))
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "symbol": symbol,
                "open": close * 0.998,
                "high": close * 1.002,
                "low": close * 0.997,
                "close": close,
                "volume": float(params["volume"]),
                "amount": close * params["volume"],
                "adj_factor": 1.0,
            }
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).set_index(["trade_date", "symbol"]).sort_index()


def test_exit_filter_flags_sell_conditions() -> None:
    config = _build_portfolio_test_config()
    exit_filter = HoldingExitFilter(config)
    signal_table = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-06-28")] * 3 + [pd.Timestamp("2024-07-05")] * 3,
            "symbol": ["AAA.SH", "BBB.SH", "CCC.SH", "AAA.SH", "BBB.SH", "CCC.SH"],
            "close": [0.99, 1.10, 1.20, 0.95, 1.10, 1.20],
            "ma60": [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
            "score": [0.10, -0.01, 0.08, 0.10, -0.01, 0.08],
            "rank": [2, 3, 7, 2, 3, 7],
            "eligible": [True, True, True, True, True, True],
            "hold_signal": [True, True, False, True, True, False],
            "market_regime_on": [True, True, True, True, True, True],
        }
    )
    holdings = pd.DataFrame({"symbol": ["AAA.SH", "BBB.SH", "CCC.SH"], "current_weight": [0.33, 0.33, 0.34]})

    result = exit_filter.apply(signal_table, holdings, as_of_date="2024-07-05")
    reason_map = result.set_index("symbol")["exit_reason"].to_dict()
    should_sell_map = result.set_index("symbol")["should_sell"].to_dict()

    assert should_sell_map["AAA.SH"]
    assert reason_map["AAA.SH"] == "stoploss"
    assert should_sell_map["BBB.SH"]
    assert "score_non_positive_2w" in reason_map["BBB.SH"]
    assert should_sell_map["CCC.SH"]
    assert "rank_out_of_buffer_2w" in reason_map["CCC.SH"]


def test_rebalance_planner_replaces_weak_incumbent_and_fills_open_slots() -> None:
    config = _build_portfolio_test_config()
    history = _build_portfolio_history()
    weekly_signals = MarketRegimeAssessor(config).attach(SignalEngine(config).generate(history).weekly_signals)
    holdings = pd.DataFrame(
        {
            "symbol": ["510300.SH", "516160.SH", "512100.SH"],
            "current_weight": [1 / 3, 1 / 3, 1 / 3],
        }
    )

    result = RebalancePlanner(config).plan(weekly_signals, holdings)

    target_symbols = result.target_portfolio["symbol"].tolist()
    action_map = result.rebalance_plan.set_index("symbol")["action"].to_dict()
    reason_map = result.rebalance_plan.set_index("symbol")["trade_reason"].to_dict()

    assert target_symbols == ["510300.SH", "159915.SZ", "510500.SH"]
    assert action_map["512100.SH"] == "sell"
    assert "score_non_positive_2w" in reason_map["512100.SH"]
    assert action_map["516160.SH"] == "sell"
    assert "score_non_positive_2w" in reason_map["516160.SH"]
    assert action_map["159915.SZ"] == "buy"
    assert reason_map["159915.SZ"] == "new_buy"
    assert action_map["510500.SH"] == "buy"
    assert reason_map["510500.SH"] == "new_buy"
    assert np.isclose(result.target_portfolio["target_weight"].sum(), 0.9933333333333334)


def test_target_portfolio_builder_supports_risk_off_and_disable_buffer_hold() -> None:
    config = replace(
        _build_portfolio_test_config(),
        strategy=replace(_build_portfolio_test_config().strategy, enable_buffer_hold=False),
        market_regime=replace(_build_portfolio_test_config().market_regime, risk_off_action="reduce", risk_off_exposure=0.3),
    )
    history = _build_portfolio_history()
    weekly_signals = MarketRegimeAssessor(config).attach(SignalEngine(config).generate(history).weekly_signals)
    holdings = pd.DataFrame(
        {
            "symbol": ["510300.SH", "516160.SH", "512100.SH"],
            "current_weight": [1 / 3, 1 / 3, 1 / 3],
        }
    )

    target = TargetPortfolioBuilder(config).build(weekly_signals, current_holdings=holdings, market_regime_on=False)

    assert target["symbol"].tolist() == ["510300.SH", "159915.SZ", "510500.SH"]
    assert np.isclose(target["target_weight"].sum(), 0.3)
    assert set(target["hold_reason"]) == {"risk_off_scaled"}


def test_target_portfolio_builder_requires_market_regime_column() -> None:
    config = _build_portfolio_test_config()
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-07-05"), pd.Timestamp("2024-07-05")],
            "symbol": ["510300.SH", "159915.SZ"],
            "score": [0.1, 0.09],
            "rank": [1, 2],
            "buy_signal": [True, True],
            "close": [1.0, 1.0],
            "ma60": [0.9, 0.9],
        }
    )

    with pytest.raises(ValueError, match="market_regime_on"):
        TargetPortfolioBuilder(config).build(signals)


def test_target_portfolio_builder_caps_single_symbol_exposure_at_half() -> None:
    config = _build_portfolio_test_config()
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-07-05")],
            "symbol": ["510300.SH"],
            "score": [0.1],
            "rank": [1],
            "buy_signal": [True],
            "close": [1.0],
            "ma60": [0.9],
            "market_regime_on": [True],
        }
    )

    target = TargetPortfolioBuilder(config).build(signals)

    assert target["symbol"].tolist() == ["510300.SH"]
    assert np.isclose(target["target_weight"].sum(), 0.5)


def test_target_portfolio_builder_fully_allocates_two_symbol_portfolio() -> None:
    config = _build_portfolio_test_config()
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-07-05"), pd.Timestamp("2024-07-05")],
            "symbol": ["510300.SH", "159915.SZ"],
            "score": [0.1, 0.09],
            "rank": [1, 2],
            "buy_signal": [True, True],
            "close": [1.0, 1.0],
            "ma60": [0.9, 0.9],
            "market_regime_on": [True, True],
        }
    )

    target = TargetPortfolioBuilder(config).build(signals)

    assert target["symbol"].tolist() == ["510300.SH", "159915.SZ"]
    assert np.isclose(target["target_weight"].sum(), 0.99)
    assert np.isclose(target["target_weight"].iloc[0], 0.495)
    assert np.isclose(target["target_weight"].iloc[1], 0.495)


def test_target_portfolio_builder_supports_inverse_volatility_weights() -> None:
    config = replace(
        _build_portfolio_test_config(),
        strategy=replace(_build_portfolio_test_config().strategy, weight_method="inverse_volatility"),
    )
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-07-05"), pd.Timestamp("2024-07-05")],
            "symbol": ["510300.SH", "159915.SZ"],
            "score": [0.1, 0.09],
            "rank": [1, 2],
            "buy_signal": [True, True],
            "close": [1.0, 1.0],
            "ma60": [0.9, 0.9],
            "volatility_20": [0.10, 0.20],
            "market_regime_on": [True, True],
        }
    )

    target = TargetPortfolioBuilder(config).build(signals)

    assert np.isclose(target["target_weight"].sum(), 0.99)
    assert target.set_index("symbol").loc["510300.SH", "target_weight"] > target.set_index("symbol").loc["159915.SZ", "target_weight"]


def test_target_portfolio_builder_keeps_incumbent_when_score_edge_is_small() -> None:
    config = replace(
        _build_portfolio_test_config(),
        strategy=replace(_build_portfolio_test_config().strategy, buy_top_n=1, hold_buffer_n=2),
    )
    holdings = pd.DataFrame({"symbol": ["AAA.SH"], "current_weight": [0.5]})
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-06-28"), pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-05"), pd.Timestamp("2024-07-05")],
            "symbol": ["AAA.SH", "BBB.SH", "AAA.SH", "BBB.SH"],
            "score": [0.011, 0.015, 0.011, 0.015],
            "rank": [3, 1, 3, 1],
            "buy_signal": [False, True, False, True],
            "close": [1.00, 1.00, 1.00, 1.00],
            "ma60": [0.99, 0.99, 0.99, 0.99],
            "market_regime_on": [True, True, True, True],
        }
    )

    target = TargetPortfolioBuilder(config).build(signals, current_holdings=holdings, as_of_date="2024-07-05")

    assert target["symbol"].tolist() == ["AAA.SH"]
    assert target.loc[0, "hold_reason"] == "keep_small_edge"
