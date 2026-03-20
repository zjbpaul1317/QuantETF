from dataclasses import replace

import numpy as np
import pandas as pd

from quant_etf.config import load_app_config
from quant_etf.signal import IndicatorCalculator, MarketRegimeAssessor, SignalEngine, SignalRanker


def _build_signal_test_config():
    config = load_app_config("configs", env_prefix=None)
    return replace(
        config,
        strategy=replace(
            config.strategy,
            ma_window=60,
            lookback_windows=(20, 60, 120),
            score_weights=(0.4, 0.4, 0.2),
            score_threshold=0.0,
            buy_top_n=3,
            signal_weekday=4,
        ),
        universe=replace(
            config.universe,
            symbols=["510300.SH", "159915.SZ", "510500.SH", "512100.SH", "512880.SH"],
            min_listed_days=120,
            liquidity_lookback=20,
            min_avg_turnover=10_000_000.0,
        ),
    )


def _build_synthetic_history() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", "2024-07-05")
    symbols = {
        "510300.SH": {"start": 3.0, "slope": 0.0030, "volume": 12_000_000},
        "159915.SZ": {"start": 1.0, "slope": 0.0024, "volume": 20_000_000},
        "510500.SH": {"start": 5.0, "slope": 0.0016, "volume": 8_000_000},
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

    history = pd.concat(frames, ignore_index=True)
    return history.set_index(["trade_date", "symbol"]).sort_index()


def test_indicator_calculator_builds_required_columns() -> None:
    config = _build_signal_test_config()
    history = _build_synthetic_history()

    features = IndicatorCalculator(config).calculate(history)

    assert {"r20", "r60", "r120", "ma60", "volatility_20", "avg_turnover"}.issubset(features.columns)
    latest_date = features.index.get_level_values("trade_date").max()
    latest_slice = features.xs(latest_date, level="trade_date")
    assert latest_slice["r120"].notna().all()
    assert latest_slice["ma60"].notna().all()


def test_signal_engine_generates_friday_rotation_signals() -> None:
    config = _build_signal_test_config()
    history = _build_synthetic_history()

    result = SignalEngine(config).generate(history)

    weekly = result.weekly_signals
    assert not weekly.empty
    assert set(["date", "symbol", "close", "ma60", "r20", "r60", "r120", "score", "rank", "eligible", "buy_signal"]).issubset(
        weekly.columns
    )
    assert weekly["date"].dt.weekday.eq(4).all()

    latest_date = weekly["date"].max()
    latest_buy = weekly.loc[(weekly["date"] == latest_date) & (weekly["buy_signal"])].sort_values("rank")
    assert latest_buy["symbol"].tolist() == ["510300.SH", "159915.SZ", "510500.SH"]
    assert latest_buy["rank"].tolist() == [1, 2, 3]
    assert weekly.loc[(weekly["date"] == latest_date) & (weekly["symbol"] == "512100.SH"), "eligible"].iloc[0]
    assert weekly.loc[(weekly["date"] == latest_date) & (weekly["symbol"] == "512100.SH"), "secondary_candidate"].iloc[0]
    assert not weekly.loc[(weekly["date"] == latest_date) & (weekly["symbol"] == "512100.SH"), "buy_signal"].iloc[0]
    assert not weekly.loc[(weekly["date"] == latest_date) & (weekly["symbol"] == "512880.SH"), "eligible"].iloc[0]


def test_market_regime_assessor_marks_risk_off_when_both_anchors_are_weak() -> None:
    config = _build_signal_test_config()
    weekly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-07-05")] * 4,
            "symbol": ["510300.SH", "512100.SH", "159915.SZ", "510500.SH"],
            "close": [1.00, 0.95, 1.10, 1.05],
            "ma60": [1.01, 0.97, 1.00, 1.00],
            "score": [-0.01, -0.02, 0.10, 0.08],
            "rank": [1, 2, 3, 4],
            "eligible": [False, False, True, True],
            "buy_signal": [False, False, True, True],
        }
    )

    tagged = MarketRegimeAssessor(config).attach(weekly)

    assert not tagged["market_regime_on"].iloc[0]
    assert tagged["risk_off"].iloc[0]


def test_market_regime_assessor_requires_two_strong_weeks_to_turn_risk_on() -> None:
    config = _build_signal_test_config()
    weekly = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-07-05")] * 2 + [pd.Timestamp("2024-07-12")] * 2,
            "symbol": ["510300.SH", "512100.SH", "510300.SH", "512100.SH"],
            "close": [1.02, 0.99, 1.03, 1.01],
            "ma60": [1.00, 0.98, 1.00, 0.99],
            "score": [0.01, 0.01, 0.02, 0.02],
            "rank": [1, 2, 1, 2],
            "eligible": [True, True, True, True],
            "buy_signal": [True, True, True, True],
        }
    )

    tagged = MarketRegimeAssessor(config).attach(weekly)
    regime_by_date = tagged.groupby("date")["market_regime_on"].first()
    signal_by_date = tagged.groupby("date")["regime_signal"].first()

    assert not regime_by_date.loc[pd.Timestamp("2024-07-05")]
    assert regime_by_date.loc[pd.Timestamp("2024-07-12")]
    assert signal_by_date.loc[pd.Timestamp("2024-07-05")] == "risk_on_setup"
    assert signal_by_date.loc[pd.Timestamp("2024-07-12")] == "confirmed_risk_on"


def test_signal_ranker_uses_secondary_candidates_when_primary_pool_is_too_small() -> None:
    config = _build_signal_test_config()
    trade_date = pd.Timestamp("2024-07-05")
    features = pd.DataFrame(
        {
            "close": [1.10, 1.05, 0.99, 1.02],
            "ma60": [1.00, 1.00, 1.00, 1.00],
            "score": [0.10, 0.08, -0.01, 0.20],
            "listed_days": [200, 200, 200, 80],
            "avg_turnover": [2e7, 2e7, 2e7, 2e7],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (trade_date, "510300.SH"),
                (trade_date, "159915.SZ"),
                (trade_date, "512100.SH"),
                (trade_date, "588080.SH"),
            ],
            names=["trade_date", "symbol"],
        ),
    )

    ranked = SignalRanker(config).rank(features)
    snapshot = ranked.xs(trade_date, level="trade_date").sort_values("rank", na_position="last")

    assert snapshot.loc["510300.SH", "primary_candidate"]
    assert snapshot.loc["159915.SZ", "primary_candidate"]
    assert snapshot.loc["512100.SH", "secondary_candidate"]
    assert not bool(snapshot.loc["588080.SH", "eligible"])
    assert snapshot.loc["512100.SH", "buy_signal"]
    assert snapshot.loc["512100.SH", "rank"] == 3
