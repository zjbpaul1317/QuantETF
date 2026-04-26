import numpy as np
import pandas as pd

from quant_etf.momentum_rotation import build_weekly_momentum_targets, calculate_momentum_score


def _build_synthetic_history() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", "2024-06-28")
    symbols = {
        "A.SH": 0.0030,
        "B.SH": 0.0024,
        "C.SH": 0.0018,
        "D.SH": 0.0012,
        "E.SH": 0.0007,
        "F.SH": -0.0006,
    }

    frames: list[pd.DataFrame] = []
    for symbol, slope in symbols.items():
        close = 1.0 * np.power(1 + slope, np.arange(len(dates)))
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "symbol": symbol,
                    "open": close * 0.999,
                    "high": close * 1.001,
                    "low": close * 0.998,
                    "close": close,
                }
            )
        )
    history = pd.concat(frames, ignore_index=True)
    return history.set_index(["trade_date", "symbol"]).sort_index()


def test_calculate_momentum_score_is_higher_for_clean_uptrend() -> None:
    dates = pd.bdate_range("2024-01-01", periods=120)
    clean_uptrend = pd.Series(1.0 * np.power(1.0025, np.arange(len(dates))), index=dates)

    rng = np.random.default_rng(7)
    noisy_uptrend = clean_uptrend * (1.0 + rng.normal(0.0, 0.01, size=len(dates)))

    clean_score = calculate_momentum_score(clean_uptrend, lookback_days=90)
    noisy_score = calculate_momentum_score(noisy_uptrend, lookback_days=90)

    assert clean_score > 0
    assert clean_score > noisy_score


def test_build_weekly_momentum_targets_selects_top4_equal_weight() -> None:
    history = _build_synthetic_history()

    weekly_signals, target_portfolio = build_weekly_momentum_targets(
        history=history,
        lookback_days=60,
        top_n=4,
    )

    assert not weekly_signals.empty
    assert not target_portfolio.empty
    assert {"date", "symbol", "momentum_score", "rank", "buy_signal", "target_weight"}.issubset(weekly_signals.columns)
    assert {"rebalance_date", "symbol", "target_weight", "rank"}.issubset(target_portfolio.columns)

    grouped = target_portfolio.groupby("rebalance_date")
    for _, snapshot in grouped:
        assert len(snapshot) == 4
        assert np.isclose(snapshot["target_weight"].sum(), 1.0)
        assert np.allclose(snapshot["target_weight"].to_numpy(), np.array([0.25, 0.25, 0.25, 0.25]))

    latest = target_portfolio[target_portfolio["rebalance_date"] == target_portfolio["rebalance_date"].max()]
    assert latest.sort_values("rank")["symbol"].tolist() == ["A.SH", "B.SH", "C.SH", "D.SH"]
