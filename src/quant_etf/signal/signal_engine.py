from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_etf.config.schema import AppConfig

from .features import IndicatorCalculator
from .momentum import MomentumScorer
from .ranking import SignalRanker


@dataclass(frozen=True)
class SignalResult:
    daily_signals: pd.DataFrame
    weekly_signals: pd.DataFrame


class SignalEngine:
    """Generate daily and weekly ETF rotation signals."""

    DAILY_SIGNAL_COLUMNS = [
        "date",
        "symbol",
        "close",
        "ma60",
        "r20",
        "r60",
        "r120",
        "volatility_20",
        "avg_turnover",
        "listed_days",
        "score",
        "rank",
        "eligible",
        "hold_signal",
        "buy_signal",
    ]

    def __init__(
        self,
        config: AppConfig,
        indicator_calculator: IndicatorCalculator | None = None,
        scorer: MomentumScorer | None = None,
        ranker: SignalRanker | None = None,
    ) -> None:
        self.config = config
        self.indicator_calculator = indicator_calculator or IndicatorCalculator(config)
        self.scorer = scorer or MomentumScorer(config)
        self.ranker = ranker or SignalRanker(config)

    def generate(self, history: pd.DataFrame) -> SignalResult:
        daily_signals = self.generate_daily_signals(history)
        weekly_signals = self.generate_weekly_signals(daily_signals)
        return SignalResult(daily_signals=daily_signals, weekly_signals=weekly_signals)

    def generate_daily_signals(self, history: pd.DataFrame) -> pd.DataFrame:
        features = self.indicator_calculator.calculate(history)
        scored = self.scorer.score(features)
        ranked = self.ranker.rank(scored)

        frame = ranked.reset_index().rename(
            columns={
                "trade_date": "date",
                f"ma{self.config.strategy.ma_window}": "ma60",
            }
        )
        frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)

        extra_columns = [
            "passed_listing",
            "passed_liquidity",
            "passed_trend",
            "passed_score",
            "primary_candidate",
            "secondary_candidate",
        ]
        ordered_columns = self.DAILY_SIGNAL_COLUMNS + extra_columns
        existing_columns = [column for column in ordered_columns if column in frame.columns]
        return frame[existing_columns].copy()

    def generate_weekly_signals(self, daily_signals: pd.DataFrame) -> pd.DataFrame:
        frame = daily_signals.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        weekly = frame.loc[frame["date"].dt.weekday == self.config.strategy.signal_weekday].copy()
        weekly = weekly.sort_values(["date", "rank", "symbol"], na_position="last").reset_index(drop=True)
        return weekly
