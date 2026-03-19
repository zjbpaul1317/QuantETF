"""Signal generation module for ETF rotation strategy."""

from .features import IndicatorCalculator
from .momentum import MomentumScorer
from .regime import MarketRegimeAssessor
from .ranking import SignalRanker
from .signal_engine import SignalEngine, SignalResult

__all__ = [
    "IndicatorCalculator",
    "MarketRegimeAssessor",
    "MomentumScorer",
    "SignalEngine",
    "SignalRanker",
    "SignalResult",
]
