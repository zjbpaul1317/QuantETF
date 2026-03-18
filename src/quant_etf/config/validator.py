from __future__ import annotations

from datetime import datetime

from .schema import AppConfig


class ConfigValidationError(ValueError):
    """Raised when the application configuration is invalid."""


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def validate_app_config(config: AppConfig) -> None:
    _ensure(bool(config.universe.symbols), "universe.symbols must not be empty")
    _ensure(config.data.provider in {"local", "akshare", "tushare"}, "data.provider must be local, akshare or tushare")
    _ensure(config.data.adjustment in {"none", "qfq", "hfq"}, "data.adjustment must be one of none, qfq, hfq")
    _ensure(config.data.file_format in {"auto", "csv", "parquet"}, "data.file_format must be auto, csv or parquet")
    _ensure("{symbol}" in config.data.file_pattern and "{ext}" in config.data.file_pattern,
            "data.file_pattern must contain {symbol} and {ext}")
    _ensure(config.strategy.rebalance_frequency == "weekly", "strategy.rebalance_frequency must be 'weekly'")
    _ensure(config.strategy.signal_weekday == 4, "strategy.signal_weekday must be 4 for Friday close signals")
    _ensure(len(config.strategy.lookback_windows) == 3, "strategy.lookback_windows must contain 3 windows")
    _ensure(len(config.strategy.score_weights) == 3, "strategy.score_weights must contain 3 weights")
    _ensure(abs(sum(config.strategy.score_weights) - 1.0) < 1e-8, "strategy.score_weights must sum to 1.0")
    _ensure(all(window > 0 for window in config.strategy.lookback_windows), "lookback windows must be positive")
    _ensure(config.strategy.lookback_windows == tuple(sorted(config.strategy.lookback_windows)),
            "strategy.lookback_windows must be sorted ascending")
    _ensure(config.universe.min_listed_days >= max(config.strategy.lookback_windows),
            "universe.min_listed_days must cover the longest lookback window")
    _ensure(config.strategy.buy_top_n > 0, "strategy.buy_top_n must be positive")
    _ensure(config.strategy.hold_buffer_n >= config.strategy.buy_top_n,
            "strategy.hold_buffer_n must be >= strategy.buy_top_n")
    if not config.strategy.enable_buffer_hold:
        _ensure(
            config.strategy.hold_buffer_n >= config.strategy.buy_top_n,
            "strategy.hold_buffer_n must still be >= strategy.buy_top_n when buffer hold is disabled",
        )
    _ensure(0 < config.strategy.stoploss_ma_ratio <= 1.5, "strategy.stoploss_ma_ratio must be in (0, 1.5]")
    _ensure(config.strategy.weight_method == "equal", "only equal weighting is supported for now")
    _ensure(config.strategy.ma_window > 0, "strategy.ma_window must be positive")
    _ensure(config.market_regime.ma_window > 0, "market_regime.ma_window must be positive")
    _ensure(config.market_regime.risk_off_action in {"flat", "reduce"},
            "market_regime.risk_off_action must be 'flat' or 'reduce'")
    _ensure(0.0 <= config.market_regime.risk_off_exposure <= 1.0,
            "market_regime.risk_off_exposure must be within [0, 1]")
    if config.market_regime.risk_off_action == "flat":
        _ensure(config.market_regime.risk_off_exposure == 0.0,
                "risk_off_exposure must be 0.0 when risk_off_action is 'flat'")
    _ensure(config.universe.liquidity_lookback > 0, "universe.liquidity_lookback must be positive")
    _ensure(config.universe.min_avg_turnover >= 0, "universe.min_avg_turnover must be non-negative")
    _ensure(config.universe.min_avg_volume >= 0, "universe.min_avg_volume must be non-negative")
    _ensure(config.backtest.initial_capital > 0, "backtest.initial_capital must be positive")
    _ensure(config.trading.lot_size > 0, "trading.lot_size must be positive")
    _ensure(0.0 <= config.trading.cash_reserve_ratio < 1.0, "trading.cash_reserve_ratio must be in [0, 1)")
    _ensure(0.0 < config.trading.max_position_weight <= 1.0, "trading.max_position_weight must be in (0, 1]")
    _ensure(config.trading.min_trade_amount >= 0.0, "trading.min_trade_amount must be non-negative")
    _ensure(config.cost.commission_rate >= 0.0, "cost.commission_rate must be non-negative")
    _ensure(config.cost.slippage_rate >= 0.0, "cost.slippage_rate must be non-negative")
    _ensure(config.cost.stamp_duty_rate >= 0.0, "cost.stamp_duty_rate must be non-negative")
    _ensure(config.cost.min_commission >= 0.0, "cost.min_commission must be non-negative")
    _ensure(config.live.max_retries >= 0, "live.max_retries must be non-negative")

    start = datetime.fromisoformat(config.backtest.start_date)
    end = datetime.fromisoformat(config.backtest.end_date)
    _ensure(start <= end, "backtest.start_date must be earlier than or equal to backtest.end_date")
