from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _path(value: str | Path, project_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class AppSettings:
    name: str = "quant-etf"
    timezone: str = "Asia/Shanghai"
    project_root: Path = Path(".")


@dataclass(frozen=True)
class DataConfig:
    provider: str = "local"
    adjustment: str = "qfq"
    price_frequency: str = "1d"
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    report_dir: Path = Path("data/reports")
    file_format: str = "auto"
    file_pattern: str = "{symbol}.{ext}"
    combined_file_name: str = "etf_daily"
    symbol_column: str = "symbol"
    date_column: str = "trade_date"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"
    amount_column: str = "amount"
    adj_factor_column: str = "adj_factor"
    use_adjusted_price: bool = True
    fill_missing_ohlc_from_close: bool = True
    incremental_update_lookback_days: int = 5


@dataclass(frozen=True)
class UniverseConfig:
    symbols: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    min_listed_days: int = 120
    liquidity_lookback: int = 20
    min_avg_turnover: float = 100_000_000.0
    min_avg_volume: float = 1_000_000.0


@dataclass(frozen=True)
class MarketRegimeConfig:
    enabled: bool = True
    hs300_symbol: str = "510300.SH"
    zz1000_symbol: str = "512100.SH"
    ma_window: int = 60
    risk_off_action: str = "flat"
    risk_off_exposure: float = 0.0


@dataclass(frozen=True)
class StrategyConfig:
    rebalance_frequency: str = "weekly"
    signal_weekday: int = 4
    execution_delay_days: int = 1
    execution_timing: str = "next_open"
    lookback_windows: tuple[int, int, int] = (20, 60, 120)
    score_weights: tuple[float, float, float] = (0.4, 0.4, 0.2)
    ma_window: int = 60
    score_threshold: float = 0.0
    buy_top_n: int = 3
    hold_buffer_n: int = 5
    enable_buffer_hold: bool = True
    stoploss_ma_ratio: float = 0.98
    weight_method: str = "equal"


@dataclass(frozen=True)
class TradingConfig:
    lot_size: int = 100
    cash_reserve_ratio: float = 0.01
    max_position_weight: float = 0.34
    allow_suspended_sell: bool = False
    allow_limit_up_buy: bool = False
    allow_limit_down_sell: bool = False
    min_trade_amount: float = 1_000.0


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 1_000_000.0
    benchmark_symbol: str = "510300.SH"
    trade_price_field: str = "open"
    risk_free_rate: float = 0.02


@dataclass(frozen=True)
class CostConfig:
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0005
    stamp_duty_rate: float = 0.0
    min_commission: float = 5.0


@dataclass(frozen=True)
class LiveConfig:
    enabled: bool = False
    dry_run: bool = True
    broker: str = "qmt"
    account_id: str | None = None
    signal_output_path: Path = Path("data/reports/live_signals.csv")
    order_output_path: Path = Path("data/reports/live_orders.csv")
    max_retries: int = 3


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path = Path("data/reports")
    export_html: bool = True
    export_csv: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    file: Path = Path("data/reports/quant_etf.log")
    rotation: str | None = None
    retention: str | None = None


@dataclass(frozen=True)
class AppConfig:
    app: AppSettings
    data: DataConfig
    universe: UniverseConfig
    market_regime: MarketRegimeConfig
    strategy: StrategyConfig
    trading: TradingConfig
    backtest: BacktestConfig
    cost: CostConfig
    live: LiveConfig
    report: ReportConfig
    logging: LoggingConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any], project_root: str | Path | None = None) -> "AppConfig":
        root = Path(project_root or raw.get("app", {}).get("project_root") or ".").resolve()

        app_raw = raw.get("app", {})
        data_raw = raw.get("data", {})
        universe_raw = raw.get("universe", {})
        market_regime_raw = raw.get("market_regime", {})
        strategy_raw = raw.get("strategy", {})
        trading_raw = raw.get("trading", {})
        backtest_raw = raw.get("backtest", {})
        cost_raw = raw.get("cost", {})
        live_raw = raw.get("live", {})
        report_raw = raw.get("report", {})
        logging_raw = raw.get("logging", {})

        return cls(
            app=AppSettings(
                name=app_raw.get("name", "quant-etf"),
                timezone=app_raw.get("timezone", "Asia/Shanghai"),
                project_root=root,
            ),
            data=DataConfig(
                provider=data_raw.get("provider", "local"),
                adjustment=data_raw.get("adjustment", "qfq"),
                price_frequency=data_raw.get("price_frequency", "1d"),
                raw_dir=_path(data_raw.get("raw_dir", "data/raw"), root),
                processed_dir=_path(data_raw.get("processed_dir", "data/processed"), root),
                cache_dir=_path(data_raw.get("cache_dir", "data/cache"), root),
                report_dir=_path(data_raw.get("report_dir", "data/reports"), root),
                file_format=data_raw.get("file_format", "auto"),
                file_pattern=data_raw.get("file_pattern", "{symbol}.{ext}"),
                combined_file_name=data_raw.get("combined_file_name", "etf_daily"),
                symbol_column=data_raw.get("symbol_column", "symbol"),
                date_column=data_raw.get("date_column", "trade_date"),
                open_column=data_raw.get("open_column", "open"),
                high_column=data_raw.get("high_column", "high"),
                low_column=data_raw.get("low_column", "low"),
                close_column=data_raw.get("close_column", "close"),
                volume_column=data_raw.get("volume_column", "volume"),
                amount_column=data_raw.get("amount_column", "amount"),
                adj_factor_column=data_raw.get("adj_factor_column", "adj_factor"),
                use_adjusted_price=_bool(data_raw.get("use_adjusted_price"), True),
                fill_missing_ohlc_from_close=_bool(data_raw.get("fill_missing_ohlc_from_close"), True),
                incremental_update_lookback_days=int(data_raw.get("incremental_update_lookback_days", 5)),
            ),
            universe=UniverseConfig(
                symbols=list(universe_raw.get("symbols", [])),
                blacklist=list(universe_raw.get("blacklist", [])),
                min_listed_days=int(universe_raw.get("min_listed_days", 120)),
                liquidity_lookback=int(universe_raw.get("liquidity_lookback", 20)),
                min_avg_turnover=float(universe_raw.get("min_avg_turnover", 100_000_000.0)),
                min_avg_volume=float(universe_raw.get("min_avg_volume", 1_000_000.0)),
            ),
            market_regime=MarketRegimeConfig(
                enabled=_bool(market_regime_raw.get("enabled"), True),
                hs300_symbol=market_regime_raw.get("hs300_symbol", "510300.SH"),
                zz1000_symbol=market_regime_raw.get("zz1000_symbol", "512100.SH"),
                ma_window=int(market_regime_raw.get("ma_window", 60)),
                risk_off_action=market_regime_raw.get("risk_off_action", "flat"),
                risk_off_exposure=float(market_regime_raw.get("risk_off_exposure", 0.0)),
            ),
            strategy=StrategyConfig(
                rebalance_frequency=strategy_raw.get("rebalance_frequency", "weekly"),
                signal_weekday=int(strategy_raw.get("signal_weekday", 4)),
                execution_delay_days=int(strategy_raw.get("execution_delay_days", 1)),
                execution_timing=strategy_raw.get("execution_timing", "next_open"),
                lookback_windows=tuple(int(v) for v in strategy_raw.get("lookback_windows", [20, 60, 120])),
                score_weights=tuple(float(v) for v in strategy_raw.get("score_weights", [0.4, 0.4, 0.2])),
                ma_window=int(strategy_raw.get("ma_window", 60)),
                score_threshold=float(strategy_raw.get("score_threshold", 0.0)),
                buy_top_n=int(strategy_raw.get("buy_top_n", 3)),
                hold_buffer_n=int(strategy_raw.get("hold_buffer_n", 5)),
                enable_buffer_hold=_bool(strategy_raw.get("enable_buffer_hold"), True),
                stoploss_ma_ratio=float(strategy_raw.get("stoploss_ma_ratio", 0.98)),
                weight_method=strategy_raw.get("weight_method", "equal"),
            ),
            trading=TradingConfig(
                lot_size=int(trading_raw.get("lot_size", 100)),
                cash_reserve_ratio=float(trading_raw.get("cash_reserve_ratio", 0.01)),
                max_position_weight=float(trading_raw.get("max_position_weight", 0.34)),
                allow_suspended_sell=bool(trading_raw.get("allow_suspended_sell", False)),
                allow_limit_up_buy=bool(trading_raw.get("allow_limit_up_buy", False)),
                allow_limit_down_sell=bool(trading_raw.get("allow_limit_down_sell", False)),
                min_trade_amount=float(trading_raw.get("min_trade_amount", 1_000.0)),
            ),
            backtest=BacktestConfig(
                start_date=str(backtest_raw.get("start_date", "2023-01-01")),
                end_date=str(backtest_raw.get("end_date", "2025-12-31")),
                initial_capital=float(backtest_raw.get("initial_capital", 1_000_000.0)),
                benchmark_symbol=backtest_raw.get("benchmark_symbol", "510300.SH"),
                trade_price_field=backtest_raw.get("trade_price_field", "open"),
                risk_free_rate=float(backtest_raw.get("risk_free_rate", 0.02)),
            ),
            cost=CostConfig(
                commission_rate=float(cost_raw.get("commission_rate", 0.0003)),
                slippage_rate=float(cost_raw.get("slippage_rate", 0.0005)),
                stamp_duty_rate=float(cost_raw.get("stamp_duty_rate", 0.0)),
                min_commission=float(cost_raw.get("min_commission", 5.0)),
            ),
            live=LiveConfig(
                enabled=_bool(live_raw.get("enabled"), False),
                dry_run=_bool(live_raw.get("dry_run"), True),
                broker=live_raw.get("broker", "qmt"),
                account_id=live_raw.get("account_id"),
                signal_output_path=_path(live_raw.get("signal_output_path", "data/reports/live_signals.csv"), root),
                order_output_path=_path(live_raw.get("order_output_path", "data/reports/live_orders.csv"), root),
                max_retries=int(live_raw.get("max_retries", 3)),
            ),
            report=ReportConfig(
                output_dir=_path(report_raw.get("output_dir", "data/reports"), root),
                export_html=_bool(report_raw.get("export_html"), True),
                export_csv=_bool(report_raw.get("export_csv"), True),
            ),
            logging=LoggingConfig(
                level=logging_raw.get("level", "INFO"),
                file=_path(logging_raw.get("file", "data/reports/quant_etf.log"), root),
                rotation=logging_raw.get("rotation"),
                retention=logging_raw.get("retention"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for section in ("app", "data", "live", "report", "logging"):
            for key, value in payload[section].items():
                if isinstance(value, Path):
                    payload[section][key] = str(value)
        return payload
