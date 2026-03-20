from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_etf.backtest import BacktestEngine, BacktestResult
from quant_etf.config import load_app_config
from quant_etf.config.schema import AppConfig
from quant_etf.data import ETFDataRepository
from quant_etf.portfolio import RebalancePlanner, TargetPortfolioBuilder
from quant_etf.report import export_backtest_bundle
from quant_etf.signal import MarketRegimeAssessor, SignalEngine


@dataclass(frozen=True)
class PipelineResult:
    config: AppConfig
    history_rows: int
    weekly_signals: object
    target_portfolio: object
    backtest: BacktestResult
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class SignalPipelineResult:
    config: AppConfig
    history_rows: int
    daily_signals: object
    weekly_signals: object
    latest_target_portfolio: object
    latest_rebalance_plan: object


def run_backtest_with_config(
    config: AppConfig,
    output_dir: str | Path | None = None,
) -> PipelineResult:
    history = ETFDataRepository(config).load_history()
    signal_result = SignalEngine(config).generate(history)
    weekly_signals = MarketRegimeAssessor(config).attach(signal_result.weekly_signals)
    target_portfolio = TargetPortfolioBuilder(config).build_all(weekly_signals)

    rebalance_dates = (
        sorted(target_portfolio["rebalance_date"].dropna().unique())
        if not target_portfolio.empty
        else []
    )
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

    return PipelineResult(
        config=config,
        history_rows=len(history),
        weekly_signals=weekly_signals,
        target_portfolio=target_portfolio,
        backtest=backtest_result,
        output_paths=output_paths,
    )


def run_backtest_pipeline(
    config_dir: str | Path = "configs",
    output_dir: str | Path | None = None,
) -> PipelineResult:
    config = load_app_config(config_dir, env_prefix=None)
    return run_backtest_with_config(config, output_dir=output_dir)


def run_signal_pipeline(
    config: AppConfig,
    current_holdings: object | None = None,
) -> SignalPipelineResult:
    history = ETFDataRepository(config).load_history()
    signal_result = SignalEngine(config).generate(history)
    weekly_signals = MarketRegimeAssessor(config).attach(signal_result.weekly_signals)
    latest_target = TargetPortfolioBuilder(config).build(
        weekly_signals,
        current_holdings=current_holdings,
    )
    latest_plan = RebalancePlanner(config).plan(
        weekly_signals,
        current_holdings=current_holdings,
    ).rebalance_plan
    return SignalPipelineResult(
        config=config,
        history_rows=len(history),
        daily_signals=signal_result.daily_signals,
        weekly_signals=weekly_signals,
        latest_target_portfolio=latest_target,
        latest_rebalance_plan=latest_plan,
    )
