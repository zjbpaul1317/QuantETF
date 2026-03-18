from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf.backtest import BacktestResult
from quant_etf.report.analyzer import PerformanceAnalyzer
from quant_etf.report.html_report import HtmlReportBuilder


def export_backtest_bundle(
    output_dir: str | Path,
    result: BacktestResult,
    weekly_signals: pd.DataFrame,
    target_portfolio: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict[str, Path]:
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    analysis = PerformanceAnalyzer(risk_free_rate=risk_free_rate).analyze(result)

    paths = {
        "daily_nav": target_dir / "daily_nav.csv",
        "daily_holdings": target_dir / "daily_holdings.csv",
        "trades": target_dir / "trades.csv",
        "weekly_signals": target_dir / "weekly_signals.csv",
        "target_portfolio": target_dir / "target_portfolio.csv",
        "metrics": target_dir / "metrics.json",
        "analysis": target_dir / "analysis.json",
        "summary": target_dir / "summary.txt",
        "html_report": target_dir / "report.html",
    }

    result.daily_nav.to_csv(paths["daily_nav"], index=False)
    result.daily_holdings.to_csv(paths["daily_holdings"], index=False)
    result.trades.to_csv(paths["trades"], index=False)
    weekly_signals.to_csv(paths["weekly_signals"], index=False)
    target_portfolio.to_csv(paths["target_portfolio"], index=False)
    paths["metrics"].write_text(json.dumps(analysis.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["analysis"].write_text(
        json.dumps({"metrics": analysis.metrics, "conclusions": analysis.conclusions}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["summary"].write_text(_build_summary_text(analysis.metrics, analysis.conclusions), encoding="utf-8")
    HtmlReportBuilder().build(
        result=result,
        metrics=analysis.metrics,
        conclusions=analysis.conclusions,
        output_path=paths["html_report"],
    )
    return paths


def _build_summary_text(metrics: dict[str, float], conclusions: dict[str, str]) -> str:
    lines = [
        "QuantETF Backtest Summary",
        f"Cumulative Return    : {metrics.get('cumulative_return', 0.0):.2%}",
        f"Annual Return : {metrics.get('annual_return', 0.0):.2%}",
        f"Annual Vol    : {metrics.get('annual_volatility', 0.0):.2%}",
        f"Max Drawdown  : {metrics.get('max_drawdown', 0.0):.2%}",
        f"Sharpe Ratio  : {metrics.get('sharpe_ratio', 0.0):.4f}",
        f"Calmar Ratio  : {metrics.get('calmar_ratio', 0.0):.4f}",
        f"Win Rate      : {metrics.get('win_rate', 0.0):.2%}",
        f"Avg Hold Days : {metrics.get('average_holding_period', 0.0):.2f}",
        f"Annual Turnover : {metrics.get('annual_turnover_rate', 0.0):.2%}",
        f"Turnover Rate : {metrics.get('turnover_rate', 0.0):.2%}",
        f"Total Return  : {metrics.get('total_return', 0.0):.2%}",
        "",
        "Conclusions",
        f"- Return Source   : {conclusions.get('return_source', '')}",
        f"- Drawdown Source : {conclusions.get('drawdown_source', '')}",
        f"- Turnover Check  : {conclusions.get('turnover_assessment', '')}",
    ]
    return "\n".join(lines) + "\n"
