from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_etf.backtest import BacktestResult
from quant_etf.backtest.metrics import calculate_metrics


@dataclass(frozen=True)
class AnalysisResult:
    metrics: dict[str, float]
    conclusions: dict[str, str]


class PerformanceAnalyzer:
    """Build extended performance analysis and qualitative conclusions."""

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self.risk_free_rate = risk_free_rate

    def analyze(self, result: BacktestResult) -> AnalysisResult:
        metrics = calculate_metrics(
            result.daily_nav,
            result.trades,
            daily_holdings=result.daily_holdings,
            risk_free_rate=self.risk_free_rate,
        ).to_dict()
        conclusions = self._build_conclusions(result, metrics)
        return AnalysisResult(metrics=metrics, conclusions=conclusions)

    def _build_conclusions(self, result: BacktestResult, metrics: dict[str, float]) -> dict[str, str]:
        nav = result.daily_nav.copy()
        nav["date"] = pd.to_datetime(nav["date"])
        nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0

        strongest_days_share = float(nav["daily_return"].nlargest(min(5, len(nav))).sum()) if not nav.empty else 0.0
        worst_day = nav.loc[nav["drawdown"].idxmin()] if not nav.empty else None
        avg_holding_count = 0.0
        if not result.daily_holdings.empty:
            holding_count = result.daily_holdings.groupby("date")["symbol"].nunique()
            avg_holding_count = float(holding_count.mean())

        if metrics["annual_return"] > 0 and strongest_days_share > 0:
            source = (
                f"收益大概率来自周频动量轮动对上涨ETF的持续跟随，前几次强收益日合计贡献约 "
                f"{strongest_days_share:.2%}，说明净值主要由少数趋势延续阶段驱动。"
            )
        elif metrics["annual_return"] > 0:
            source = "收益更像是由分散持仓下的稳定累积贡献，而不是单次大行情爆发。"
        else:
            source = "收益来源不稳定，说明当前参数下动量筛选对上涨段的捕捉还不够强。"

        if worst_day is None:
            drawdown_reason = "样本不足，暂时无法归因回撤来源。"
        elif avg_holding_count <= 1.5:
            drawdown_reason = (
                f"最大回撤大概率来自持仓过于集中以及趋势反转，最深回撤附近日期为 "
                f"{worst_day['date'].date()}。"
            )
        else:
            drawdown_reason = (
                f"最大回撤更可能来自多只ETF同步走弱时的组合性回撤，最深回撤附近日期为 "
                f"{worst_day['date'].date()}。"
            )

        annual_turnover = metrics.get("annual_turnover_rate", 0.0)
        if annual_turnover > 6:
            turnover_comment = "存在较明显的过度换手风险，建议进一步加强缓冲持有或调低调仓敏感度。"
        elif annual_turnover > 3:
            turnover_comment = "换手率偏高但仍在周频策略常见范围内，可以继续观察交易成本侵蚀。"
        else:
            turnover_comment = "换手率整体可控，缓冲持有机制对降低交易频率已经有一定效果。"

        return {
            "return_source": source,
            "drawdown_source": drawdown_reason,
            "turnover_assessment": turnover_comment,
        }
