from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from quant_etf.backtest import BacktestResult


class HtmlReportBuilder:
    """Generate a lightweight self-contained HTML backtest report."""

    def build(
        self,
        result: BacktestResult,
        metrics: dict[str, float],
        conclusions: dict[str, str],
        output_path: str | Path,
    ) -> Path:
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        nav = result.daily_nav.copy()
        nav["date"] = pd.to_datetime(nav["date"])
        nav["drawdown"] = nav["nav"] / nav["nav"].cummax() - 1.0
        holding_count = self._build_holding_count(result.daily_holdings, nav["date"])

        html_text = self._build_html(nav, holding_count, metrics, conclusions)
        output_file.write_text(html_text, encoding="utf-8")
        return output_file

    @staticmethod
    def _build_holding_count(daily_holdings: pd.DataFrame, nav_dates: pd.Series) -> pd.DataFrame:
        if daily_holdings.empty:
            return pd.DataFrame({"date": nav_dates, "holding_count": [0] * len(nav_dates)})

        counts = (
            daily_holdings.groupby("date")["symbol"]
            .nunique()
            .rename("holding_count")
            .reset_index()
        )
        counts["date"] = pd.to_datetime(counts["date"])
        merged = pd.DataFrame({"date": pd.to_datetime(nav_dates)}).merge(counts, on="date", how="left")
        merged["holding_count"] = merged["holding_count"].fillna(0)
        return merged

    def _build_html(
        self,
        nav: pd.DataFrame,
        holding_count: pd.DataFrame,
        metrics: dict[str, float],
        conclusions: dict[str, str],
    ) -> str:
        nav_svg = self._line_chart_svg(nav["nav"], stroke="#0f766e", fill="#ccfbf1")
        drawdown_svg = self._area_chart_svg(nav["drawdown"], stroke="#b91c1c", fill="#fecaca", baseline=0.0)
        holding_svg = self._line_chart_svg(holding_count["holding_count"], stroke="#1d4ed8", fill="#dbeafe")

        metric_cards = "\n".join(
            [
                self._metric_card("累计收益", f"{metrics.get('cumulative_return', 0.0):.2%}"),
                self._metric_card("年化收益", f"{metrics.get('annual_return', 0.0):.2%}"),
                self._metric_card("年化波动", f"{metrics.get('annual_volatility', 0.0):.2%}"),
                self._metric_card("夏普比率", f"{metrics.get('sharpe_ratio', 0.0):.4f}"),
                self._metric_card("最大回撤", f"{metrics.get('max_drawdown', 0.0):.2%}"),
                self._metric_card("卡玛比率", f"{metrics.get('calmar_ratio', 0.0):.4f}"),
                self._metric_card("胜率", f"{metrics.get('win_rate', 0.0):.2%}"),
                self._metric_card("平均持仓周期", f"{metrics.get('average_holding_period', 0.0):.2f} 天"),
                self._metric_card("年均换手率", f"{metrics.get('annual_turnover_rate', 0.0):.2%}"),
            ]
        )

        conclusions_html = "\n".join(
            [
                f"<li><strong>收益来源：</strong>{html.escape(conclusions.get('return_source', ''))}</li>",
                f"<li><strong>回撤来源：</strong>{html.escape(conclusions.get('drawdown_source', ''))}</li>",
                f"<li><strong>换手评估：</strong>{html.escape(conclusions.get('turnover_assessment', ''))}</li>",
            ]
        )

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QuantETF Backtest Report</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --line: #e2e8f0;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: "SF Pro Text", "PingFang SC", "Helvetica Neue", sans-serif;
      background: linear-gradient(180deg, #eff6ff 0%, var(--bg) 32%);
      color: var(--text);
    }}
    .container {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      background: radial-gradient(circle at top left, #ccfbf1, #ffffff 45%);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 12px 36px rgba(15, 23, 42, 0.06);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }}
    .card .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .card .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .section {{
      margin-top: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    .chart {{
      border-radius: 14px;
      overflow: hidden;
      background: #fcfdff;
      border: 1px solid #e5eef8;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.8;
    }}
    .caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>ETF 周频轮动回测报告</h1>
      <p>报告覆盖净值表现、回撤、持仓数量变化与定性分析结论，便于快速评估策略收益质量与换手特征。</p>
      <div class="grid">{metric_cards}</div>
    </section>

    <section class="section">
      <h2>策略净值曲线</h2>
      <div class="chart">{nav_svg}</div>
      <div class="caption">净值曲线基于每日收盘估值得到。</div>
    </section>

    <section class="section">
      <h2>回撤曲线</h2>
      <div class="chart">{drawdown_svg}</div>
      <div class="caption">回撤以历史净值峰值为基准计算。</div>
    </section>

    <section class="section">
      <h2>持仓数量变化</h2>
      <div class="chart">{holding_svg}</div>
      <div class="caption">反映组合在不同阶段的持仓集中度变化。</div>
    </section>

    <section class="section">
      <h2>策略分析结论</h2>
      <ul>{conclusions_html}</ul>
    </section>
  </div>
</body>
</html>
"""

    @staticmethod
    def _metric_card(label: str, value: str) -> str:
        return (
            '<div class="card">'
            f'<div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>'
            "</div>"
        )

    def _line_chart_svg(self, series: pd.Series, stroke: str, fill: str) -> str:
        width = 960
        height = 280
        padding = 24
        values = pd.Series(series).ffill().fillna(0.0).astype(float).tolist()
        points = self._series_to_points(values, width, height, padding)
        polygon = f"{padding},{height - padding} " + points + f" {width - padding},{height - padding}"
        return (
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
            f'<polygon points="{polygon}" fill="{fill}" opacity="0.75"></polygon>'
            f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>'
            "</svg>"
        )

    def _area_chart_svg(self, series: pd.Series, stroke: str, fill: str, baseline: float) -> str:
        width = 960
        height = 280
        padding = 24
        values = pd.Series(series).fillna(0.0).astype(float).tolist()
        points = self._series_to_points(values, width, height, padding, baseline=baseline)
        baseline_y = self._value_to_y(baseline, values, height, padding, baseline=baseline)
        polygon = f"{padding},{baseline_y} " + points + f" {width - padding},{baseline_y}"
        return (
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
            f'<line x1="{padding}" y1="{baseline_y}" x2="{width - padding}" y2="{baseline_y}" stroke="#cbd5e1" stroke-width="1.5"></line>'
            f'<polygon points="{polygon}" fill="{fill}" opacity="0.8"></polygon>'
            f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></polyline>'
            "</svg>"
        )

    def _series_to_points(
        self,
        values: list[float],
        width: int,
        height: int,
        padding: int,
        baseline: float | None = None,
    ) -> str:
        if not values:
            y = height / 2
            return f"{padding},{y} {width - padding},{y}"

        n = max(len(values) - 1, 1)
        points: list[str] = []
        for idx, value in enumerate(values):
            x = padding + (width - 2 * padding) * idx / n
            y = self._value_to_y(value, values, height, padding, baseline=baseline)
            points.append(f"{x:.2f},{y:.2f}")
        return " ".join(points)

    @staticmethod
    def _value_to_y(
        value: float,
        values: list[float],
        height: int,
        padding: int,
        baseline: float | None = None,
    ) -> float:
        min_value = min(values if baseline is None else values + [baseline])
        max_value = max(values if baseline is None else values + [baseline])
        if max_value == min_value:
            return height / 2
        ratio = (value - min_value) / (max_value - min_value)
        return height - padding - ratio * (height - 2 * padding)
