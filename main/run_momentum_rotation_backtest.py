from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_etf.config import load_app_config
from quant_etf.momentum_rotation import DEFAULT_ROTATION_SYMBOLS, run_momentum_rotation_backtest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone ETF momentum rotation backtest using annualized slope * R^2 and weekly Top-N equal weights."
        )
    )
    parser.add_argument("--config-dir", default="configs", help="Configuration directory path.")
    parser.add_argument(
        "--output-dir",
        default="data/reports_momentum_rotation",
        help="Output directory for backtest artifacts.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Lookback window N for momentum regression factor.",
    )
    parser.add_argument("--top-n", type=int, default=4, help="Hold Top-N ETFs each rebalance date.")
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated ETF symbols. If omitted, use built-in common ETF universe.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional local data directory. If set, strategy uses local CSV files.",
    )
    parser.add_argument("--start-date", default=None, help="Optional backtest start date, e.g. 2023-01-01")
    parser.add_argument("--end-date", default=None, help="Optional backtest end date, e.g. 2025-12-31")
    parser.add_argument("--initial-capital", type=float, default=None, help="Optional initial capital override.")
    return parser


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_ROTATION_SYMBOLS)
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def main() -> int:
    args = build_parser().parse_args()
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        print("Momentum rotation backtest failed: empty symbol universe.")
        return 1

    config = load_app_config(args.config_dir, env_prefix=None)
    if args.data_dir:
        config = replace(
            config,
            data=replace(
                config.data,
                provider="local",
                raw_dir=Path(args.data_dir).resolve(),
                file_format="csv",
            ),
        )
    if args.start_date or args.end_date or args.initial_capital is not None:
        config = replace(
            config,
            backtest=replace(
                config.backtest,
                start_date=args.start_date or config.backtest.start_date,
                end_date=args.end_date or config.backtest.end_date,
                initial_capital=float(args.initial_capital or config.backtest.initial_capital),
            ),
        )

    try:
        result = run_momentum_rotation_backtest(
            config=config,
            symbols=symbols,
            lookback_days=args.lookback_days,
            top_n=args.top_n,
            output_dir=args.output_dir,
        )
    except FileNotFoundError as exc:
        print("Momentum rotation backtest failed: no local historical data files were found.")
        print(str(exc))
        return 1
    except RuntimeError as exc:
        print("Momentum rotation backtest failed.")
        print(str(exc))
        return 1
    except ValueError as exc:
        print("Momentum rotation parameter error.")
        print(str(exc))
        return 1

    metrics = result.backtest.metrics
    output_root = Path(next(iter(result.output_paths.values()))).parent
    print("Momentum rotation backtest completed")
    print(f"Symbols used    : {', '.join(result.symbols_used)}")
    print(f"History rows    : {result.history_rows}")
    print(f"Rebalance count : {result.target_portfolio['rebalance_date'].nunique() if not result.target_portfolio.empty else 0}")
    print(f"Annual Return   : {metrics['annual_return']:.2%}")
    print(f"Max Drawdown    : {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio    : {metrics['sharpe_ratio']:.4f}")
    print(f"Output Dir      : {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
