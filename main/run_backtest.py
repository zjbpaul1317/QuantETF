from __future__ import annotations

import argparse
from pathlib import Path
import sys
from dataclasses import replace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_etf.config import load_app_config
from quant_etf.main import run_backtest_pipeline, run_backtest_with_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QuantETF end-to-end backtest pipeline.")
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Configuration directory path. Default: configs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for backtest artifacts. Default: config.report.output_dir",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional local history data directory. If set, overrides config.data.raw_dir",
    )
    parser.add_argument(
        "--file-format",
        default=None,
        choices=["csv", "parquet", "auto"],
        help="Optional local history file format override.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["akshare", "easyquotation", "local"],
        help="Optional data provider override.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.data_dir or args.file_format or args.provider:
            config = load_app_config(args.config_dir, env_prefix=None)
            data_config = config.data
            if args.data_dir:
                data_config = replace(data_config, raw_dir=Path(args.data_dir).resolve(), provider="local")
            if args.file_format:
                data_config = replace(data_config, file_format=args.file_format)
            if args.provider:
                data_config = replace(data_config, provider=args.provider)
            config = replace(config, data=data_config)
            result = run_backtest_with_config(config, output_dir=args.output_dir)
        else:
            result = run_backtest_pipeline(
                config_dir=args.config_dir,
                output_dir=args.output_dir,
            )
    except FileNotFoundError as exc:
        message = str(exc)
        if "returned no ETF history" in message:
            print("Backtest failed: the selected data provider returned no ETF history.")
        else:
            print("Backtest failed: historical ETF files were not found.")
        print(str(exc))
        if "returned no ETF history" not in message:
            print("Tip: run with --data-dir /path/to/your/etf/files and optional --file-format csv|parquet")
        return 1
    except RuntimeError as exc:
        print("Backtest failed while loading market data.")
        print(str(exc))
        if "AkShare failed" in str(exc):
            print("Tip: this usually means the AkShare upstream source was temporarily unavailable. Please retry later.")
        if "easyquotation" in str(exc) or "EasyQuotation" in str(exc):
            print("Tip: easyquotation cannot backfill missing historical ETF bars; seed the local cache first or use --provider local.")
        return 1

    metrics = result.backtest.metrics
    print("Backtest completed")
    print(f"History rows   : {result.history_rows}")
    print(f"Weekly signals : {len(result.weekly_signals)}")
    print(f"Targets        : {len(result.target_portfolio)}")
    print(f"Annual Return  : {metrics['annual_return']:.2%}")
    print(f"Max Drawdown   : {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio   : {metrics['sharpe_ratio']:.4f}")
    print(f"Win Rate       : {metrics['win_rate']:.2%}")
    print(f"Turnover Rate  : {metrics['turnover_rate']:.2%}")
    print(f"Output Dir     : {Path(next(iter(result.output_paths.values()))).parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
