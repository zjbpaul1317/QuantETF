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
from quant_etf.data import ETFDataRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch ETF history from the configured provider and cache it locally.")
    parser.add_argument("--config-dir", default="configs", help="Configuration directory path.")
    parser.add_argument("--start-date", default=None, help="Optional override start date, e.g. 2023-01-01")
    parser.add_argument("--end-date", default=None, help="Optional override end date, e.g. 2025-12-31")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["easyquotation", "akshare"],
        help="Optional remote data provider override. Default: config.data.provider",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for cached CSV files. Default: config.data.raw_dir",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one combined CSV file in addition to per-symbol files.",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Ignore local cache and rebuild the requested date range from the remote provider.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_app_config(args.config_dir, env_prefix=None)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else config.data.raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_config = replace(
        config,
        backtest=replace(
            config.backtest,
            start_date=args.start_date or config.backtest.start_date,
            end_date=args.end_date or config.backtest.end_date,
        ),
        data=replace(config.data, provider=args.provider or config.data.provider, raw_dir=output_dir),
    )

    try:
        history = ETFDataRepository(fetch_config).load_history(
            start_date=fetch_config.backtest.start_date,
            end_date=fetch_config.backtest.end_date,
            force_reload=args.force_reload,
        )
    except FileNotFoundError as exc:
        print("Cache build failed: no ETF history was returned for the requested symbols.")
        print(str(exc))
        return 1
    except RuntimeError as exc:
        print("Cache build failed while downloading market data.")
        print(str(exc))
        if "AkShare failed" in str(exc):
            print("Tip: AkShare upstream may be temporarily unstable. Please retry later or shorten the date range.")
        if "easyquotation" in str(exc):
            print("Tip: easyquotation only refreshes live snapshots; older days must already exist in your local cache.")
        return 1

    frame = history.reset_index().copy()
    export_columns = [
        "trade_date",
        "symbol",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "volume",
        "amount",
        "adj_factor",
    ]
    available_columns = [column for column in export_columns if column in frame.columns]
    export_frame = frame[available_columns].rename(
        columns={
            "raw_open": "open",
            "raw_high": "high",
            "raw_low": "low",
            "raw_close": "close",
        }
    )

    symbol_count = export_frame["symbol"].nunique()
    if fetch_config.data.provider != "easyquotation":
        for symbol, snapshot in export_frame.groupby("symbol", sort=True):
            path = output_dir / f"{symbol}.csv"
            snapshot.to_csv(path, index=False)

    if args.combined:
        combined_path = output_dir / "etf_daily.csv"
        export_frame.to_csv(combined_path, index=False)

    print("ETF history cache completed")
    print(f"Output directory : {output_dir}")
    print(f"Symbols cached   : {symbol_count}")
    print(f"Rows cached      : {len(export_frame)}")
    print("Next step        : run local backtest with --data-dir and --file-format csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
