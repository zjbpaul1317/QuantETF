from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_etf.config import load_app_config
from quant_etf.main import run_signal_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate latest ETF rebalance signals for manual trading.")
    parser.add_argument("--config-dir", default="configs", help="Configuration directory path.")
    parser.add_argument(
        "--holdings-csv",
        default=None,
        help="Optional current holdings csv, with at least a symbol column and optional current_weight/market_value.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional local data directory. If set, signal generation will use cached local CSV files.",
    )
    return parser


def _load_holdings(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_csv(path)


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_app_config(args.config_dir, env_prefix=None)
        if args.data_dir:
            config = replace(config, data=replace(config.data, provider="local", raw_dir=Path(args.data_dir).resolve(), file_format="csv"))
        holdings = _load_holdings(args.holdings_csv)
        result = run_signal_pipeline(config, current_holdings=holdings)
    except FileNotFoundError as exc:
        message = str(exc)
        if "returned no ETF history" in message:
            print("Signal generation failed: the selected data provider returned no ETF history.")
        else:
            print("Signal generation failed: no local historical data files were found.")
        print(str(exc))
        return 1
    except RuntimeError as exc:
        print("Signal generation failed while loading market data.")
        print(str(exc))
        if "AkShare failed" in str(exc):
            print("Tip: this usually means the AkShare upstream source was temporarily unavailable. Please retry later.")
        if "Tushare" in str(exc):
            print("Tip: verify your Tushare token configuration before retrying.")
        return 1

    latest_date = pd.to_datetime(result.weekly_signals["date"]).max() if not result.weekly_signals.empty else None
    print("Latest Weekly Rebalance Signals")
    print(f"History rows        : {result.history_rows}")
    print(f"Latest signal date  : {latest_date.date() if latest_date is not None else 'N/A'}")
    print("")
    print("Target Portfolio")
    if result.latest_target_portfolio.empty:
        print("No target positions. Strategy is flat on the latest signal date.")
    else:
        print(result.latest_target_portfolio.to_string(index=False))
    print("")
    print("Manual Trade Plan")
    if result.latest_rebalance_plan.empty:
        print("No trade actions.")
    else:
        trade_plan = result.latest_rebalance_plan.loc[result.latest_rebalance_plan["action"] != "hold"].copy()
        if trade_plan.empty:
            print("No trade actions. Current holdings already match target portfolio.")
        else:
            print(
                trade_plan[
                    ["symbol", "action", "current_weight", "target_weight", "trade_reason", "rank"]
                ].to_string(index=False)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
