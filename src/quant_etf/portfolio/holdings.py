from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_holdings(current_holdings: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize current holdings into a standard weight-based representation."""
    if current_holdings is None or current_holdings.empty:
        return pd.DataFrame(columns=["symbol", "current_weight", "quantity", "market_value"])

    frame = current_holdings.copy()
    if "symbol" not in frame.columns:
        raise ValueError("current_holdings must contain a 'symbol' column")

    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    for column in ("current_weight", "quantity", "market_value"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["current_weight"].isna().all():
        if frame["market_value"].notna().any() and frame["market_value"].sum() > 0:
            total_market_value = frame["market_value"].sum()
            frame["current_weight"] = frame["market_value"] / total_market_value
        else:
            frame["current_weight"] = 1.0 / len(frame)

    normalized = frame.groupby("symbol", as_index=False).agg(
        current_weight=("current_weight", "sum"),
        quantity=("quantity", "sum"),
        market_value=("market_value", "sum"),
    )
    return normalized.sort_values("symbol").reset_index(drop=True)
