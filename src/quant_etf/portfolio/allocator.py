from __future__ import annotations

import pandas as pd

from quant_etf.config.schema import AppConfig


class EqualWeightAllocator:
    """Allocate target weights equally across selected symbols."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def allocate(self, selected_symbols: list[str], total_exposure: float | None = None) -> pd.DataFrame:
        unique_symbols = list(dict.fromkeys(symbol.upper() for symbol in selected_symbols))
        if not unique_symbols:
            return pd.DataFrame(columns=["symbol", "target_weight"])

        if total_exposure is None:
            total_exposure = 1.0 - self.config.trading.cash_reserve_ratio

        base_weight = total_exposure / len(unique_symbols)
        target_weight = min(base_weight, self.config.trading.max_position_weight)
        return pd.DataFrame(
            {
                "symbol": unique_symbols,
                "target_weight": [target_weight] * len(unique_symbols),
            }
        )
