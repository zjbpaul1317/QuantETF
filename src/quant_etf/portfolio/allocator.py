from __future__ import annotations

import numpy as np
import pandas as pd

from quant_etf.config.schema import AppConfig


class EqualWeightAllocator:
    """Allocate target weights with optional volatility scaling."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def allocate(
        self,
        selected_symbols: list[str],
        total_exposure: float | None = None,
        signal_snapshot: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        unique_symbols = list(dict.fromkeys(symbol.upper() for symbol in selected_symbols))
        if not unique_symbols:
            return pd.DataFrame(columns=["symbol", "target_weight"])

        if total_exposure is None:
            total_exposure = 1.0 - self.config.trading.cash_reserve_ratio

        if self.config.strategy.weight_method == "inverse_volatility":
            weights = self._inverse_volatility_weights(unique_symbols, total_exposure, signal_snapshot)
        else:
            weights = np.full(len(unique_symbols), total_exposure / len(unique_symbols), dtype="float64")
        weights = self._cap_weights(weights, total_exposure)
        return pd.DataFrame(
            {
                "symbol": unique_symbols,
                "target_weight": weights.tolist(),
            }
        )

    def _inverse_volatility_weights(
        self,
        selected_symbols: list[str],
        total_exposure: float,
        signal_snapshot: pd.DataFrame | None,
    ) -> np.ndarray:
        if signal_snapshot is None or "volatility_20" not in signal_snapshot.columns:
            raise ValueError("signal_snapshot with 'volatility_20' is required for inverse_volatility weighting")

        snapshot = signal_snapshot.copy()
        snapshot["symbol"] = snapshot["symbol"].astype(str).str.upper()
        vol_series = (
            snapshot.drop_duplicates(subset=["symbol"], keep="last")
            .set_index("symbol")["volatility_20"]
            .reindex(selected_symbols)
        )
        vol = pd.to_numeric(vol_series, errors="coerce").fillna(vol_series.dropna().median()).fillna(1.0)
        inverse_vol = 1.0 / vol.clip(lower=1e-6)
        weight_sum = float(inverse_vol.sum())
        if weight_sum <= 0:
            return np.full(len(selected_symbols), total_exposure / len(selected_symbols), dtype="float64")
        return (inverse_vol / weight_sum * total_exposure).to_numpy(dtype="float64")

    def _cap_weights(self, raw_weights: np.ndarray, total_exposure: float) -> np.ndarray:
        max_weight = float(self.config.trading.max_position_weight)
        if len(raw_weights) == 1:
            return np.array([min(float(raw_weights[0]), max_weight)], dtype="float64")

        capped = np.minimum(raw_weights.astype("float64"), max_weight)
        for _ in range(len(capped)):
            remaining = total_exposure - float(capped.sum())
            if remaining <= 1e-10:
                break

            active = capped < (max_weight - 1e-10)
            if not active.any():
                break

            active_weights = raw_weights[active]
            active_sum = float(active_weights.sum())
            if active_sum <= 0:
                capped[active] += remaining / int(active.sum())
            else:
                capped[active] += remaining * (active_weights / active_sum)
            capped = np.minimum(capped, max_weight)

        total = float(capped.sum())
        if total <= 0:
            return np.zeros_like(capped)
        return capped * (total_exposure / total)
