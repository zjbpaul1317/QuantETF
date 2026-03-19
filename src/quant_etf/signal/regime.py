from __future__ import annotations

import pandas as pd

from quant_etf.config.schema import AppConfig


class MarketRegimeAssessor:
    """Attach market regime state to weekly signal snapshots."""

    REQUIRED_COLUMNS = {"date", "symbol", "close", "score"}

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.anchor_symbols = (
            str(config.market_regime.hs300_symbol).strip().upper(),
            str(config.market_regime.zz1000_symbol).strip().upper(),
        )
        self.require_positive_score = bool(config.market_regime.require_positive_score)
        self.score_threshold = max(float(config.strategy.score_threshold), 0.0)
        self.on_confirm_weeks = int(config.market_regime.on_confirm_weeks)
        self.off_confirm_weeks = int(config.market_regime.off_confirm_weeks)
        self.ma_column = f"ma{config.market_regime.ma_window}"

    def attach(self, weekly_signals: pd.DataFrame) -> pd.DataFrame:
        frame = weekly_signals.copy()
        if frame.empty:
            frame["market_regime_on"] = pd.Series(dtype="bool")
            frame["risk_off"] = pd.Series(dtype="bool")
            return frame

        self._ensure_required_columns(frame)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()

        if not self.config.market_regime.enabled:
            frame["market_regime_on"] = True
            frame["risk_off"] = False
            frame["regime_signal"] = "disabled"
            return frame

        ma_column = self._resolve_ma_column(frame)
        regime_rows: list[pd.DataFrame] = []
        market_regime_on = False
        strong_streak = 0
        weak_streak = 0
        for trade_date, snapshot in frame.groupby("date", sort=True):
            status_map = self._evaluate_snapshot(snapshot, ma_column=ma_column, trade_date=trade_date)
            all_strong = all(status["strong"] for status in status_map.values())
            all_weak = all(status["weak"] for status in status_map.values())

            if all_strong:
                strong_streak += 1
                weak_streak = 0
            elif all_weak:
                weak_streak += 1
                strong_streak = 0
            else:
                strong_streak = 0
                weak_streak = 0

            regime_signal = "mixed"
            if all_strong and strong_streak >= self.on_confirm_weeks:
                market_regime_on = True
                regime_signal = "confirmed_risk_on"
            elif all_weak and weak_streak >= self.off_confirm_weeks:
                market_regime_on = False
                regime_signal = "confirmed_risk_off"
            elif all_strong:
                regime_signal = "risk_on_setup"
            elif all_weak:
                regime_signal = "risk_off_setup"

            tagged = snapshot.copy()
            tagged["market_regime_on"] = market_regime_on
            tagged["risk_off"] = not market_regime_on
            tagged["regime_signal"] = regime_signal
            for symbol, status in status_map.items():
                prefix = symbol.lower().replace(".", "_")
                tagged[f"{prefix}_strong"] = status["strong"]
                tagged[f"{prefix}_weak"] = status["weak"]
            regime_rows.append(tagged)

        return pd.concat(regime_rows, ignore_index=True, sort=False)

    def _ensure_required_columns(self, frame: pd.DataFrame) -> None:
        missing = sorted(self.REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"weekly_signals is missing required regime columns: {missing_text}")

    def _resolve_ma_column(self, frame: pd.DataFrame) -> str:
        if self.ma_column in frame.columns:
            return self.ma_column
        if "ma60" in frame.columns:
            return "ma60"
        raise ValueError(
            f"weekly_signals must contain '{self.ma_column}' or 'ma60' before attaching market regime"
        )

    def _evaluate_snapshot(
        self,
        snapshot: pd.DataFrame,
        ma_column: str,
        trade_date: pd.Timestamp,
    ) -> dict[str, dict[str, bool]]:
        status_map: dict[str, dict[str, bool]] = {}
        for symbol in self.anchor_symbols:
            anchor = snapshot.loc[snapshot["symbol"] == symbol]
            if anchor.empty:
                raise ValueError(
                    f"Missing market regime anchor {symbol} in weekly_signals for {trade_date.date()}"
                )

            row = anchor.iloc[-1]
            ma_value = pd.to_numeric(pd.Series([row[ma_column]]), errors="coerce").iloc[0]
            close_value = pd.to_numeric(pd.Series([row["close"]]), errors="coerce").iloc[0]
            score_value = pd.to_numeric(pd.Series([row["score"]]), errors="coerce").iloc[0]

            is_trending = pd.notna(close_value) and pd.notna(ma_value) and float(close_value) > float(ma_value)
            if self.require_positive_score:
                has_positive_score = pd.notna(score_value) and float(score_value) > self.score_threshold
                lacks_score = pd.isna(score_value) or float(score_value) <= self.score_threshold
            else:
                has_positive_score = True
                lacks_score = False

            strong = bool(is_trending and has_positive_score)
            weak = bool((not is_trending) or lacks_score)
            status_map[symbol] = {"strong": strong, "weak": weak}

        return status_map
