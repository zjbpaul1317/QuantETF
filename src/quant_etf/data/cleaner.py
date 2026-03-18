from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_etf.config.schema import DataConfig


@dataclass(frozen=True)
class NormalizedColumnSet:
    symbol: str = "symbol"
    trade_date: str = "trade_date"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    amount: str = "amount"
    adj_factor: str = "adj_factor"


class ETFDataCleaner:
    """Normalize raw ETF daily bars into a standard schema."""

    COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
        "symbol": ("symbol", "code", "ts_code", "基金代码", "证券代码", "股票代码"),
        "trade_date": ("trade_date", "date", "datetime", "交易日期", "日期"),
        "open": ("open", "open_price", "开盘", "开盘价"),
        "high": ("high", "high_price", "最高", "最高价"),
        "low": ("low", "low_price", "最低", "最低价"),
        "close": ("close", "close_price", "收盘", "收盘价"),
        "volume": ("volume", "vol", "成交量", "volume_shares"),
        "amount": ("amount", "turnover", "成交额", "成交金额"),
        "adj_factor": ("adj_factor", "复权因子", "factor"),
    }

    REQUIRED_COLUMNS = ("symbol", "trade_date", "open", "high", "low", "close", "volume")

    def __init__(self, config: DataConfig, normalized_columns: NormalizedColumnSet | None = None) -> None:
        self.config = config
        self.columns = normalized_columns or NormalizedColumnSet()

    def clean(self, raw_bars: pd.DataFrame) -> pd.DataFrame:
        if raw_bars.empty:
            return self._empty_frame()

        frame = raw_bars.copy()
        frame = self._rename_columns(frame)
        self._ensure_required_columns(frame)
        frame = self._normalize_symbols(frame)
        frame = self._normalize_dates(frame)
        frame = self._normalize_numeric_fields(frame)
        frame = self._fill_missing_ohlc(frame)
        frame = self._derive_amount(frame)
        frame = self._drop_invalid_rows(frame)
        frame = self._deduplicate(frame)
        frame = self._sort(frame)
        return frame.reset_index(drop=True)

    def _rename_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        existing_lower_map = {column.lower(): column for column in frame.columns if isinstance(column, str)}
        columns_to_drop: set[str] = set()
        for target, aliases in self.COLUMN_ALIASES.items():
            present = []
            for alias in aliases:
                existing = existing_lower_map.get(alias.lower())
                if existing is not None and existing not in present:
                    present.append(existing)
            if not present:
                continue

            ordered = [target] + [column for column in present if column != target] if target in frame.columns else present
            frame[target] = frame[ordered].bfill(axis=1).iloc[:, 0]
            columns_to_drop.update(column for column in ordered if column != target)

        if columns_to_drop:
            frame = frame.drop(columns=sorted(columns_to_drop), errors="ignore")
        return frame

    def _ensure_required_columns(self, frame: pd.DataFrame) -> None:
        missing = [column for column in self.REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required ETF bar columns: {missing_str}")

    def _normalize_symbols(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["symbol"] = frame["symbol"].astype(str).map(self._normalize_symbol)
        return frame

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        value = symbol.strip().upper()
        if "." in value:
            return value

        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) != 6:
            return value
        exchange = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
        return f"{digits}.{exchange}"

    def _normalize_dates(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        return frame

    @staticmethod
    def _normalize_numeric_fields(frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("open", "high", "low", "close", "volume", "amount", "adj_factor"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def _fill_missing_ohlc(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.config.fill_missing_ohlc_from_close:
            return frame

        for column in ("open", "high", "low"):
            frame[column] = frame[column].fillna(frame["close"])
        return frame

    @staticmethod
    def _derive_amount(frame: pd.DataFrame) -> pd.DataFrame:
        if "amount" not in frame.columns:
            frame["amount"] = np.nan
        frame["amount"] = frame["amount"].fillna(frame["close"] * frame["volume"])
        return frame

    @staticmethod
    def _drop_invalid_rows(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.dropna(subset=["symbol", "trade_date", "close"])
        frame = frame.loc[frame["close"] > 0].copy()
        for column in ("open", "high", "low"):
            frame = frame.loc[frame[column] > 0].copy()
        frame = frame.loc[frame["volume"].fillna(0) >= 0].copy()
        if "amount" in frame.columns:
            frame = frame.loc[frame["amount"].fillna(0) >= 0].copy()
        return frame

    @staticmethod
    def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.drop_duplicates(subset=["symbol", "trade_date"], keep="last")

    @staticmethod
    def _sort(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    def _empty_frame(self) -> pd.DataFrame:
        columns = [
            self.columns.symbol,
            self.columns.trade_date,
            self.columns.open,
            self.columns.high,
            self.columns.low,
            self.columns.close,
            self.columns.volume,
            self.columns.amount,
            self.columns.adj_factor,
        ]
        return pd.DataFrame(columns=columns)
