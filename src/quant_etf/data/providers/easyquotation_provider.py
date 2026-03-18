from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any

import pandas as pd

from quant_etf.config.schema import DataConfig

from ..models import DataLoadRequest
from ..source_base import ETFHistoryDataSource

logger = logging.getLogger(__name__)


class EasyQuotationETFSource(ETFHistoryDataSource):
    """Fetch ETF bars from easyquotation live snapshots and maintain a local cache."""

    MAX_RETRIES = 3
    REQUEST_PAUSE_SECONDS = 0.2
    DEFAULT_START_DATE = "2000-01-01"
    SNAPSHOT_PROVIDER = "sina"

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = config.raw_dir
        self._client: Any | None = None

    def load_bars(self, request: DataLoadRequest) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        missing_symbols: list[str] = []
        normalized_symbols = [self._normalize_symbol(symbol) for symbol in request.symbols]

        start_dt = self._parse_date(request.start_date, self.DEFAULT_START_DATE)
        end_dt = self._parse_date(request.end_date, datetime.now().strftime("%Y-%m-%d"))

        cached_by_symbol = {
            symbol: self._read_cache(self._cache_path(symbol))
            for symbol in normalized_symbols
        }

        live_quotes: dict[str, dict[str, Any]] = {}
        if any(self._needs_live_update(cached, end_dt, request.force_reload) for cached in cached_by_symbol.values()):
            live_quotes = self._fetch_live_quotes(normalized_symbols)

        for symbol in normalized_symbols:
            cached = cached_by_symbol[symbol]
            live_frame = self._normalize_live_quote(live_quotes.get(symbol), symbol)
            merged = self._merge_frames([cached, live_frame], symbol=symbol)

            if not merged.empty and not merged.equals(cached):
                self._write_cache(self._cache_path(symbol), merged)

            if merged.empty:
                missing_symbols.append(symbol)
                continue

            mask = (
                (merged["trade_date"] >= start_dt.strftime("%Y-%m-%d"))
                & (merged["trade_date"] <= end_dt.strftime("%Y-%m-%d"))
            )
            sliced = merged.loc[mask].reset_index(drop=True)
            if sliced.empty:
                missing_symbols.append(symbol)
                logger.warning(
                    "EasyQuotation has no cached ETF history for %s between %s and %s",
                    symbol,
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
                continue
            frames.append(sliced)

        if not frames:
            requested = ", ".join(missing_symbols or normalized_symbols)
            raise FileNotFoundError(
                "EasyQuotation returned no ETF history for symbols: "
                f"{requested}. easyquotation exposes live snapshots and ETF metadata, "
                "so older trading days must already exist in the local CSV cache."
            )

        if missing_symbols:
            logger.warning("No ETF history was returned for %s", ", ".join(missing_symbols))

        return pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import easyquotation
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "easyquotation is not installed. Please install easyquotation or switch data.provider to 'local'."
            ) from exc

        self._client = easyquotation.use(self.SNAPSHOT_PROVIDER)
        return self._client

    def _fetch_live_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        if not symbols:
            return {}

        client = self._get_client()
        codes = [self._symbol_to_code(symbol) for symbol in symbols]
        raw_quotes = self._call_with_retry(client.real, codes)

        result: dict[str, dict[str, Any]] = {}
        for symbol, code in zip(symbols, codes, strict=True):
            quote = self._extract_quote(raw_quotes, symbol=symbol, code=code)
            if quote:
                result[symbol] = quote
        return result

    @classmethod
    def _call_with_retry(cls, func: Any, codes: list[str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                result = func(codes)
                return result if isinstance(result, dict) else {}
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == cls.MAX_RETRIES:
                    break
                time.sleep(max(cls.REQUEST_PAUSE_SECONDS, attempt * cls.REQUEST_PAUSE_SECONDS))

        assert last_error is not None
        raise RuntimeError(
            f"easyquotation request failed after {cls.MAX_RETRIES} attempts for symbols: {', '.join(codes)}"
        ) from last_error

    def _normalize_live_quote(self, quote: dict[str, Any] | None, symbol: str) -> pd.DataFrame:
        if not quote:
            return pd.DataFrame()

        trade_date = pd.to_datetime(quote.get("date"), errors="coerce")
        if pd.isna(trade_date):
            return pd.DataFrame()

        last_price = self._to_float(quote.get("now"))
        if last_price is None or last_price <= 0:
            return pd.DataFrame()

        previous_close = self._to_float(quote.get("close"))
        open_price = self._coalesce_numeric(quote.get("open"), last_price, previous_close)
        high_price = self._coalesce_numeric(quote.get("high"), last_price, previous_close)
        low_price = self._coalesce_numeric(quote.get("low"), last_price, previous_close)

        frame = pd.DataFrame(
            [
                {
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": last_price,
                    # easyquotation.sina uses turnover for traded shares, volume for traded amount.
                    "volume": self._to_float(quote.get("turnover")),
                    "amount": self._to_float(quote.get("volume")),
                    "adj_factor": pd.NA,
                }
            ]
        )
        return self._merge_frames([frame], symbol=symbol)

    @staticmethod
    def _extract_quote(payload: dict[str, Any], symbol: str, code: str) -> dict[str, Any] | None:
        if not payload:
            return None

        exchange = symbol.split(".")[-1].lower()
        prefixed_code = f"{exchange}{code}"
        candidates = (code, code.lower(), code.upper(), prefixed_code, prefixed_code.lower(), prefixed_code.upper())
        for candidate in candidates:
            quote = payload.get(candidate)
            if isinstance(quote, dict):
                return quote
        return None

    @staticmethod
    def _needs_live_update(cached: pd.DataFrame, end_dt: datetime, force_reload: bool) -> bool:
        if force_reload or cached.empty:
            return True
        cached_dates = pd.to_datetime(cached["trade_date"], errors="coerce").dropna()
        if cached_dates.empty:
            return True
        return cached_dates.max().to_pydatetime() < end_dt

    def _read_cache(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path)
        return self._merge_frames([frame], symbol=path.stem.upper())

    @staticmethod
    def _write_cache(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def _cache_path(self, symbol: str) -> Path:
        return self.raw_dir / f"{symbol}.csv"

    def _merge_frames(self, frames: list[pd.DataFrame], symbol: str) -> pd.DataFrame:
        valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not valid_frames:
            return pd.DataFrame(
                columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount", "adj_factor"]
            )

        merged = pd.concat(valid_frames, ignore_index=True, sort=False)
        if "symbol" not in merged.columns:
            merged["symbol"] = symbol
        merged["symbol"] = merged["symbol"].astype(str).str.upper()
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        for column in ("open", "high", "low", "close", "volume", "amount", "adj_factor"):
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(float)

        merged = merged.dropna(subset=["trade_date"])
        merged = merged.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return merged

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol).strip().upper()

    @staticmethod
    def _symbol_to_code(symbol: str) -> str:
        return symbol.split(".")[0]

    @staticmethod
    def _parse_date(value: str | None, default: str) -> datetime:
        candidate = value or default
        normalized = str(candidate).replace("/", "-")
        return datetime.fromisoformat(normalized)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return None
        return float(numeric)

    @classmethod
    def _coalesce_numeric(cls, *values: Any) -> float:
        for value in values:
            numeric = cls._to_float(value)
            if numeric is not None and numeric > 0:
                return numeric
        return 0.0
