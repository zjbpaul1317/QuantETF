from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

from quant_etf.config.schema import DataConfig

from .models import DataLoadRequest
from .source_base import ETFHistoryDataSource

logger = logging.getLogger(__name__)


class AkShareETFSource(ETFHistoryDataSource):
    """Fetch and cache ETF daily history from AkShare."""

    CHUNK_DAYS = 120
    SINGLE_DAY_PADDING_DAYS = 3
    MAX_RETRIES = 3
    REQUEST_PAUSE_SECONDS = 0.5
    DEFAULT_START_DATE = "2000-01-01"
    PROXY_ENV_VARS = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = config.raw_dir
        self.lookback_days = max(0, int(config.incremental_update_lookback_days))

    def load_bars(self, request: DataLoadRequest) -> pd.DataFrame:
        try:
            import akshare as ak
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AkShare is not installed. Please install akshare or switch data.provider to 'local'."
            ) from exc

        frames: list[pd.DataFrame] = []
        missing_symbols: list[str] = []
        start_dt = self._parse_date(request.start_date, self.DEFAULT_START_DATE)
        end_dt = self._parse_date(request.end_date, datetime.now().strftime("%Y-%m-%d"))
        adjust = self._map_adjustment(self.config.adjustment)

        for symbol in request.symbols:
            normalized_symbol = self._normalize_symbol(symbol)
            frame = self._load_symbol_history(
                ak=ak,
                symbol=normalized_symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                adjust=adjust,
                force_reload=request.force_reload,
            )
            if frame.empty:
                missing_symbols.append(normalized_symbol)
                logger.warning(
                    "AkShare returned no ETF history for %s between %s and %s",
                    normalized_symbol,
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
                continue
            frames.append(frame)

        if not frames:
            requested = ", ".join(missing_symbols or [self._normalize_symbol(symbol) for symbol in request.symbols])
            raise FileNotFoundError(f"AkShare returned no ETF history for symbols: {requested}")

        if missing_symbols:
            logger.warning("No ETF history was returned for %s", ", ".join(missing_symbols))

        return pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)

    def _load_symbol_history(
        self,
        ak: Any,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        adjust: str,
        force_reload: bool,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol)
        cached = pd.DataFrame() if force_reload else self._read_cache(cache_path, symbol)
        fetch_ranges = self._determine_fetch_ranges(cached, start_dt, end_dt)

        fetched_frames: list[pd.DataFrame] = []
        for fetch_start, fetch_end in fetch_ranges:
            fetched = self._fetch_symbol_range(
                ak=ak,
                symbol=symbol,
                start_dt=fetch_start,
                end_dt=fetch_end,
                adjust=adjust,
            )
            if not fetched.empty:
                fetched_frames.append(fetched)

        merged = self._merge_frames([cached, *fetched_frames], symbol=symbol)
        if merged.empty:
            return merged

        if fetched_frames or (force_reload and not merged.empty):
            self._write_cache(cache_path, merged)

        mask = (merged["trade_date"] >= start_dt.strftime("%Y-%m-%d")) & (merged["trade_date"] <= end_dt.strftime("%Y-%m-%d"))
        return merged.loc[mask].reset_index(drop=True)

    def _determine_fetch_ranges(
        self,
        cached: pd.DataFrame,
        start_dt: datetime,
        end_dt: datetime,
    ) -> list[tuple[datetime, datetime]]:
        if cached.empty:
            return [(start_dt, end_dt)] if start_dt <= end_dt else []

        cached_dates = pd.to_datetime(cached["trade_date"], errors="coerce").dropna()
        if cached_dates.empty:
            return [(start_dt, end_dt)] if start_dt <= end_dt else []

        cache_start = cached_dates.min().to_pydatetime()
        cache_end = cached_dates.max().to_pydatetime()
        ranges: list[tuple[datetime, datetime]] = []

        if start_dt < cache_start:
            older_end = min(end_dt, cache_start - timedelta(days=1))
            if start_dt <= older_end:
                ranges.append((start_dt, older_end))

        if end_dt > cache_end:
            newer_start = max(start_dt, cache_end - timedelta(days=self.lookback_days))
            if newer_start <= end_dt:
                ranges.append((newer_start, end_dt))

        return ranges

    def _fetch_symbol_range(
        self,
        ak: Any,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        adjust: str,
    ) -> pd.DataFrame:
        if start_dt > end_dt:
            return self._empty_frame()

        chunks: list[pd.DataFrame] = []
        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(current_start + timedelta(days=self.CHUNK_DAYS - 1), end_dt)
            chunks.extend(
                self._fetch_range_resilient(
                    ak=ak,
                    symbol=symbol,
                    start_dt=current_start,
                    end_dt=current_end,
                    adjust=adjust,
                )
            )
            current_start = current_end + timedelta(days=1)

        return self._merge_frames(chunks, symbol=symbol)

    def _fetch_range_resilient(
        self,
        ak: Any,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        adjust: str,
    ) -> list[pd.DataFrame]:
        if start_dt > end_dt:
            return []

        code = self._normalize_symbol_code(symbol)
        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

        try:
            frame = self._fetch_with_retry(
                ak=ak,
                code=code,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception as exc:  # noqa: BLE001
            span_days = (end_dt - start_dt).days + 1
            fallback = self._fetch_with_padded_window(
                ak=ak,
                symbol=symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                adjust=adjust,
            )
            if not fallback.empty:
                return [fallback]
            if span_days <= 1:
                logger.warning(
                    "Skipping unresolved AkShare single-day window for %s on %s after retries: %s",
                    symbol,
                    start_dt.strftime("%Y-%m-%d"),
                    exc,
                )
                return []

            midpoint = start_dt + timedelta(days=(span_days // 2) - 1)
            logger.warning(
                "AkShare request failed for %s between %s and %s; retrying with smaller windows",
                symbol,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
            return [
                *self._fetch_range_resilient(ak=ak, symbol=symbol, start_dt=start_dt, end_dt=midpoint, adjust=adjust),
                *self._fetch_range_resilient(ak=ak, symbol=symbol, start_dt=midpoint + timedelta(days=1), end_dt=end_dt, adjust=adjust),
            ]

        if frame is None or frame.empty:
            if start_dt == end_dt:
                fallback = self._fetch_with_padded_window(
                    ak=ak,
                    symbol=symbol,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    adjust=adjust,
                )
                if not fallback.empty:
                    return [fallback]
            return []
        return [self._normalize_remote_frame(frame, symbol)]

    def _fetch_with_retry(
        self,
        ak: Any,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                with self._without_proxies():
                    frame = ak.fund_etf_hist_em(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                    )
                return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.MAX_RETRIES:
                    break
                time.sleep(max(self.REQUEST_PAUSE_SECONDS, attempt * self.REQUEST_PAUSE_SECONDS))

        assert last_error is not None
        raise RuntimeError(
            f"AkShare failed for ETF {code} between {start_date} and {end_date} after {self.MAX_RETRIES} attempts"
        ) from last_error

    def _normalize_remote_frame(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        required_columns = {"日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"}
        missing = sorted(required_columns.difference(frame.columns))
        if missing:
            raise RuntimeError(f"AkShare ETF history response for {symbol} is missing columns: {', '.join(missing)}")

        normalized = frame.copy()
        normalized["trade_date"] = pd.to_datetime(normalized["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
        normalized["symbol"] = symbol

        rename_map = {
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
        normalized = normalized.rename(columns=rename_map)

        for column in ("open", "high", "low", "close", "volume", "amount"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(float)
        columns = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
        return self._merge_frames([normalized[columns]], symbol=symbol)

    def _fetch_with_padded_window(
        self,
        ak: Any,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        adjust: str,
    ) -> pd.DataFrame:
        padded_start = start_dt - timedelta(days=self.SINGLE_DAY_PADDING_DAYS)
        padded_end = end_dt + timedelta(days=self.SINGLE_DAY_PADDING_DAYS)
        try:
            frame = self._fetch_with_retry(
                ak=ak,
                code=self._normalize_symbol_code(symbol),
                start_date=padded_start.strftime("%Y%m%d"),
                end_date=padded_end.strftime("%Y%m%d"),
                adjust=adjust,
            )
        except Exception:  # noqa: BLE001
            return self._empty_frame()

        if frame is None or frame.empty:
            return self._empty_frame()

        normalized = self._normalize_remote_frame(frame, symbol)
        mask = (normalized["trade_date"] >= start_dt.strftime("%Y-%m-%d")) & (normalized["trade_date"] <= end_dt.strftime("%Y-%m-%d"))
        return normalized.loc[mask].reset_index(drop=True)

    def _read_cache(self, path: Path, symbol: str) -> pd.DataFrame:
        if not path.exists():
            return self._empty_frame()
        frame = pd.read_csv(path)
        if frame.empty:
            return self._empty_frame()
        return self._merge_frames([self._normalize_cached_frame(frame, symbol)], symbol=symbol)

    def _normalize_cached_frame(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        normalized = frame.copy()
        normalized = normalized.rename(
            columns={
                "日期": "trade_date",
                "date": "trade_date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            }
        )
        if "symbol" not in normalized.columns:
            normalized["symbol"] = symbol
        columns = [column for column in self._empty_frame().columns if column in normalized.columns]
        return normalized[columns]

    @staticmethod
    def _write_cache(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def _cache_path(self, symbol: str) -> Path:
        return self.raw_dir / f"{symbol}.csv"

    def _merge_frames(self, frames: list[pd.DataFrame], symbol: str) -> pd.DataFrame:
        valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not valid_frames:
            return self._empty_frame()

        merged = pd.concat(valid_frames, ignore_index=True, sort=False)
        if "symbol" not in merged.columns:
            merged["symbol"] = symbol
        merged["symbol"] = merged["symbol"].astype(str).map(self._normalize_symbol)
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")

        for column in ("open", "high", "low", "close", "volume", "amount", "adj_factor"):
            if column in merged.columns:
                merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(float)

        merged = merged.dropna(subset=["trade_date"])
        merged = merged.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

        ordered_columns = [column for column in self._empty_frame().columns if column in merged.columns]
        return merged[ordered_columns]

    @classmethod
    @contextmanager
    def _without_proxies(cls) -> Any:
        previous = {key: os.environ.get(key) for key in cls.PROXY_ENV_VARS if key in os.environ}
        try:
            for key in cls.PROXY_ENV_VARS:
                os.environ.pop(key, None)
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"
            yield
        finally:
            for key in cls.PROXY_ENV_VARS:
                os.environ.pop(key, None)
            for key, value in previous.items():
                os.environ[key] = value

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        value = str(symbol).strip().upper()
        if "." in value:
            return value

        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) != 6:
            return value
        exchange = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
        return f"{digits}.{exchange}"

    @staticmethod
    def _normalize_symbol_code(symbol: str) -> str:
        return str(symbol).split(".")[0]

    @staticmethod
    def _parse_date(value: str | None, default: str) -> datetime:
        candidate = value or default
        normalized = str(candidate).replace("/", "-")
        return datetime.fromisoformat(normalized)

    @staticmethod
    def _map_adjustment(adjustment: str) -> str:
        if adjustment == "none":
            return ""
        return adjustment

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount", "adj_factor"])
