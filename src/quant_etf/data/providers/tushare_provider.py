from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import time
from typing import Any

import pandas as pd

from quant_etf.config.schema import DataConfig

from ..models import DataLoadRequest
from ..source_base import ETFHistoryDataSource

logger = logging.getLogger(__name__)


class TushareETFSource(ETFHistoryDataSource):
    """Fetch and cache ETF daily history from Tushare Pro."""

    MAX_RETRIES = 3
    REQUEST_PAUSE_SECONDS = 0.2
    DEFAULT_START_DATE = "2000-01-01"

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = config.raw_dir
        self.lookback_days = max(0, int(config.incremental_update_lookback_days))
        self._client: Any | None = None

    def load_bars(self, request: DataLoadRequest) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        missing_symbols: list[str] = []

        start_dt = self._parse_date(request.start_date, self.DEFAULT_START_DATE)
        end_dt = self._parse_date(request.end_date, datetime.now().strftime("%Y-%m-%d"))

        for symbol in request.symbols:
            normalized_symbol = self._normalize_symbol(symbol)
            frame = self._load_symbol_history(
                client=None,
                symbol=normalized_symbol,
                start_dt=start_dt,
                end_dt=end_dt,
                force_reload=request.force_reload,
            )
            if frame.empty:
                missing_symbols.append(normalized_symbol)
                logger.warning(
                    "Tushare returned no ETF history for %s between %s and %s",
                    normalized_symbol,
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
                continue
            frames.append(frame)

        if not frames:
            requested = ", ".join(missing_symbols or [self._normalize_symbol(symbol) for symbol in request.symbols])
            raise FileNotFoundError(f"Tushare returned no ETF history for symbols: {requested}")

        if missing_symbols:
            logger.warning("No ETF history was returned for %s", ", ".join(missing_symbols))

        return pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        token = self._resolve_token()
        if not token:
            raise RuntimeError(
                "Tushare token is not configured. Set data.tushare_token in configs or export "
                "QUANT_ETF_DATA__TUSHARE_TOKEN / TUSHARE_TOKEN before using provider='tushare'."
            )

        try:
            import tushare as ts
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Tushare is not installed. Please install tushare or switch data.provider to 'local'."
            ) from exc

        ts.set_token(token)
        self._client = ts.pro_api(token)
        return self._client

    def _resolve_token(self) -> str | None:
        token = self.config.tushare_token or os.getenv("QUANT_ETF_DATA__TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
        return token.strip() if isinstance(token, str) and token.strip() else None

    def _load_symbol_history(
        self,
        client: Any | None,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        force_reload: bool,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol)
        cached = pd.DataFrame() if force_reload else self._read_cache(cache_path)
        fetch_ranges = self._determine_fetch_ranges(cached, start_dt, end_dt)

        fetched_frames: list[pd.DataFrame] = []
        for fetch_start, fetch_end in fetch_ranges:
            if client is None:
                client = self._get_client()
            fetched = self._fetch_symbol_range(client, symbol, fetch_start, fetch_end)
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
        client: Any,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        if start_dt > end_dt:
            return pd.DataFrame()

        daily = self._call_with_retry(
            client.fund_daily,
            ts_code=symbol,
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )

        if daily is None or daily.empty:
            return pd.DataFrame()

        adj = self._fetch_adj_factor(client, symbol, start_dt, end_dt)
        return self._normalize_remote_frame(daily, adj, symbol)

    def _fetch_adj_factor(
        self,
        client: Any,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        try:
            adj = self._call_with_retry(
                client.fund_adj,
                ts_code=symbol,
                start_date=start_dt.strftime("%Y%m%d"),
                end_date=end_dt.strftime("%Y%m%d"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch fund_adj for %s: %s", symbol, exc)
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])

        if adj is None or adj.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        return adj

    def _normalize_remote_frame(
        self,
        daily: pd.DataFrame,
        adj: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        required_columns = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
        missing = sorted(required_columns.difference(daily.columns))
        if missing:
            raise RuntimeError(f"Tushare fund_daily response for {symbol} is missing columns: {', '.join(missing)}")

        frame = daily.copy()
        if not adj.empty and {"trade_date", "adj_factor"}.issubset(adj.columns):
            adj_frame = adj[["trade_date", "adj_factor"]].copy()
            frame = frame.merge(adj_frame, on="trade_date", how="left")

        frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
        frame["symbol"] = frame["ts_code"].astype(str).str.upper()

        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

        # Tushare fund_daily uses vol in hands and amount in thousand yuan.
        frame["volume"] = pd.to_numeric(frame["vol"], errors="coerce").astype(float) * 100.0
        frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").astype(float) * 1000.0
        if "adj_factor" in frame.columns:
            frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce").astype(float)

        columns = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
        if "adj_factor" in frame.columns:
            columns.append("adj_factor")

        result = frame[columns].dropna(subset=["trade_date"]).copy()
        result = result.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        result = result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return result

    def _merge_frames(self, frames: list[pd.DataFrame], symbol: str) -> pd.DataFrame:
        valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
        if not valid_frames:
            return pd.DataFrame(columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount", "adj_factor"])

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

    @classmethod
    def _call_with_retry(cls, func: Any, **kwargs: Any) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                result = func(**kwargs)
                return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == cls.MAX_RETRIES:
                    break
                time.sleep(max(cls.REQUEST_PAUSE_SECONDS, attempt * cls.REQUEST_PAUSE_SECONDS))

        assert last_error is not None
        raise RuntimeError(f"Tushare request failed after {cls.MAX_RETRIES} attempts: {kwargs}") from last_error

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

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol).strip().upper()

    @staticmethod
    def _parse_date(value: str | None, default: str) -> datetime:
        candidate = value or default
        normalized = str(candidate).replace("/", "-")
        return datetime.fromisoformat(normalized)
