from __future__ import annotations

from datetime import datetime
import time

import pandas as pd

from quant_etf.config.schema import DataConfig

from .models import DataLoadRequest
from .source_base import ETFHistoryDataSource


class AkShareETFSource(ETFHistoryDataSource):
    """Fetch ETF daily history from AkShare."""

    CHUNK_DAYS = 120
    MAX_RETRIES = 3

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def load_bars(self, request: DataLoadRequest) -> pd.DataFrame:
        try:
            import akshare as ak
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "AkShare is not installed. Please install akshare or switch data.provider to 'local'."
            ) from exc

        frames: list[pd.DataFrame] = []
        start_dt = self._parse_date(request.start_date)
        end_dt = self._parse_date(request.end_date)
        adjust = self._map_adjustment(self.config.adjustment)

        for symbol in request.symbols:
            code = self._normalize_symbol_code(symbol)
            frame = self._fetch_symbol_chunks(
                ak=ak,
                code=code,
                start_dt=start_dt,
                end_dt=end_dt,
                adjust=adjust,
            )
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame["symbol"] = symbol.upper()
            frames.append(frame)

        if not frames:
            requested = ", ".join(request.symbols)
            raise FileNotFoundError(f"AkShare returned no ETF history for symbols: {requested}")

        return pd.concat(frames, ignore_index=True, sort=False)

    @staticmethod
    def _normalize_symbol_code(symbol: str) -> str:
        return str(symbol).split(".")[0]

    @staticmethod
    def _parse_date(value: str | None) -> datetime:
        if value is None:
            return datetime(2000, 1, 1)
        return datetime.fromisoformat(str(value))

    @staticmethod
    def _map_adjustment(adjustment: str) -> str:
        if adjustment == "none":
            return ""
        return adjustment

    def _fetch_symbol_chunks(
        self,
        ak: object,
        code: str,
        start_dt: datetime,
        end_dt: datetime,
        adjust: str,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(current_start + pd.Timedelta(days=self.CHUNK_DAYS - 1), end_dt)
            frame = self._fetch_with_retry(
                ak=ak,
                code=code,
                start_date=current_start.strftime("%Y%m%d"),
                end_date=current_end.strftime("%Y%m%d"),
                adjust=adjust,
            )
            if frame is not None and not frame.empty:
                frames.append(frame)
            current_start = current_end + pd.Timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True, sort=False)
        date_col = "日期" if "日期" in result.columns else "trade_date"
        if date_col in result.columns:
            result = result.drop_duplicates(subset=[date_col], keep="last")
        return result.reset_index(drop=True)

    def _fetch_with_retry(
        self,
        ak: object,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.MAX_RETRIES:
                    break
                time.sleep(attempt)

        assert last_error is not None
        raise RuntimeError(
            f"AkShare failed for ETF {code} between {start_date} and {end_date} after {self.MAX_RETRIES} attempts"
        ) from last_error
