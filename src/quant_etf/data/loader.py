from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf.config.schema import DataConfig

from .models import DataLoadRequest
from .source_base import ETFHistoryDataSource


class LocalETFFileSource(ETFHistoryDataSource):
    """Read ETF history from local csv/parquet files."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = config.raw_dir
        self.file_format = config.file_format
        self.file_pattern = config.file_pattern
        self.combined_file_name = config.combined_file_name

    def load_bars(self, request: DataLoadRequest) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        symbols = [symbol.upper() for symbol in request.symbols]

        for symbol in symbols:
            frame = self._load_single_symbol(symbol)
            if frame is not None and not frame.empty:
                if self.config.symbol_column not in frame.columns:
                    frame[self.config.symbol_column] = symbol
                frames.append(frame)

        if not frames:
            combined = self._load_combined_file()
            if combined is None or combined.empty:
                requested = ", ".join(symbols)
                raise FileNotFoundError(f"No ETF history files found for symbols: {requested}")

            symbol_col = self.config.symbol_column
            if symbol_col not in combined.columns:
                raise ValueError(f"Combined data file must contain symbol column: {symbol_col}")
            frames.append(combined[combined[symbol_col].astype(str).str.upper().isin(symbols)].copy())

        return pd.concat(frames, ignore_index=True, sort=False).reset_index(drop=True)

    def _load_single_symbol(self, symbol: str) -> pd.DataFrame | None:
        for extension in self._extensions():
            path = self.raw_dir / self.file_pattern.format(symbol=symbol, ext=extension)
            if path.exists():
                return self._read_file(path)
        return None

    def _load_combined_file(self) -> pd.DataFrame | None:
        for extension in self._extensions():
            path = self.raw_dir / f"{self.combined_file_name}.{extension}"
            if path.exists():
                return self._read_file(path)
        return None

    def _extensions(self) -> tuple[str, ...]:
        if self.file_format == "csv":
            return ("csv",)
        if self.file_format == "parquet":
            return ("parquet",)
        return ("parquet", "csv")

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file type: {path}")
