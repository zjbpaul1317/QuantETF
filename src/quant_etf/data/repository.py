from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf.config.schema import AppConfig

from .cleaner import ETFDataCleaner
from .loader import LocalETFFileSource
from .models import DataLoadRequest
from .preprocessor import ETFDataPreprocessor
from .providers import EasyQuotationETFSource
from .source_akshare import AkShareETFSource
from .source_base import ETFHistoryDataSource


class ETFDataRepository:
    """Facade for raw file reading, cleaning and preprocessing."""

    def __init__(
        self,
        config: AppConfig,
        source: ETFHistoryDataSource | None = None,
        cleaner: ETFDataCleaner | None = None,
        preprocessor: ETFDataPreprocessor | None = None,
    ) -> None:
        self.config = config
        if source is None:
            if config.data.provider == "local":
                source = LocalETFFileSource(config.data)
            elif config.data.provider == "akshare":
                source = AkShareETFSource(config.data)
            elif config.data.provider == "easyquotation":
                source = EasyQuotationETFSource(config.data)
            else:
                raise NotImplementedError(
                    f"Data provider '{config.data.provider}' is reserved but not implemented yet. "
                    "Use provider='local', 'akshare' or 'easyquotation' for the current pipeline."
                )
        self.source = source
        self.cleaner = cleaner or ETFDataCleaner(config.data)
        self.preprocessor = preprocessor or ETFDataPreprocessor(config.data)

    def load_history(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        columns: list[str] | None = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        request = DataLoadRequest(
            symbols=symbols or self.config.universe.symbols,
            start_date=start_date or self.config.backtest.start_date,
            end_date=end_date or self.config.backtest.end_date,
            columns=columns,
            force_reload=force_reload,
        )
        raw = self.source.load_bars(request)
        cleaned = self.cleaner.clean(raw)
        if request.start_date:
            cleaned = cleaned.loc[cleaned["trade_date"] >= pd.Timestamp(request.start_date)].copy()
        if request.end_date:
            cleaned = cleaned.loc[cleaned["trade_date"] <= pd.Timestamp(request.end_date)].copy()
        processed = self.preprocessor.preprocess(cleaned)
        if columns:
            available_columns = [column for column in columns if column in processed.columns]
            processed = processed[available_columns]
        return processed

    def save_processed_history(
        self,
        history: pd.DataFrame,
        filename: str = "etf_history_processed.csv",
    ) -> Path:
        output_path = self.config.data.processed_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            history.reset_index().to_csv(output_path, index=False)
        else:
            try:
                history.reset_index().to_parquet(output_path, index=False)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "Saving parquet requires an additional parquet engine such as pyarrow."
                ) from exc
        return output_path
