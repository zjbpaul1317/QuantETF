from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_etf.config import load_app_config
from quant_etf.data import DataLoadRequest, ETFDataCleaner, ETFDataRepository, LocalETFFileSource


def _build_test_config():
    config = load_app_config("configs", env_prefix=None)
    data_config = replace(
        config.data,
        raw_dir=Path("tests/fixtures/data").resolve(),
        provider="local",
        file_format="csv",
    )
    return replace(
        config,
        data=data_config,
        universe=replace(config.universe, symbols=["510300.SH", "159915.SZ"]),
        backtest=replace(config.backtest, start_date="2024-01-01", end_date="2024-01-31"),
    )


def test_cleaner_normalizes_duplicate_and_chinese_columns() -> None:
    config = _build_test_config()
    source = LocalETFFileSource(config.data)
    raw = source.load_bars(DataLoadRequest(symbols=["510300.SH", "159915.SZ"]))
    cleaner = ETFDataCleaner(config.data)

    cleaned = cleaner.clean(raw)

    assert "symbol" in cleaned.columns
    assert "trade_date" in cleaned.columns
    assert cleaned.loc[cleaned["symbol"] == "510300.SH"].shape[0] == 3
    assert cleaned["trade_date"].dtype.kind == "M"


def test_preprocessor_returns_multi_index_history() -> None:
    config = _build_test_config()
    repo = ETFDataRepository(config)

    history = repo.load_history()

    assert isinstance(history.index, pd.MultiIndex)
    assert history.index.names == ["trade_date", "symbol"]
    assert "return_1d" in history.columns
    assert "listed_days" in history.columns
    assert history.loc[(pd.Timestamp("2024-01-04"), "510300.SH"), "is_suspended"]
