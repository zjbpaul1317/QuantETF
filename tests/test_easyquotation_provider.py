from dataclasses import replace
from pathlib import Path
import sys
import types

import pandas as pd
import pytest

from quant_etf.config import load_app_config
from quant_etf.data import DataLoadRequest, EasyQuotationETFSource


class FakeEasyQuotationClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.payload: dict[str, dict[str, object]] = {
            "510300": {
                "name": "沪深300ETF",
                "open": 1.01,
                "high": 1.05,
                "low": 1.00,
                "close": 1.00,
                "now": 1.045,
                "turnover": 100000.0,
                "volume": 1010000.0,
                "date": "2024-01-05",
                "time": "15:00:00",
            }
        }

    def real(self, symbols: list[str]) -> dict[str, dict[str, object]]:
        self.calls.append(list(symbols))
        return {symbol: self.payload[symbol] for symbol in symbols if symbol in self.payload}


def _install_fake_easyquotation(monkeypatch: pytest.MonkeyPatch, client: FakeEasyQuotationClient) -> None:
    module = types.SimpleNamespace(use=lambda provider: client)
    monkeypatch.setitem(sys.modules, "easyquotation", module)


def _build_source(tmp_path: Path) -> EasyQuotationETFSource:
    config = load_app_config("configs", env_prefix=None)
    data_config = replace(
        config.data,
        provider="easyquotation",
        raw_dir=tmp_path.resolve(),
        file_format="csv",
    )
    return EasyQuotationETFSource(data_config)


def test_easyquotation_source_maps_and_caches_latest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeEasyQuotationClient()
    _install_fake_easyquotation(monkeypatch, client)
    source = _build_source(tmp_path)

    frame = source.load_bars(
        DataLoadRequest(
            symbols=["510300.SH"],
            start_date="2024-01-05",
            end_date="2024-01-05",
        )
    )

    assert list(frame["trade_date"]) == ["2024-01-05"]
    assert list(frame.columns) == [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adj_factor",
    ]
    assert frame["symbol"].unique().tolist() == ["510300.SH"]
    assert frame.loc[0, "volume"] == 100000.0
    assert frame.loc[0, "amount"] == 1010000.0
    assert frame.loc[0, "close"] == 1.045
    assert (tmp_path / "510300.SH.csv").exists()
    assert client.calls == [["510300"]]


def test_easyquotation_source_incrementally_updates_existing_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "symbol": "510300.SH",
                "open": 1.01,
                "high": 1.02,
                "low": 1.00,
                "close": 1.015,
                "volume": 100000.0,
                "amount": 1010000.0,
                "adj_factor": 1.00,
            },
            {
                "trade_date": "2024-01-03",
                "symbol": "510300.SH",
                "open": 1.02,
                "high": 1.03,
                "low": 1.01,
                "close": 1.025,
                "volume": 110000.0,
                "amount": 1130000.0,
                "adj_factor": 1.01,
            },
            {
                "trade_date": "2024-01-04",
                "symbol": "510300.SH",
                "open": 1.03,
                "high": 1.04,
                "low": 1.02,
                "close": 1.035,
                "volume": 120000.0,
                "amount": 1240000.0,
                "adj_factor": 1.02,
            },
        ]
    )
    existing.to_csv(tmp_path / "510300.SH.csv", index=False)

    client = FakeEasyQuotationClient()
    _install_fake_easyquotation(monkeypatch, client)
    source = _build_source(tmp_path)

    frame = source.load_bars(
        DataLoadRequest(
            symbols=["510300.SH"],
            start_date="2024-01-02",
            end_date="2024-01-05",
        )
    )

    assert len(frame) == 4
    cached = pd.read_csv(tmp_path / "510300.SH.csv")
    assert cached["trade_date"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert cached["symbol"].unique().tolist() == ["510300.SH"]
    assert client.calls == [["510300"]]


def test_easyquotation_source_requires_local_cache_for_older_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeEasyQuotationClient()
    _install_fake_easyquotation(monkeypatch, client)
    source = _build_source(tmp_path)

    with pytest.raises(FileNotFoundError, match="live snapshots and ETF metadata"):
        source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH"],
                start_date="2024-01-02",
                end_date="2024-01-04",
            )
        )
