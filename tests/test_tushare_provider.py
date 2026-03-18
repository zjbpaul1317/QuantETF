from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_etf.config import load_app_config
from quant_etf.data import DataLoadRequest, TushareETFSource


class FakeTusharePro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def fund_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append(("fund_daily", ts_code, start_date, end_date))
        frame = pd.DataFrame(
            [
                {
                    "ts_code": ts_code,
                    "trade_date": "20240105",
                    "open": 1.04,
                    "high": 1.05,
                    "low": 1.03,
                    "close": 1.045,
                    "vol": 1300,
                    "amount": 1350,
                },
                {
                    "ts_code": ts_code,
                    "trade_date": "20240104",
                    "open": 1.03,
                    "high": 1.04,
                    "low": 1.02,
                    "close": 1.035,
                    "vol": 1200,
                    "amount": 1240,
                },
                {
                    "ts_code": ts_code,
                    "trade_date": "20240103",
                    "open": 1.02,
                    "high": 1.03,
                    "low": 1.01,
                    "close": 1.025,
                    "vol": 1100,
                    "amount": 1130,
                },
                {
                    "ts_code": ts_code,
                    "trade_date": "20240102",
                    "open": 1.01,
                    "high": 1.02,
                    "low": 1.00,
                    "close": 1.015,
                    "vol": 1000,
                    "amount": 1010,
                },
            ]
        )
        mask = (frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)
        return frame.loc[mask].reset_index(drop=True)

    def fund_adj(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.calls.append(("fund_adj", ts_code, start_date, end_date))
        frame = pd.DataFrame(
            [
                {"ts_code": ts_code, "trade_date": "20240105", "adj_factor": 1.03},
                {"ts_code": ts_code, "trade_date": "20240104", "adj_factor": 1.02},
                {"ts_code": ts_code, "trade_date": "20240103", "adj_factor": 1.01},
                {"ts_code": ts_code, "trade_date": "20240102", "adj_factor": 1.00},
            ]
        )
        mask = (frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)
        return frame.loc[mask].reset_index(drop=True)


def _build_source(tmp_path: Path) -> tuple[TushareETFSource, FakeTusharePro]:
    config = load_app_config("configs", env_prefix=None)
    data_config = replace(
        config.data,
        provider="tushare",
        raw_dir=tmp_path.resolve(),
        file_format="csv",
        tushare_token="unit-test-token",
        incremental_update_lookback_days=1,
    )
    source = TushareETFSource(data_config)
    client = FakeTusharePro()
    source._client = client
    return source, client


def test_tushare_source_maps_and_caches_history(tmp_path: Path) -> None:
    source, client = _build_source(tmp_path)

    frame = source.load_bars(
        DataLoadRequest(
            symbols=["510300.SH"],
            start_date="2024-01-02",
            end_date="2024-01-05",
        )
    )

    assert list(frame["trade_date"]) == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
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
    assert frame.loc[0, "open"] == 1.01
    assert (tmp_path / "510300.SH.csv").exists()
    assert ("fund_daily", "510300.SH", "20240102", "20240105") in client.calls
    assert ("fund_adj", "510300.SH", "20240102", "20240105") in client.calls


def test_tushare_source_incrementally_updates_existing_cache(tmp_path: Path) -> None:
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
        ]
    )
    existing.to_csv(tmp_path / "510300.SH.csv", index=False)

    source, client = _build_source(tmp_path)

    frame = source.load_bars(
        DataLoadRequest(
            symbols=["510300.SH"],
            start_date="2024-01-02",
            end_date="2024-01-05",
        )
    )

    assert len(frame) == 4
    assert ("fund_daily", "510300.SH", "20240102", "20240105") in client.calls

    cached = pd.read_csv(tmp_path / "510300.SH.csv")
    assert cached["trade_date"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert cached["symbol"].unique().tolist() == ["510300.SH"]
