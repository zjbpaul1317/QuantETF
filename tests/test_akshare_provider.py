from dataclasses import replace
import os
from pathlib import Path
import sys
import types

import pandas as pd

from quant_etf.config import load_app_config
from quant_etf.data import AkShareETFSource, DataLoadRequest


class FakeAkShareClient:
    def __init__(
        self,
        max_span_days: int | None = None,
        fail_exact_days: set[str] | None = None,
        always_fail_symbols: set[str] | None = None,
    ) -> None:
        self.max_span_days = max_span_days
        self.fail_exact_days = fail_exact_days or set()
        self.always_fail_symbols = always_fail_symbols or set()
        self.calls: list[dict[str, object]] = []
        self.dataset = pd.DataFrame(
            [
                {"日期": "2024-01-02", "开盘": 1.01, "收盘": 1.015, "最高": 1.02, "最低": 1.00, "成交量": 1000, "成交额": 1010},
                {"日期": "2024-01-03", "开盘": 1.02, "收盘": 1.025, "最高": 1.03, "最低": 1.01, "成交量": 1100, "成交额": 1130},
                {"日期": "2024-01-04", "开盘": 1.03, "收盘": 1.035, "最高": 1.04, "最低": 1.02, "成交量": 1200, "成交额": 1240},
                {"日期": "2024-01-05", "开盘": 1.04, "收盘": 1.045, "最高": 1.05, "最低": 1.03, "成交量": 1300, "成交额": 1350},
            ]
        )

    def fund_etf_hist_em(self, symbol: str, period: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
        self.calls.append(
            {
                "symbol": symbol,
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "adjust": adjust,
                "http_proxy": os.environ.get("HTTP_PROXY"),
                "https_proxy": os.environ.get("HTTPS_PROXY"),
                "all_proxy": os.environ.get("ALL_PROXY"),
                "no_proxy": os.environ.get("NO_PROXY"),
            }
        )

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        span_days = (end - start).days + 1
        if symbol in self.always_fail_symbols:
            raise RuntimeError("simulated symbol-wide disconnect")
        if start_date == end_date and start_date in self.fail_exact_days:
            raise RuntimeError("simulated single-day disconnect")
        if self.max_span_days is not None and span_days > self.max_span_days:
            raise RuntimeError("simulated upstream disconnect")

        frame = self.dataset.copy()
        frame["日期"] = pd.to_datetime(frame["日期"])
        mask = (frame["日期"] >= start) & (frame["日期"] <= end)
        result = frame.loc[mask].copy()
        result["日期"] = result["日期"].dt.strftime("%Y-%m-%d")
        return result.reset_index(drop=True)


def _build_source(tmp_path: Path, lookback_days: int = 1) -> AkShareETFSource:
    config = load_app_config("configs", env_prefix=None)
    data_config = replace(
        config.data,
        provider="akshare",
        raw_dir=tmp_path.resolve(),
        file_format="csv",
        incremental_update_lookback_days=lookback_days,
    )
    return AkShareETFSource(data_config)


def _install_fake_akshare(client: FakeAkShareClient):
    module = types.SimpleNamespace(fund_etf_hist_em=client.fund_etf_hist_em)
    original = sys.modules.get("akshare")
    sys.modules["akshare"] = module
    return original


def _restore_fake_akshare(original) -> None:
    if original is None:
        sys.modules.pop("akshare", None)
    else:
        sys.modules["akshare"] = original


def test_akshare_source_maps_and_caches_history(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    client = FakeAkShareClient()
    original = _install_fake_akshare(client)

    try:
        frame = source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH"],
                start_date="2024-01-02",
                end_date="2024-01-05",
            )
        )
    finally:
        _restore_fake_akshare(original)

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
    ]
    assert frame["symbol"].unique().tolist() == ["510300.SH"]
    assert frame.loc[0, "volume"] == 1000.0
    assert frame.loc[0, "amount"] == 1010.0
    assert (tmp_path / "510300.SH.csv").exists()
    assert client.calls[0]["symbol"] == "510300"
    assert client.calls[0]["period"] == "daily"
    assert client.calls[0]["adjust"] == "qfq"

    cached_client = FakeAkShareClient()
    original = _install_fake_akshare(cached_client)
    try:
        cached_frame = source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH"],
                start_date="2024-01-02",
                end_date="2024-01-05",
            )
        )
    finally:
        _restore_fake_akshare(original)

    assert cached_frame.equals(frame)
    assert cached_client.calls == []


def test_akshare_source_incrementally_updates_existing_cache(tmp_path: Path) -> None:
    existing = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "symbol": "510300.SH",
                "open": 1.01,
                "high": 1.02,
                "low": 1.00,
                "close": 1.015,
                "volume": 1000.0,
                "amount": 1010.0,
            },
            {
                "trade_date": "2024-01-03",
                "symbol": "510300.SH",
                "open": 1.02,
                "high": 1.03,
                "low": 1.01,
                "close": 1.025,
                "volume": 1100.0,
                "amount": 1130.0,
            },
        ]
    )
    existing.to_csv(tmp_path / "510300.SH.csv", index=False)

    source = _build_source(tmp_path, lookback_days=1)
    client = FakeAkShareClient()
    original = _install_fake_akshare(client)

    try:
        frame = source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH"],
                start_date="2024-01-02",
                end_date="2024-01-05",
            )
        )
    finally:
        _restore_fake_akshare(original)

    assert len(frame) == 4
    assert client.calls[0]["start_date"] == "20240102"
    assert client.calls[0]["end_date"] == "20240105"

    cached = pd.read_csv(tmp_path / "510300.SH.csv")
    assert cached["trade_date"].tolist() == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    assert cached["symbol"].unique().tolist() == ["510300.SH"]


def test_akshare_source_splits_failed_ranges_and_ignores_proxy_env(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    client = FakeAkShareClient(max_span_days=2)
    original = _install_fake_akshare(client)
    original_proxy = {
        "HTTP_PROXY": os.environ.get("HTTP_PROXY"),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY"),
        "ALL_PROXY": os.environ.get("ALL_PROXY"),
        "NO_PROXY": os.environ.get("NO_PROXY"),
    }
    os.environ["HTTP_PROXY"] = "http://bad-proxy"
    os.environ["HTTPS_PROXY"] = "http://bad-proxy"
    os.environ["ALL_PROXY"] = "socks5://bad-proxy"

    try:
        frame = source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH"],
                start_date="2024-01-02",
                end_date="2024-01-05",
            )
        )
    finally:
        for key, value in original_proxy.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _restore_fake_akshare(original)

    assert len(frame) == 4
    requested_ranges = {(call["start_date"], call["end_date"]) for call in client.calls}
    assert ("20240102", "20240105") in requested_ranges
    assert ("20240102", "20240103") in requested_ranges
    assert ("20240104", "20240105") in requested_ranges
    assert all(call["http_proxy"] is None for call in client.calls)
    assert all(call["https_proxy"] is None for call in client.calls)
    assert all(call["all_proxy"] is None for call in client.calls)
    assert all(call["no_proxy"] == "*" for call in client.calls)


def test_akshare_source_recovers_single_day_with_padded_window(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    client = FakeAkShareClient(fail_exact_days={"20240103"})
    original = _install_fake_akshare(client)

    try:
        frame = source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH"],
                start_date="2024-01-03",
                end_date="2024-01-03",
            )
        )
    finally:
        _restore_fake_akshare(original)

    assert len(frame) == 1
    assert frame.loc[0, "trade_date"] == "2024-01-03"
    requested_ranges = [(call["start_date"], call["end_date"]) for call in client.calls]
    assert ("20240103", "20240103") in requested_ranges
    assert ("20231231", "20240106") in requested_ranges


def test_akshare_source_skips_symbol_after_consecutive_failures(tmp_path: Path) -> None:
    source = _build_source(tmp_path)
    source.MAX_CONSECUTIVE_SINGLE_DAY_FAILURES = 3
    client = FakeAkShareClient(always_fail_symbols={"588080"})
    original = _install_fake_akshare(client)

    try:
        frame = source.load_bars(
            DataLoadRequest(
                symbols=["510300.SH", "588080.SH"],
                start_date="2024-01-02",
                end_date="2024-01-08",
            )
        )
    finally:
        _restore_fake_akshare(original)

    assert frame["symbol"].unique().tolist() == ["510300.SH"]
    failed_symbol_calls = [call for call in client.calls if call["symbol"] == "588080"]
    assert failed_symbol_calls
    assert not (tmp_path / "588080.SH.csv").exists()
    assert (tmp_path / "510300.SH.csv").exists()
