from argparse import Namespace

from main.cache_data import _resolve_requested_dates


def test_resolve_requested_dates_prefers_today_flag() -> None:
    args = Namespace(today=True, start_date="2024-01-01", end_date="2024-12-31")

    start_date, end_date = _resolve_requested_dates(
        args,
        timezone="Asia/Shanghai",
        default_start="2023-01-01",
        default_end="2025-12-31",
    )

    assert start_date == end_date
    assert len(start_date) == 10


def test_resolve_requested_dates_uses_explicit_or_default_range() -> None:
    args = Namespace(today=False, start_date=None, end_date="2025-03-18")

    start_date, end_date = _resolve_requested_dates(
        args,
        timezone="Asia/Shanghai",
        default_start="2023-01-01",
        default_end="2025-12-31",
    )

    assert start_date == "2023-01-01"
    assert end_date == "2025-03-18"
