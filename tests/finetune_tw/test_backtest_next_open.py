from __future__ import annotations

import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices


def _seed_calendar_db(tmp_path) -> str:
    db_path = str(tmp_path / "calendar.db")
    init_db(db_path)

    benchmark = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1_000.0] * 5,
            "amount": [100_500.0, 101_500.0, 102_500.0, 103_500.0, 104_500.0],
        }
    )
    upsert_prices(db_path, "^TWII", benchmark)
    return db_path


def test_load_trading_calendar_uses_benchmark_dates(tmp_path):
    import finetune_tw.backtest_next_open as bo

    db_path = _seed_calendar_db(tmp_path)
    cfg = Config(
        db_path=db_path,
        benchmark_symbol="^TWII",
        test_start_date="2024-01-01",
    )

    dates = bo._load_trading_calendar(cfg, end="2024-01-31")

    assert list(dates.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]


def test_build_signal_and_execution_dates_drops_last_anchor_without_next_day():
    import finetune_tw.backtest_next_open as bo

    trading_dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"]
    )

    signal_dates, execution_dates = bo._build_signal_and_execution_dates(
        trading_dates,
        hold_days=2,
    )

    assert list(signal_dates.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-05"]
    assert list(execution_dates.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-08"]
