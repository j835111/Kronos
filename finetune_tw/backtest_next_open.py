from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch

from finetune_tw.backtest import (
    build_model_specs,
    compute_metrics,
    compute_raw_signals,
    load_predictor_from_spec,
    rank_stocks,
    signals_to_holdings,
)
from finetune_tw.config import Config
from finetune_tw.db import list_symbols, query_symbol


def _load_trading_calendar(cfg: Config, end: str) -> pd.DatetimeIndex:
    bm_df = query_symbol(
        cfg.db_path,
        cfg.benchmark_symbol,
        start=cfg.test_start_date,
        end=end,
    )
    if bm_df.empty:
        raise ValueError(
            f"No benchmark rows found for {cfg.benchmark_symbol} between "
            f"{cfg.test_start_date} and {end}."
        )
    return pd.DatetimeIndex(pd.to_datetime(bm_df["date"]))


def _build_signal_and_execution_dates(
    trading_dates: pd.DatetimeIndex,
    hold_days: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if hold_days <= 0:
        raise ValueError(f"hold_days must be positive, got {hold_days}")

    signal_dates = trading_dates[::hold_days]
    kept_signal_dates: list[pd.Timestamp] = []
    execution_dates: list[pd.Timestamp] = []

    for signal_date in signal_dates:
        idx = trading_dates.get_loc(signal_date)
        if idx + 1 >= len(trading_dates):
            continue
        kept_signal_dates.append(signal_date)
        execution_dates.append(trading_dates[idx + 1])

    return pd.DatetimeIndex(kept_signal_dates), pd.DatetimeIndex(execution_dates)
