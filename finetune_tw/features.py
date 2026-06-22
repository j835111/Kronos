"""Technical and market-relative feature builders.

All functions are pure (no side effects, no DB access) — input is a DataFrame
with columns: date, open, high, low, close, volume, amount. All rows up to
and including as_of are used; later rows are ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _close_tail(df: pd.DataFrame, as_of: pd.Timestamp, n: int) -> "np.ndarray | None":
    sub = df[pd.to_datetime(df["date"]) <= as_of].tail(n)
    if len(sub) < n:
        return None
    return sub["close"].values.astype(float)


def _returns(close: np.ndarray) -> np.ndarray:
    return np.diff(close) / (close[:-1] + 1e-9)


def build_tech_features(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
) -> "dict[str, float] | None":
    """Compute technical indicators as of a given date.

    Requires at least 60 rows ending at or before as_of.
    Returns None if insufficient data.
    """
    sub = df[pd.to_datetime(df["date"]) <= as_of].tail(lookback)
    if len(sub) < 60:
        return None
    close = sub["close"].values.astype(float)

    last = close[-1]

    # MA gaps
    ma20 = close[-20:].mean()
    ma60 = close[-60:].mean()
    ma20_gap = (last - ma20) / (ma20 + 1e-9)
    ma60_gap = (last - ma60) / (ma60 + 1e-9)

    # RSI-14 (Wilder's, simplified as SMA-based)
    rets_14 = _returns(close[-15:])
    gains = np.where(rets_14 > 0, rets_14, 0.0)
    losses = np.where(rets_14 < 0, -rets_14, 0.0)
    rs = gains.mean() / (losses.mean() + 1e-9)
    rsi_14 = 100.0 - 100.0 / (1.0 + rs)

    # Bollinger %B (20-day, 2σ)
    c20 = close[-20:]
    bb_mid = c20.mean()
    bb_std = c20.std()
    bb_pct = (last - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # Momentum
    mom_10d = (last / (close[-11] + 1e-9) - 1.0) if len(close) >= 11 else float("nan")
    mom_20d = (last / (close[-21] + 1e-9) - 1.0) if len(close) >= 21 else float("nan")

    # 20-day volatility
    vol_20d = float(np.std(_returns(close[-21:]))) if len(close) >= 21 else float("nan")

    return {
        "ma20_gap": float(ma20_gap),
        "ma60_gap": float(ma60_gap),
        "rsi_14": float(rsi_14),
        "bb_pct": float(bb_pct),
        "mom_10d": float(mom_10d),
        "mom_20d": float(mom_20d),
        "vol_20d": float(vol_20d),
    }


def build_market_relative_features(
    sym_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 90,
) -> "dict[str, float] | None":
    """Compute alpha and relative volatility vs a benchmark.

    Returns None if either series has fewer than 61 rows.
    """
    sym60 = _close_tail(sym_df, as_of, 61)
    bench60 = _close_tail(bench_df, as_of, 61)
    if sym60 is None or bench60 is None:
        return None

    sym_ret_20 = sym60[-1] / (sym60[-21] + 1e-9) - 1.0
    sym_ret_60 = sym60[-1] / (sym60[0] + 1e-9) - 1.0
    bench_ret_20 = bench60[-1] / (bench60[-21] + 1e-9) - 1.0
    bench_ret_60 = bench60[-1] / (bench60[0] + 1e-9) - 1.0

    sym_vol = float(np.std(_returns(sym60[-21:])))
    bench_vol = float(np.std(_returns(bench60[-21:])))

    return {
        "alpha_20d": float(sym_ret_20 - bench_ret_20),
        "alpha_60d": float(sym_ret_60 - bench_ret_60),
        "rel_vol": float(sym_vol / (bench_vol + 1e-9)),
    }
