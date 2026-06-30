"""Cross-sectional date sampler for auxiliary ranking loss computation.

Randomly samples a trading date from the training window, loads all available
stock contexts ending on that date (lookback rows), applies the same z-score
normalization as MultiStockDataset, and returns both the normalized context
tensors and the realized open-to-open return at horizon h.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import torch

from finetune_tw.dataset import _build_stamps
from finetune_tw.db import list_symbols, query_symbol

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


class CrossSectionalDateSampler:
    """Provides cross-sectional batches (all stocks on one date) for ranking loss.

    Parameters
    ----------
    db_path : str
    lookback : int
        Number of context rows per sample (must match training lookback_window).
    horizon : int
        h in open[T+h+1]/open[T+1]-1.
    start_date : str
        Earliest possible signal date (inclusive).
    end_date : str
        Latest possible signal date (inclusive).
    clip : float
        Clipping for z-score normalization (matches training cfg.clip).
    seed : int
        RNG seed for reproducible date sampling.
    benchmark_symbol : str
        Excluded from universe.
    """

    def __init__(
        self,
        db_path: str,
        lookback: int,
        horizon: int,
        start_date: str,
        end_date: str,
        clip: float = 5.0,
        seed: int = 42,
        benchmark_symbol: str = "^TWII",
    ) -> None:
        self.db_path = db_path
        self.lookback = lookback
        self.horizon = horizon
        self.clip = clip
        self.benchmark_symbol = benchmark_symbol
        self._dates: list[str] = (
            pd.bdate_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
        )
        self._rng = np.random.default_rng(seed)
        self._symbols = [
            sym for sym in self._list_symbols() if sym != benchmark_symbol
        ]

    def _list_symbols(self) -> list[str]:
        try:
            symbols = sorted(set(list_symbols(self.db_path)))
            if symbols:
                return symbols
        except Exception:
            pass

        with sqlite3.connect(self.db_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "stocks" in tables:
                rows = conn.execute(
                    "SELECT DISTINCT symbol FROM stocks ORDER BY symbol"
                ).fetchall()
                return [row[0] for row in rows]
            if "daily_prices" in tables:
                rows = conn.execute(
                    "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"
                ).fetchall()
                return [row[0] for row in rows]
        return []

    def _query_symbol_window(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        try:
            return query_symbol(self.db_path, symbol, start=start, end=end)
        except Exception:
            pass

        q = "SELECT date,open,high,low,close,volume,amount FROM stocks WHERE symbol=?"
        params: list[object] = [symbol]
        if start:
            q += " AND date>=?"
            params.append(start)
        if end:
            q += " AND date<=?"
            params.append(end)
        q += " ORDER BY date"
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(q, conn, params=params)

    def sample_date_batch(
        self,
        n_stocks: int,
        seed: int | None = None,
    ) -> dict:
        """Sample a random date and return up to n_stocks cross-sectional contexts.

        Returns a dict with keys:
          "x"               : Tensor[N, lookback, 6]  normalized context
          "stamps"          : Tensor[N, lookback, 5]
          "actual_return_h" : Tensor[N]  realized open[T+h+1]/open[T+1]-1
          "date"            : str        the sampled date
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        date_str = str(rng.choice(self._dates))
        lookback_start = (
            pd.Timestamp(date_str) - pd.Timedelta(days=self.lookback * 3)
        ).strftime("%Y-%m-%d")
        future_end = (
            pd.Timestamp(date_str) + pd.Timedelta(days=(self.horizon + 1) * 3)
        ).strftime("%Y-%m-%d")

        sym_order = list(self._symbols)
        rng.shuffle(sym_order)

        xs: list[torch.Tensor] = []
        stamps: list[torch.Tensor] = []
        actual_rets: list[float] = []
        for sym in sym_order:
            if len(xs) >= n_stocks:
                break

            df = self._query_symbol_window(sym, start=lookback_start, end=future_end)
            if df.empty:
                continue

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            signal_mask = df["date"] == pd.Timestamp(date_str)
            if not signal_mask.any():
                continue
            signal_idx = int(signal_mask.idxmax())
            if signal_idx < self.lookback - 1:
                continue

            ctx = df.iloc[signal_idx - self.lookback + 1 : signal_idx + 1].reset_index(drop=True)
            future = df.iloc[signal_idx + 1 : signal_idx + self.horizon + 2].reset_index(drop=True)
            if len(ctx) != self.lookback or len(future) < self.horizon + 1:
                continue

            arr = ctx[FEATURES].values.astype(np.float32)
            if not np.isfinite(arr).all():
                continue

            open_t1 = float(future.iloc[0]["open"])
            open_th1 = float(future.iloc[self.horizon]["open"])
            if open_t1 <= 0 or not np.isfinite(open_th1):
                continue

            realized_ret = open_th1 / open_t1 - 1.0
            if not np.isfinite(realized_ret):
                continue

            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-5
            arr_norm = np.clip((arr - mean) / std, -self.clip, self.clip)
            stamp_arr = _build_stamps(ctx["date"])

            xs.append(torch.from_numpy(arr_norm))
            stamps.append(torch.from_numpy(stamp_arr))
            actual_rets.append(realized_ret)

        if not xs:
            return {
                "x": torch.zeros((0, self.lookback, len(FEATURES)), dtype=torch.float32),
                "stamps": torch.zeros((0, self.lookback, 5), dtype=torch.float32),
                "actual_return_h": torch.zeros((0,), dtype=torch.float32),
                "date": date_str,
            }

        return {
            "x": torch.stack(xs),
            "stamps": torch.stack(stamps),
            "actual_return_h": torch.tensor(actual_rets, dtype=torch.float32),
            "date": date_str,
        }
