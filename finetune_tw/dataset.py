from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from finetune_tw.db import query_symbol, list_symbols

FEATURES = ["open", "high", "low", "close", "volume", "amount"]  # 6 features, matches d_in=6


class MultiStockDataset(Dataset):
    """
    Samples (lookback_window + predict_window + 1)-length windows from any stock in the DB.
    Windows are isolated per stock — never cross stock boundaries.
    Returns (x_tensor, x_stamp_tensor) matching CustomKlineDataset's interface.
    """

    def __init__(
        self,
        db_path: str,
        lookback_window: int,
        predict_window: int,
        start_date: str,
        end_date: str,
        clip: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.window = lookback_window + predict_window + 1
        self.lookback_window = lookback_window
        self.clip = clip
        self.seed = seed

        self._data: dict[str, np.ndarray] = {}          # symbol -> (T, 6) float32
        self._stamps: dict[str, np.ndarray] = {}         # symbol -> (T, 5) float32
        self._samples: list[tuple[str, int]] = []        # (symbol, start_row)

        for sym in list_symbols(db_path):
            df = query_symbol(db_path, sym, start=start_date, end=end_date)
            if len(df) < self.window:
                continue
            df = df.reset_index(drop=True)
            arr = df[FEATURES].values.astype(np.float32)
            stamps = _build_stamps(df["date"])
            self._data[sym] = arr
            self._stamps[sym] = stamps
            for i in range(len(arr) - self.window + 1):
                self._samples.append((sym, i))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sym, start = self._samples[idx]
        x = self._data[sym][start : start + self.window].copy()
        s = self._stamps[sym][start : start + self.window].copy()

        past = x[: self.lookback_window]
        mean = past.mean(axis=0)
        std = past.std(axis=0) + 1e-5
        x = np.clip((x - mean) / std, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(s)

    def set_epoch_seed(self, epoch: int) -> None:
        # Provided for compatibility with existing training loop; no-op here
        # because windows are addressed deterministically by index.
        pass


def _build_stamps(dates: pd.Series) -> np.ndarray:
    """Returns (T, 5) array: [minute=0, hour=9, weekday, day, month]."""
    dt = pd.to_datetime(dates)
    stamps = np.stack([
        np.zeros(len(dt), dtype=np.float32),          # minute (fixed 0 for daily)
        np.full(len(dt), 9, dtype=np.float32),         # hour (fixed 9 for market open)
        dt.dt.weekday.values.astype(np.float32),
        dt.dt.day.values.astype(np.float32),
        dt.dt.month.values.astype(np.float32),
    ], axis=1)
    return stamps
