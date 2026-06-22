"""Point-in-time-safe analog retrieval for historical close windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finetune_tw.db import list_symbols, query_symbol


class AnalogEngine:
    """Nearest-neighbor retrieval engine over historical close windows."""

    def __init__(self, window: int = 20, pred_len: int = 10) -> None:
        self.window = window
        self.pred_len = pred_len
        self._keys = np.empty((0, max(self.window - 1, 0)), dtype=float)
        self._matches: list[dict[str, object]] = []

    def fit(
        self,
        db_path: str,
        cutoff_date: str,
        symbols: list[str] | None = None,
    ) -> "AnalogEngine":
        strict_cutoff = (
            pd.Timestamp(cutoff_date) - pd.Timedelta(days=self.pred_len * 2)
        ).strftime("%Y-%m-%d")

        universe = list_symbols(db_path) if symbols is None else list(symbols)
        keys: list[np.ndarray] = []
        matches: list[dict[str, object]] = []

        for symbol in universe:
            df = query_symbol(db_path, symbol, end=strict_cutoff)
            if len(df) < self.window + self.pred_len:
                continue

            close = df["close"].to_numpy(dtype=float)
            dates = df["date"].tolist()
            last_start = len(close) - self.window - self.pred_len + 1

            for start_idx in range(last_start):
                end_idx = start_idx + self.window
                future_idx = end_idx + self.pred_len - 1
                window_close = close[start_idx:end_idx]
                base_close = window_close[-1]
                future_close = close[future_idx]

                keys.append(self._featurize(window_close))
                matches.append(
                    {
                        "symbol": symbol,
                        "end_date": dates[end_idx - 1],
                        "future_return": float(
                            future_close / (base_close + 1e-12) - 1.0
                        ),
                    }
                )

        if keys:
            self._keys = np.vstack(keys)
        else:
            self._keys = np.empty((0, max(self.window - 1, 0)), dtype=float)
        self._matches = matches
        return self

    def query(
        self, close_series: np.ndarray | list[float], top_k: int = 5
    ) -> list[dict[str, object]] | None:
        if self._keys.shape[0] == 0:
            return None

        key = self._featurize(np.asarray(close_series, dtype=float))
        k = min(top_k, self._keys.shape[0])
        if k <= 0:
            return []

        distances = np.linalg.norm(self._keys - key, axis=1)
        indices = np.argpartition(distances, k - 1)[:k]
        ordered = indices[np.argsort(distances[indices])]

        results: list[dict[str, object]] = []
        for idx in ordered:
            match = dict(self._matches[int(idx)])
            match["distance"] = float(distances[int(idx)])
            results.append(match)
        return results

    def _featurize(self, close: np.ndarray | list[float]) -> np.ndarray:
        close_arr = np.asarray(close, dtype=float)
        if close_arr.size < 2:
            log_returns = np.empty(0, dtype=float)
        else:
            safe_close = np.clip(close_arr, 1e-12, None)
            log_returns = np.diff(np.log(safe_close))

        feature_len = max(self.window - 1, 0)
        if log_returns.size < feature_len:
            log_returns = np.pad(
                log_returns, (feature_len - log_returns.size, 0), constant_values=0.0
            )
        elif log_returns.size > feature_len:
            log_returns = log_returns[-feature_len:]

        return log_returns.astype(float, copy=False)
