# Backtest In-Memory Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不新增任何磁碟 cache 檔、不改 CLI 介面、不改 JSON/PNG 輸出 schema 的前提下，加速 `finetune_tw.backtest` 與 `finetune_tw.backtest_next_open`。

**Architecture:** 新增一個共享的 `finetune_tw.backtest_data` 模組，集中處理「一次 bulk 從 SQLite 載入多檔歷史資料」與「從記憶體切出單次 rebalance 的 lookback context」。兩個 backtest script 保留原本的 ranking、metrics、ATR 權重與輸出流程，只把最慢的 `rebal_date × symbol` 單筆查詢改成單次 bulk 載入後的 in-memory slicing。

**Tech Stack:** Python, pandas, SQLite, PyTorch, pytest

---

## File Map

- Create: `finetune_tw/backtest_data.py`
  - 共享 bulk history 載入、欄位裁切、rebalance batch 準備邏輯。
- Create: `tests/finetune_tw/test_backtest_data.py`
  - 驗證 bulk 載入後的資料結構與 rebalance batch 準備邏輯。
- Modify: `finetune_tw/backtest.py:35-36,257-308,419-440`
  - `compute_raw_signals()` 改成先 bulk 載入歷史資料再切 batch。
  - `run_backtest()` 的 close price preload 改成單次 bulk 載入。
- Modify: `tests/finetune_tw/test_backtest.py`
  - 新增 `compute_raw_signals()` 只 bulk 載入一次的回歸測試。
- Modify: `finetune_tw/backtest_next_open.py:27,70-84,87-151,491-516`
  - `_load_price_frames()` 改成包 shared loader。
  - `compute_raw_signals_open()` 改成 bulk 載入歷史資料再切 batch。
- Modify: `tests/finetune_tw/test_backtest_next_open.py`
  - 新增 `compute_raw_signals_open()` bulk 載入一次且 ATR metadata 不變的測試。

## Non-Goals

- 不新增 `raw_preds_*.json` 或其他新 cache 檔。
- 不修改回測排序、報酬計算、ATR 權重或 benchmark 定義。
- 不碰 `model/kronos.py` 的生成邏輯。
- 不改 `grid_search_backtest.py`；它會自然吃到 `backtest.py` 的加速結果。

### Task 1: 建立共享的 in-memory 歷史資料 helper

**Files:**
- Create: `finetune_tw/backtest_data.py`
- Create: `tests/finetune_tw/test_backtest_data.py`

- [ ] **Step 1: 寫失敗測試，先鎖定 shared helper 的輸入輸出**

```python
import pandas as pd

from finetune_tw.db import init_db, upsert_prices


def _make_symbol_df(offset: float) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [10.0 + offset, 11.0 + offset, 12.0 + offset, 13.0 + offset, 14.0 + offset],
            "high": [10.5 + offset, 11.5 + offset, 12.5 + offset, 13.5 + offset, 14.5 + offset],
            "low": [9.5 + offset, 10.5 + offset, 11.5 + offset, 12.5 + offset, 13.5 + offset],
            "close": [10.2 + offset, 11.2 + offset, 12.2 + offset, 13.2 + offset, 14.2 + offset],
            "volume": [100.0, 101.0, 102.0, 103.0, 104.0],
            "amount": [1000.0, 1111.0, 1222.0, 1333.0, 1444.0],
        }
    )


def test_load_symbol_history_frames_groups_rows_by_symbol(tmp_path):
    from finetune_tw.backtest_data import load_symbol_history_frames

    db = str(tmp_path / "history.db")
    init_db(db)
    upsert_prices(db, "1101.TW", _make_symbol_df(0.0))
    upsert_prices(db, "1216.TW", _make_symbol_df(20.0))

    frames = load_symbol_history_frames(
        db_path=db,
        symbols=["1101.TW", "1216.TW"],
        start="2024-01-02",
        end="2024-01-05",
    )

    assert sorted(frames) == ["1101.TW", "1216.TW"]
    assert list(frames["1101.TW"].columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert list(frames["1101.TW"].index.strftime("%Y-%m-%d")) == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_build_rebalance_inputs_uses_preloaded_frames_and_skips_short_or_nan_contexts():
    from finetune_tw.backtest_data import build_rebalance_inputs

    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    history_frames = {
        "GOOD": pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0],
                "high": [10.5, 11.5, 12.5, 13.5],
                "low": [9.5, 10.5, 11.5, 12.5],
                "close": [10.2, 11.2, 12.2, 13.2],
                "volume": [100.0, 101.0, 102.0, 103.0],
                "amount": [1000.0, 1111.0, 1222.0, 1333.0],
            },
            index=dates,
        ),
        "SHORT": pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 11.0],
                "amount": [12.0, 24.0],
            },
            index=dates[:2],
        ),
        "NAN": pd.DataFrame(
            {
                "open": [5.0, 6.0, 7.0],
                "high": [5.5, 6.5, 7.5],
                "low": [4.5, 5.5, 6.5],
                "close": [5.2, float("nan"), 7.2],
                "volume": [50.0, 51.0, 52.0],
                "amount": [260.0, 306.0, 374.4],
            },
            index=dates[:3],
        ),
    }

    syms, dfs, xts, yts = build_rebalance_inputs(
        history_frames=history_frames,
        symbols=["GOOD", "SHORT", "NAN"],
        rebal_date=pd.Timestamp("2024-01-03"),
        lookback_window=3,
        pred_len=2,
    )

    assert syms == ["GOOD"]
    assert list(dfs[0].columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert list(xts[0].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert list(yts[0].dt.strftime("%Y-%m-%d")) == ["2024-01-03", "2024-01-04"]
```

- [ ] **Step 2: 跑測試，確認目前真的失敗**

Run: `python3 -m pytest tests/finetune_tw/test_backtest_data.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'finetune_tw.backtest_data'`

- [ ] **Step 3: 寫最小實作，提供 bulk 載入與 rebalance batch 準備**

```python
from __future__ import annotations

import pandas as pd

from finetune_tw.db import query_symbols_window

_OHLCVA_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def load_symbol_history_frames(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    window = query_symbols_window(db_path, symbols, start=start, end=end)
    if window.empty:
        return {}

    window = window.copy()
    window["date"] = pd.to_datetime(window["date"])

    frames: dict[str, pd.DataFrame] = {}
    for sym, group in window.groupby("symbol", sort=False):
        frames[sym] = (
            group.loc[:, ["date", *_OHLCVA_COLUMNS]]
            .set_index("date")
            .sort_index()
        )
    return frames


def load_price_field_series(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
    field: str,
) -> dict[str, pd.Series]:
    frames = load_symbol_history_frames(db_path, symbols, start=start, end=end)
    return {
        sym: frame[field].copy()
        for sym, frame in frames.items()
        if field in frame.columns
    }


def load_price_frame_fields(
    db_path: str,
    symbols: list[str],
    start: str,
    end: str,
    fields: list[str],
) -> dict[str, pd.DataFrame]:
    frames = load_symbol_history_frames(db_path, symbols, start=start, end=end)
    wanted = list(fields)
    return {
        sym: frame.loc[:, wanted].copy()
        for sym, frame in frames.items()
        if set(wanted).issubset(frame.columns)
    }


def build_rebalance_inputs(
    history_frames: dict[str, pd.DataFrame],
    symbols: list[str],
    rebal_date: pd.Timestamp,
    lookback_window: int,
    pred_len: int,
) -> tuple[list[str], list[pd.DataFrame], list[pd.Series], list[pd.Series]]:
    y_ts = pd.Series(pd.date_range(rebal_date, periods=pred_len, freq="B"))

    batch_syms: list[str] = []
    batch_dfs: list[pd.DataFrame] = []
    batch_xts: list[pd.Series] = []
    batch_yts: list[pd.Series] = []

    for sym in symbols:
        frame = history_frames.get(sym)
        if frame is None or frame.empty:
            continue

        idx = frame.index.searchsorted(pd.Timestamp(rebal_date), side="right")
        if idx < lookback_window:
            continue

        ctx = frame.iloc[idx - lookback_window:idx]
        ctx_df = ctx.loc[:, _OHLCVA_COLUMNS].reset_index(drop=True)
        if ctx_df.isnull().any().any():
            continue

        batch_syms.append(sym)
        batch_dfs.append(ctx_df)
        batch_xts.append(pd.Series(ctx.index))
        batch_yts.append(y_ts)

    return batch_syms, batch_dfs, batch_xts, batch_yts
```

- [ ] **Step 4: 重跑測試，確認 shared helper 綠燈**

Run: `python3 -m pytest tests/finetune_tw/test_backtest_data.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/backtest_data.py tests/finetune_tw/test_backtest_data.py
git commit -m "feat(finetune_tw): add shared backtest data loaders"
```

### Task 2: `finetune_tw.backtest` 改成 bulk 載入歷史資料與 close prices

**Files:**
- Modify: `finetune_tw/backtest.py:35-36,257-308,419-440`
- Modify: `tests/finetune_tw/test_backtest.py`

- [ ] **Step 1: 先加失敗測試，鎖定 `compute_raw_signals()` 應只 bulk 載入一次**

```python
from types import SimpleNamespace

import pandas as pd
import pytest


class _FakeBatchPredictor:
    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        **kwargs,
    ):
        assert pred_len == 2
        assert len(df_list) == 1
        return [
            pd.DataFrame(
                {
                    "open": [10.0, 10.0],
                    "high": [10.5, 10.5],
                    "low": [9.5, 9.5],
                    "close": [11.0, 12.0],
                    "volume": [1.0, 1.0],
                    "amount": [1.0, 1.0],
                }
            )
        ]


def test_compute_raw_signals_preloads_history_once(monkeypatch):
    import finetune_tw.backtest as bt

    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    history_frames = {
        "1101.TW": pd.DataFrame(
            {
                "open": [8.0, 9.0, 10.0],
                "high": [8.5, 9.5, 10.5],
                "low": [7.5, 8.5, 9.5],
                "close": [8.0, 9.0, 10.0],
                "volume": [100.0, 101.0, 102.0],
                "amount": [800.0, 909.0, 1020.0],
            },
            index=dates,
        ),
        "SHORT.TW": pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.0, 2.0],
                "volume": [10.0, 11.0],
                "amount": [10.0, 22.0],
            },
            index=dates[:2],
        ),
    }

    calls = {"count": 0}

    def fake_load_symbol_history_frames(db_path, symbols, start, end):
        calls["count"] += 1
        assert symbols == ["1101.TW", "SHORT.TW"]
        return history_frames

    monkeypatch.setattr(bt, "load_symbol_history_frames", fake_load_symbol_history_frames)

    raw = bt.compute_raw_signals(
        predictor=_FakeBatchPredictor(),
        cfg=SimpleNamespace(db_path="ignored.db", lookback_window=3),
        rebal_dates=pd.DatetimeIndex(["2024-01-03"]),
        pred_len=2,
        symbols=["1101.TW", "SHORT.TW"],
    )

    assert calls["count"] == 1
    assert list(raw["2024-01-03"]) == ["1101.TW"]
    assert raw["2024-01-03"]["1101.TW"].tolist() == pytest.approx([0.1, 0.2])
```

- [ ] **Step 2: 跑單測，確認它先失敗**

Run: `python3 -m pytest tests/finetune_tw/test_backtest.py::test_compute_raw_signals_preloads_history_once -v`

Expected: FAIL with `assert 0 == 1` because `compute_raw_signals()` 還沒呼叫 shared bulk loader

- [ ] **Step 3: 改 `backtest.py`，把 signal phase 與 close price preload 換成 shared helper**

```python
from finetune_tw.backtest_data import (
    build_rebalance_inputs,
    load_price_field_series,
    load_symbol_history_frames,
)


def compute_raw_signals(
    predictor: KronosPredictor,
    cfg: Config,
    rebal_dates: pd.DatetimeIndex,
    pred_len: int,
    symbols: list[str],
) -> dict[str, dict[str, pd.Series]]:
    """raw_preds[date_str][sym] = Series of predicted close returns, iloc[h-1] = h-day return."""
    if len(rebal_dates) == 0:
        return {}

    BATCH_SIZE = 64
    raw_preds: dict[str, dict[str, pd.Series]] = {}
    lookback_start = (
        rebal_dates.min() - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    history_frames = load_symbol_history_frames(
        cfg.db_path,
        symbols,
        start=lookback_start,
        end=rebal_dates.max().strftime("%Y-%m-%d"),
    )

    for i, rebal_date in enumerate(rebal_dates):
        rebal_str = rebal_date.strftime("%Y-%m-%d")
        batch_syms, batch_dfs, batch_xts, batch_yts = build_rebalance_inputs(
            history_frames=history_frames,
            symbols=symbols,
            rebal_date=rebal_date,
            lookback_window=cfg.lookback_window,
            pred_len=pred_len,
        )

        date_preds: dict[str, pd.Series] = {}
        with torch.no_grad():
            for b in range(0, len(batch_syms), BATCH_SIZE):
                preds = predictor.predict_batch(
                    df_list=batch_dfs[b:b + BATCH_SIZE],
                    x_timestamp_list=batch_xts[b:b + BATCH_SIZE],
                    y_timestamp_list=batch_yts[b:b + BATCH_SIZE],
                    pred_len=pred_len,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
                for sym, pred, ctx_df in zip(
                    batch_syms[b:b + BATCH_SIZE],
                    preds,
                    batch_dfs[b:b + BATCH_SIZE],
                ):
                    if pred is not None and len(pred) >= pred_len:
                        last_close = ctx_df["close"].iloc[-1]
                        date_preds[sym] = pred["close"].reset_index(drop=True) / last_close - 1

        raw_preds[rebal_str] = date_preds
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(rebal_dates)}] {rebal_str}: {len(date_preds)} signals")
            sys.stdout.flush()

    return raw_preds


def run_backtest(cfg: Config, model_key: str, hold_days_list: list[int]) -> Path:
    # ... keep existing spec / date setup ...
    close_prices = load_price_field_series(
        cfg.db_path,
        symbols,
        start=cfg.test_start_date,
        end=test_end,
        field="close",
    )
    print(f"Loaded close prices: {len(close_prices)} symbols")
    sys.stdout.flush()

    bm_df = query_symbol(
        cfg.db_path,
        cfg.benchmark_symbol,
        start=cfg.test_start_date,
        end=test_end,
    )
    # ... keep the rest unchanged ...
```

- [ ] **Step 4: 重跑單測，再跑 backtest 相關測試**

Run: `python3 -m pytest tests/finetune_tw/test_backtest.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/backtest.py tests/finetune_tw/test_backtest.py
git commit -m "refactor(finetune_tw): bulk-load backtest signal histories"
```

### Task 3: `finetune_tw.backtest_next_open` 改成同一套 bulk loader

**Files:**
- Modify: `finetune_tw/backtest_next_open.py:27,70-84,87-151,491-516`
- Modify: `tests/finetune_tw/test_backtest_next_open.py`

- [ ] **Step 1: 先寫失敗測試，鎖定 next-open signal phase 只 bulk 載入一次且保留 ATR metadata**

```python
class _FakeOpenBatchPredictor:
    def predict_batch(
        self,
        df_list,
        x_timestamp_list,
        y_timestamp_list,
        pred_len,
        **kwargs,
    ):
        assert pred_len == 3
        assert len(df_list) == 1
        return [
            pd.DataFrame(
                {
                    "open": [10.0, 11.0, 12.0],
                    "high": [10.5, 11.5, 12.5],
                    "low": [9.5, 10.5, 11.5],
                    "close": [10.2, 11.2, 12.2],
                    "volume": [1.0, 1.0, 1.0],
                    "amount": [1.0, 1.0, 1.0],
                }
            )
        ]


def test_compute_raw_signals_open_preloads_history_once_and_keeps_pred_frame(monkeypatch):
    import finetune_tw.backtest_next_open as bo

    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    history_frames = {
        "1101.TW": pd.DataFrame(
            {
                "open": [8.0, 9.0, 10.0],
                "high": [8.5, 9.5, 10.5],
                "low": [7.5, 8.5, 9.5],
                "close": [8.0, 9.0, 10.0],
                "volume": [100.0, 101.0, 102.0],
                "amount": [800.0, 909.0, 1020.0],
            },
            index=dates,
        ),
        "SHORT.TW": pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.0, 2.0],
                "volume": [10.0, 11.0],
                "amount": [10.0, 22.0],
            },
            index=dates[:2],
        ),
    }

    calls = {"count": 0}

    def fake_load_symbol_history_frames(db_path, symbols, start, end):
        calls["count"] += 1
        assert symbols == ["1101.TW", "SHORT.TW"]
        return history_frames

    monkeypatch.setattr(bo, "load_symbol_history_frames", fake_load_symbol_history_frames)

    raw = bo.compute_raw_signals_open(
        predictor=_FakeOpenBatchPredictor(),
        cfg=Config(db_path="ignored.db", lookback_window=3),
        rebal_dates=pd.DatetimeIndex(["2024-01-03"]),
        pred_len=3,
        symbols=["1101.TW", "SHORT.TW"],
        attach_pred_frame=True,
    )

    assert calls["count"] == 1
    assert list(raw["2024-01-03"]) == ["1101.TW"]
    assert raw["2024-01-03"]["1101.TW"].tolist() == pytest.approx([0.1, 0.2])
    assert list(raw["2024-01-03"]["1101.TW"].attrs["pred_frame"].columns) == [
        "high",
        "low",
        "close",
    ]
```

- [ ] **Step 2: 跑單測，確認目前先紅燈**

Run: `python3 -m pytest tests/finetune_tw/test_backtest_next_open.py::test_compute_raw_signals_open_preloads_history_once_and_keeps_pred_frame -v`

Expected: FAIL with `assert 0 == 1` because `compute_raw_signals_open()` 還在逐檔查 DB

- [ ] **Step 3: 改 `backtest_next_open.py`，重用 shared helper**

```python
from finetune_tw.backtest_data import (
    build_rebalance_inputs,
    load_price_frame_fields,
    load_symbol_history_frames,
)


def _load_price_frames(
    cfg: Config,
    symbols: list[str],
    end: str,
) -> dict[str, pd.DataFrame]:
    return load_price_frame_fields(
        cfg.db_path,
        symbols,
        start=cfg.test_start_date,
        end=end,
        fields=["open", "close"],
    )


def compute_raw_signals_open(
    predictor,
    cfg: Config,
    rebal_dates: pd.DatetimeIndex,
    pred_len: int,
    symbols: list[str],
    attach_pred_frame: bool = False,
) -> dict[str, dict[str, pd.Series]]:
    if len(rebal_dates) == 0:
        return {}

    BATCH_SIZE = 64
    raw_preds: dict[str, dict[str, pd.Series]] = {}
    lookback_start = (
        rebal_dates.min() - pd.Timedelta(days=cfg.lookback_window * 2)
    ).strftime("%Y-%m-%d")
    history_frames = load_symbol_history_frames(
        cfg.db_path,
        symbols,
        start=lookback_start,
        end=rebal_dates.max().strftime("%Y-%m-%d"),
    )

    for i, rebal_date in enumerate(rebal_dates):
        rebal_str = rebal_date.strftime("%Y-%m-%d")
        batch_syms, batch_dfs, batch_xts, batch_yts = build_rebalance_inputs(
            history_frames=history_frames,
            symbols=symbols,
            rebal_date=rebal_date,
            lookback_window=cfg.lookback_window,
            pred_len=pred_len,
        )

        date_preds: dict[str, pd.Series] = {}
        with torch.no_grad():
            for b in range(0, len(batch_syms), BATCH_SIZE):
                preds = predictor.predict_batch(
                    df_list=batch_dfs[b:b + BATCH_SIZE],
                    x_timestamp_list=batch_xts[b:b + BATCH_SIZE],
                    y_timestamp_list=batch_yts[b:b + BATCH_SIZE],
                    pred_len=pred_len,
                    T=1.0,
                    top_k=1,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
                for sym, pred in zip(batch_syms[b:b + BATCH_SIZE], preds):
                    if pred is not None and len(pred) >= pred_len:
                        pred_opens = pred["open"].reset_index(drop=True)
                        returns = pred_opens.iloc[1:].reset_index(drop=True) / pred_opens.iloc[0] - 1
                        if attach_pred_frame:
                            returns.attrs["pred_frame"] = pred.loc[:, ["high", "low", "close"]].reset_index(drop=True)
                        date_preds[sym] = returns

        raw_preds[rebal_str] = date_preds
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(rebal_dates)}] {rebal_str}: {len(date_preds)} signals")
            sys.stdout.flush()

    return raw_preds
```

- [ ] **Step 4: 跑 next-open 測試檔，確認整體回歸綠燈**

Run: `python3 -m pytest tests/finetune_tw/test_backtest_next_open.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add finetune_tw/backtest_next_open.py tests/finetune_tw/test_backtest_next_open.py
git commit -m "refactor(finetune_tw): bulk-load next-open backtest data"
```

## Final Verification

- [ ] Run: `python3 -m pytest tests/finetune_tw/test_backtest_data.py tests/finetune_tw/test_backtest.py tests/finetune_tw/test_backtest_next_open.py -v`
  - Expected: targeted backtest-related tests all PASS

- [ ] Run:

```bash
python3 - <<'PY'
import time
import pandas as pd

from finetune_tw.config import Config
from finetune_tw.db import list_symbols
from finetune_tw.backtest import compute_raw_signals


class DummyPredictor:
    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kwargs):
        import numpy as np
        import pandas as pd

        base = pd.DataFrame(
            {
                "open": np.linspace(100, 100 + pred_len - 1, pred_len),
                "high": np.linspace(101, 101 + pred_len - 1, pred_len),
                "low": np.linspace(99, 99 + pred_len - 1, pred_len),
                "close": np.linspace(100, 100 + pred_len - 1, pred_len),
                "volume": np.ones(pred_len),
                "amount": np.ones(pred_len),
            }
        )
        return [base.copy() for _ in df_list]


cfg = Config.from_yaml("finetune_tw/configs/config_tw_daily.yaml")
symbols = [s for s in list_symbols(cfg.db_path) if s != cfg.benchmark_symbol]
rebal_dates = pd.bdate_range(cfg.test_start_date, pd.Timestamp.today().date())[::cfg.hold_days][:5]

t0 = time.perf_counter()
compute_raw_signals(DummyPredictor(), cfg, rebal_dates, cfg.pred_len, symbols)
print(f"elapsed={time.perf_counter() - t0:.3f}s")
PY
```

  - Expected: 同樣 5 個 rebalance dates 的 smoke benchmark，時間明顯低於本次調查記錄的舊版基線 `51.8s`；不應再看到 `rebal_date × symbol` 級別的 SQLite 重複查詢成本

- [ ] Run: `python3 -m finetune_tw.backtest --config finetune_tw/configs/config_tw_daily.yaml --model round0 --hold_days_list 5`
  - Expected: 印出 signal 計數、回測 metrics，並寫出 `finetune_tw/outputs/tw_daily/backtest_returns_round0.json` 與對應 PNG

- [ ] Run: `python3 -m finetune_tw.backtest_next_open --config finetune_tw/configs/config_tw_daily.yaml --model round0 --hold_days_list 5`
  - Expected: 印出 signal 計數、回測 metrics，並寫出 `finetune_tw/outputs/tw_daily/backtest_returns_round0_next_open.json` 與對應 PNG
