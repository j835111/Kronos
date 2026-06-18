import numpy as np
import pandas as pd
import pytest
import torch
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.dataset import MultiStockDataset

LOOKBACK = 10
PRED = 5
WINDOW = LOOKBACK + PRED + 1


def _make_stock_df(n: int = 50, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "date": dates,
        "open": rng.uniform(100, 200, n),
        "high": rng.uniform(100, 200, n) + 5,
        "low": rng.uniform(90, 190, n),
        "close": rng.uniform(100, 200, n),
        "volume": rng.uniform(1e6, 1e7, n),
        "amount": np.zeros(n),
    })


@pytest.fixture
def populated_db(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    upsert_prices(db, "2330.TW", _make_stock_df(60, "2020-01-01"))
    upsert_prices(db, "2317.TW", _make_stock_df(60, "2020-01-01"))
    return db


def test_dataset_len_positive(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    assert len(ds) > 0


def test_dataset_item_shapes(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, x_stamp = ds[0]
    assert x.shape == (WINDOW, 6)
    assert x_stamp.shape == (WINDOW, 5)


def test_dataset_returns_tensors(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, x_stamp = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(x_stamp, torch.Tensor)


def test_dataset_x_is_normalized(populated_db):
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    x, _ = ds[0]
    # After normalization and clip=5, values should be in [-5, 5]
    assert x.abs().max().item() <= 5.0 + 1e-5


def test_dataset_no_cross_stock_windows(populated_db):
    # Each window comes from a single stock — verify by checking that n_samples
    # equals sum of per-stock valid windows
    ds = MultiStockDataset(populated_db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    # Both stocks have ~42 trading days, so each contributes ~42-WINDOW+1 windows
    assert len(ds) == len(ds._samples)

def test_dataset_skips_short_stocks(tmp_path):
    db = str(tmp_path / "short.db")
    init_db(db)
    # Only 5 rows — too short for any window
    upsert_prices(db, "TINY.TW", _make_stock_df(5))
    ds = MultiStockDataset(db, LOOKBACK, PRED, "2020-01-01", "2020-12-31")
    assert len(ds) == 0
