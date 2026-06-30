from datetime import datetime

import numpy as np
import pytest
import torch


def _make_toy_db(tmp_path, n_syms=5, n_days=120):
    """Create a minimal SQLite DB with synthetic OHLCV data."""
    import sqlite3

    import pandas as pd

    db = tmp_path / "toy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE stocks (symbol TEXT, date TEXT, open REAL, high REAL, "
        "low REAL, close REAL, volume REAL, amount REAL)"
    )
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    for sym in [f"SYM{i:04d}.TW" for i in range(n_syms)]:
        price = 100.0
        for d in dates:
            price *= 1 + rng.normal(0, 0.01)
            conn.execute(
                "INSERT INTO stocks VALUES (?,?,?,?,?,?,?,?)",
                (
                    sym,
                    d,
                    price,
                    price * 1.01,
                    price * 0.99,
                    price,
                    1000.0,
                    1000.0 * price,
                ),
            )
    conn.commit()
    conn.close()
    return str(db)


def test_sample_date_batch_shapes(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    batch = sampler.sample_date_batch(n_stocks=3, seed=0)
    assert "x" in batch and "actual_return_h" in batch
    N = batch["x"].shape[0]
    T = batch["x"].shape[1]
    assert N <= 3
    assert batch["x"].ndim == 3 and batch["x"].shape[2] == 6
    assert batch["x"].shape == (N, T, 6)
    assert batch["stamps"].shape == (N, T, 5)
    assert batch["actual_return_h"].shape == (N,)
    assert batch["actual_return_h"].dtype == torch.float32


def test_sample_date_batch_actual_returns_finite(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    for seed in range(5):
        batch = sampler.sample_date_batch(n_stocks=5, seed=seed)
        assert batch["actual_return_h"].dtype == torch.float32
        assert torch.isfinite(batch["actual_return_h"]).all()


def test_different_seeds_give_different_dates(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    dates = {sampler.sample_date_batch(5, seed=i)["date"] for i in range(20)}
    if len(sampler._dates) <= 1:
        assert len(dates) == 1
    else:
        assert len(dates) > 1


def test_sample_date_batch_date_is_valid_date_string(tmp_path):
    from finetune_tw.cross_sectional_dataset import CrossSectionalDateSampler

    db = _make_toy_db(tmp_path, n_syms=5, n_days=120)
    sampler = CrossSectionalDateSampler(
        db_path=db,
        lookback=20,
        horizon=5,
        start_date="2020-01-02",
        end_date="2020-10-01",
        clip=5.0,
        seed=42,
    )
    batch = sampler.sample_date_batch(n_stocks=5, seed=7)
    parsed = datetime.strptime(batch["date"], "%Y-%m-%d")
    assert parsed.strftime("%Y-%m-%d") == batch["date"]
