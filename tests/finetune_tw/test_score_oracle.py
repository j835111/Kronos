import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
import pandas as pd


def _make_fake_tokenizer(s1_bits=4, s2_bits=4):
    tok = MagicMock()
    tok.s1_bits = s1_bits
    v_s1 = 2 ** s1_bits
    v_s2 = 2 ** s2_bits

    def fake_encode(x, half=False):
        batch_size, time_steps, _ = x.shape
        s1 = torch.randint(0, v_s1, (batch_size, time_steps))
        s2 = torch.randint(0, v_s2, (batch_size, time_steps))
        return s1, s2

    tok.encode = fake_encode
    return tok, v_s1


def _make_tuple_samples(n_samples=200, lookback=10, pred_len=6):
    total_steps = lookback + pred_len + 1
    samples = []
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        opens = 100 * np.cumprod(1 + rng.normal(0, 0.01, total_steps))
        x = np.zeros((total_steps, 6), dtype=np.float32)
        x[:, 0] = opens.astype(np.float32)
        x[:, 3] = opens.astype(np.float32)
        samples.append((torch.from_numpy(x), torch.zeros(total_steps, 5)))
    return samples


def _make_dict_samples():
    return [
        {
            "s1_ids": [0, 1, 2, 3],
            "open_prices": [100.0, 101.0, 102.0, 105.0, 110.0, 115.0],
        },
        {
            "s1_ids": [4, 5, 2, 3],
            "open_prices": [90.0, 92.0, 94.0, 96.0, 99.0, 102.0],
        },
        {
            "s1_ids": [7, 8, 9, 1],
            "open_prices": [80.0, 81.0, 82.0, 83.0, 84.0, 90.0],
        },
    ]


def test_build_s1_oracle_from_tuple_samples_shape_and_type():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    tok, v_s1 = _make_fake_tokenizer(s1_bits=4)
    dataset = _make_tuple_samples(n_samples=300, lookback=10, pred_len=6)

    oracle = build_s1_oracle_from_samples(tok, dataset, lookback=10, horizon=5, min_count=5)

    assert oracle.shape == (v_s1,)
    assert oracle.dtype == torch.float32
    assert torch.isfinite(oracle).all()


def test_build_s1_oracle_from_raw_dict_samples_uses_mean_return():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    tok, _ = _make_fake_tokenizer(s1_bits=4)
    samples = _make_dict_samples()

    oracle = build_s1_oracle_from_samples(tok, samples, lookback=3, horizon=2, min_count=2)

    expected = (((115.0 / 105.0) - 1.0) + ((102.0 / 96.0) - 1.0)) / 2.0
    assert oracle[2].item() == pytest.approx(expected, abs=1e-7)
    assert oracle[1].item() == 0.0


def test_build_s1_oracle_signature_accepts_raw_samples_in_db_path_slot():
    from finetune_tw.score_oracle import build_s1_oracle

    tok, _ = _make_fake_tokenizer(s1_bits=4)
    samples = _make_dict_samples()

    oracle = build_s1_oracle(
        tok,
        samples,
        start="2020-01-01",
        end="2020-12-31",
        lookback=3,
        predict_window=3,
        horizon=2,
        clip=5.0,
        seed=42,
        min_count=2,
    )

    assert oracle.shape == (16,)
    assert oracle[2].item() != 0.0


def test_oracle_pred_score_differentiable():
    from finetune_tw.score_oracle import oracle_pred_score

    oracle = torch.randn(16)
    s1_logits = torch.randn(8, 16, requires_grad=True)

    scores = oracle_pred_score(s1_logits, oracle)

    assert scores.shape == (8,)
    scores.sum().backward()
    assert s1_logits.grad is not None
    assert not torch.all(s1_logits.grad == 0)


def test_oracle_tokens_with_few_samples_get_zero():
    from finetune_tw.score_oracle import build_s1_oracle_from_samples

    tok, _ = _make_fake_tokenizer(s1_bits=4)
    dataset = _make_tuple_samples(n_samples=1, lookback=10, pred_len=6)

    oracle = build_s1_oracle_from_samples(tok, dataset, lookback=10, horizon=5, min_count=20)

    assert oracle.abs().sum().item() == 0.0


def test_build_s1_oracle_from_db_path_uses_exact_signature(tmp_path):
    from finetune_tw.db import init_db, upsert_prices
    from finetune_tw.score_oracle import build_s1_oracle

    db_path = tmp_path / "oracle.db"
    init_db(str(db_path))
    dates = pd.bdate_range("2024-01-01", periods=8)
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
            "volume": [1000.0] * 8,
            "amount": [100000.0] * 8,
        }
    )
    upsert_prices(str(db_path), "AAA", frame)

    tok = MagicMock()
    tok.s1_bits = 4

    def fake_encode(x, half=False):
        _, time_steps, _ = x.shape
        s1 = torch.arange(time_steps).unsqueeze(0)
        s2 = torch.zeros((1, time_steps), dtype=torch.long)
        return s1, s2

    tok.encode = fake_encode

    oracle = build_s1_oracle(
        tok,
        str(db_path),
        start="2024-01-01",
        end="2024-01-31",
        lookback=3,
        predict_window=3,
        horizon=2,
        clip=5.0,
        seed=42,
        min_count=2,
    )

    expected = (((105.0 / 103.0) - 1.0) + ((106.0 / 104.0) - 1.0)) / 2.0
    assert oracle[2].item() == pytest.approx(expected, abs=1e-7)
