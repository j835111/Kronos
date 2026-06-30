from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ToyTokenizer:
    def encode(self, x, half=True):
        base = x[:, :, 0].round().to(torch.long).remainder(4)
        return base, (base + 1).remainder(4)


class _ToyPredictor(nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.last_stamp = None

    def forward(self, s1_ids, s2_ids, stamp=None):
        del s2_ids
        self.last_stamp = stamp.detach().clone() if stamp is not None else None
        next_ids = (s1_ids + 1).remainder(self.vocab_size)
        logits = F.one_hot(next_ids, num_classes=self.vocab_size).to(torch.float32)
        logits = logits * self.logit_scale
        aux_logits = torch.zeros_like(logits)
        return logits, aux_logits


class _ToyCrossSampler:
    def __init__(self) -> None:
        self.last_batch = None

    def sample_date_batch(self, n_stocks, seed=None):
        del seed
        assert n_stocks == 3
        x = torch.tensor(
            [
                [[0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]],
                [[1.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]],
                [[2.0, 0, 0, 0, 0, 0], [3.0, 0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        )
        stamps = torch.tensor(
            [
                [[0.0, 9.0, 1.0, 2.0, 1.0], [0.0, 9.0, 2.0, 3.0, 1.0], [0.0, 9.0, 3.0, 4.0, 1.0], [0.0, 9.0, 4.0, 5.0, 1.0]],
                [[0.0, 10.0, 1.0, 6.0, 2.0], [0.0, 10.0, 2.0, 7.0, 2.0], [0.0, 10.0, 3.0, 8.0, 2.0], [0.0, 10.0, 4.0, 9.0, 2.0]],
                [[0.0, 11.0, 1.0, 10.0, 3.0], [0.0, 11.0, 2.0, 11.0, 3.0], [0.0, 11.0, 3.0, 12.0, 3.0], [0.0, 11.0, 4.0, 13.0, 3.0]],
            ],
            dtype=torch.float32,
        )
        batch = {
            "x": x,
            "stamps": stamps,
            "actual_return_h": torch.tensor([0.25, -0.10, 0.05], dtype=torch.float32),
            "date": "2024-01-02",
        }
        self.last_batch = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        return batch


class _OracleTokenizer:
    def encode(self, x, half=True):
        del half
        batch_size, seq_len, _ = x.shape
        ids = torch.arange(seq_len, dtype=torch.long).repeat(batch_size, 1)
        return ids, ids


class _OracleDataset:
    def __init__(self) -> None:
        self.lookback_window = 4
        self.window = 6
        self.clip = 5.0
        self._samples = [("AAA", 0)]
        self._data = {
            "AAA": np.array(
                [
                    [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [13.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [14.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [15.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        }


def _make_ranking_components():
    cfg = SimpleNamespace(cross_sectional_batch_size=3)
    device = torch.device("cpu")
    tokenizer = _ToyTokenizer()
    predictor = _ToyPredictor()
    cross_sampler = _ToyCrossSampler()
    oracle_table = torch.tensor([0.3, -0.2, 0.6, 0.1], dtype=torch.float32)
    return cfg, device, tokenizer, predictor, cross_sampler, oracle_table


def test_run_cross_sectional_ranking_step_runs_without_error():
    from finetune_tw.train_predictor import _run_cross_sectional_ranking_step

    cfg, device, tokenizer, predictor, cross_sampler, oracle_table = _make_ranking_components()

    loss = _run_cross_sectional_ranking_step(
        cfg=cfg,
        device=device,
        tokenizer=tokenizer,
        predictor=predictor,
        cross_sampler=cross_sampler,
        oracle_table=oracle_table,
        step_seed=7,
    )

    assert loss is not None
    assert predictor.last_stamp is not None
    assert predictor.last_stamp.shape == (3, 4, 5)
    assert cross_sampler.last_batch is not None
    assert torch.equal(
        predictor.last_stamp,
        cross_sampler.last_batch["stamps"].to(device),
    )


def test_run_cross_sectional_ranking_step_returns_finite_scalar():
    from finetune_tw.train_predictor import _run_cross_sectional_ranking_step

    cfg, device, tokenizer, predictor, cross_sampler, oracle_table = _make_ranking_components()

    loss = _run_cross_sectional_ranking_step(
        cfg=cfg,
        device=device,
        tokenizer=tokenizer,
        predictor=predictor,
        cross_sampler=cross_sampler,
        oracle_table=oracle_table,
        step_seed=11,
    )

    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_run_cross_sectional_ranking_step_backpropagates_to_predictor_params():
    from finetune_tw.train_predictor import _run_cross_sectional_ranking_step

    cfg, device, tokenizer, predictor, cross_sampler, oracle_table = _make_ranking_components()

    loss = _run_cross_sectional_ranking_step(
        cfg=cfg,
        device=device,
        tokenizer=tokenizer,
        predictor=predictor,
        cross_sampler=cross_sampler,
        oracle_table=oracle_table,
        step_seed=13,
    )

    assert loss is not None
    loss.backward()

    assert predictor.logit_scale.grad is not None
    assert torch.isfinite(predictor.logit_scale.grad)


def test_iter_s1_oracle_samples_uses_bar_t_token_for_oracle_key():
    from finetune_tw.train_predictor import _iter_s1_oracle_samples

    dataset = _OracleDataset()
    tokenizer = _OracleTokenizer()

    samples = list(
        _iter_s1_oracle_samples(
            dataset=dataset,
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            batch_size=1,
        )
    )

    assert len(samples) == 1
    assert samples[0]["s1_ids"].shape == (dataset.lookback_window,)
    assert samples[0]["s1_ids"][-1].item() == dataset.lookback_window
