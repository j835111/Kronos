from types import SimpleNamespace

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
        return {
            "x": x,
            "stamps": torch.zeros((3, 4, 5), dtype=torch.float32),
            "actual_return_h": torch.tensor([0.25, -0.10, 0.05], dtype=torch.float32),
            "date": "2024-01-02",
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
    assert torch.allclose(predictor.last_stamp[:, :, 1], torch.full((3, 4), 9.0))


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
