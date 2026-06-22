from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from finetune_tw.analog import AnalogFeatures
from finetune_tw.config import Config
from finetune_tw.db import init_db, upsert_prices
from finetune_tw.stacking import FEATURE_COLS, StackingModel
from finetune_tw.walkforward import WalkForwardFold


def _make_price_frame(
    dates: pd.DatetimeIndex,
    base: float,
    slope: float,
    phase: float,
) -> pd.DataFrame:
    idx = np.arange(len(dates), dtype=float)
    close = base + slope * idx + 1.5 * np.sin(idx / 7.0 + phase)
    volume = 1_000.0 + 30.0 * np.cos(idx / 5.0 + phase)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "amount": close * volume,
        }
    )


def _build_config(tmp_path, db_path: str) -> Config:
    return Config(
        db_path=db_path,
        output_dir=str(tmp_path),
        exp_name="stacking-backtest-test",
        lookback_window=60,
        pred_len=5,
        top_k=2,
        hold_days=5,
        mc_sample_count=4,
        stacking_enabled=True,
        stacking_train_start="2023-04-03",
        stacking_train_end="2023-08-31",
        test_start_date="2023-09-01",
        benchmark_symbol="^TWII",
        analog_enabled=False,
        analog_n_neighbors=4,
        analog_window=20,
        wf_embargo_days=5,
    )


def _seed_synthetic_db(tmp_path) -> tuple[str, list[str]]:
    db_path = str(tmp_path / "synthetic.db")
    init_db(db_path)
    dates = pd.bdate_range("2023-01-02", periods=240)
    symbols = ["1101.TW", "1216.TW", "1301.TW", "2330.TW"]

    for idx, symbol in enumerate(symbols):
        frame = _make_price_frame(
            dates=dates,
            base=80.0 + idx * 15.0,
            slope=0.18 + idx * 0.03,
            phase=idx * 0.4,
        )
        upsert_prices(db_path, symbol, frame)

    benchmark = _make_price_frame(dates=dates, base=1_000.0, slope=0.12, phase=0.2)
    upsert_prices(db_path, "^TWII", benchmark)
    return db_path, symbols


class _FakeExtractor:
    def __init__(self, predictor, n_samples: int, top_k: int) -> None:
        del predictor, n_samples, top_k

    def extract_date(self, date: pd.Timestamp, symbols: list[str], cfg, horizon: int = 4):
        del cfg, horizon
        import finetune_tw.signal as signal_mod

        signals = {}
        for idx, symbol in enumerate(symbols):
            mean = 0.01 * (idx + 1) + (date.day % 7) * 0.001
            signals[symbol] = signal_mod.KronosSignal(
                mean_return=mean,
                q10=mean - 0.01,
                q50=mean,
                q90=mean + 0.01,
                dispersion=0.002 * (idx + 1),
                dir_prob=min(0.55 + 0.05 * idx, 0.95),
            )
        return signals


class _FakeAnalogEngine:
    window = 20

    def query(self, recent_close, recent_volume):
        del recent_close, recent_volume
        return AnalogFeatures(
            fwd_q25=0.01,
            fwd_q50=0.02,
            fwd_q75=0.03,
            up_prob=0.7,
            max_gain=0.08,
            max_loss=-0.04,
            dispersion=0.015,
            n_analogs=4,
        )


def test_collect_oof_features_returns_expected_columns(tmp_path, monkeypatch):
    import finetune_tw.stacking_backtest as sb

    db_path, symbols = _seed_synthetic_db(tmp_path)
    cfg = _build_config(tmp_path, db_path)
    monkeypatch.setattr(sb, "KronosSignalExtractor", _FakeExtractor)

    dates = pd.bdate_range("2023-07-03", periods=3, freq="B")
    features = sb._collect_oof_features(cfg, object(), None, symbols, dates)

    assert isinstance(features, pd.DataFrame)
    assert set(FEATURE_COLS).issubset(features.columns)
    assert "fwd_return" in features.columns
    assert features.index.names == ["date", "symbol"]
    assert not features.empty


def test_collect_oof_features_with_analog_engine_populates_analog_columns(
    tmp_path,
    monkeypatch,
):
    import finetune_tw.stacking_backtest as sb

    db_path, symbols = _seed_synthetic_db(tmp_path)
    cfg = _build_config(tmp_path, db_path)
    monkeypatch.setattr(sb, "KronosSignalExtractor", _FakeExtractor)

    dates = pd.bdate_range("2023-07-10", periods=2, freq="B")
    features = sb._collect_oof_features(
        cfg,
        object(),
        _FakeAnalogEngine(),
        symbols,
        dates,
    )

    assert not features.empty
    assert (features["analog_q50"] != 0.0).all()
    assert (features["analog_up_prob"] > 0.0).all()


def test_run_stacking_backtest_returns_expected_keys_and_saves_artifacts(
    tmp_path,
    monkeypatch,
):
    import finetune_tw.stacking_backtest as sb

    db_path, symbols = _seed_synthetic_db(tmp_path)
    cfg = _build_config(tmp_path, db_path)
    cfg.model_key = "round0"

    fold = WalkForwardFold(
        train_start="2023-04-03",
        train_end="2023-06-30",
        embargo_end="2023-07-10",
        val_start="2023-07-10",
        val_end="2023-08-31",
    )

    monkeypatch.setattr(sb, "KronosSignalExtractor", _FakeExtractor)
    monkeypatch.setattr(sb, "build_model_specs", lambda cfg: {"round0": SimpleNamespace(label="Round 0")})
    monkeypatch.setattr(sb, "load_predictor_from_spec", lambda spec, cfg: object())
    monkeypatch.setattr(sb, "oof_folds", lambda *args, **kwargs: [fold])
    monkeypatch.setattr(sb, "_today", lambda: pd.Timestamp("2023-10-31"))

    result = sb.run_stacking_backtest(cfg)

    out_dir = tmp_path / cfg.exp_name
    assert set(["model_key", "model_label", "test_start", "test_end", "stacker", "kronos_only", "benchmark"]).issubset(result)
    assert (out_dir / "stacking_model.lgb").exists()
    assert (out_dir / "stacking_features_oof.parquet").exists()
    assert (out_dir / "backtest_stacking.json").exists()
    assert (out_dir / "backtest_stacking.png").exists()
    assert result["stacker"]["metrics"]["max_drawdown"] >= 0.0
    assert len(result["stacker"]["daily_returns"]) == len(result["stacker"]["dates"])


def test_stacking_model_save_load_round_trip_from_oof_features(tmp_path, monkeypatch):
    import finetune_tw.stacking_backtest as sb

    db_path, symbols = _seed_synthetic_db(tmp_path)
    cfg = _build_config(tmp_path, db_path)
    monkeypatch.setattr(sb, "KronosSignalExtractor", _FakeExtractor)

    dates = pd.bdate_range("2023-07-03", periods=6, freq="B")
    features = sb._collect_oof_features(cfg, object(), None, symbols, dates)

    model = StackingModel(num_rounds=15)
    model.fit(features)

    model_path = tmp_path / "round-trip.lgb"
    model.save(str(model_path))
    loaded = StackingModel.load(str(model_path))

    base_frame = features.drop(columns=["fwd_return"])
    original_scores = model.predict(base_frame)
    loaded_scores = loaded.predict(base_frame)

    pd.testing.assert_series_equal(original_scores, loaded_scores, check_names=False)
