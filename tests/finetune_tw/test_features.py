import numpy as np
import pandas as pd
import pytest
from finetune_tw.features import build_tech_features, build_market_relative_features


def _make_df(close_values: list[float], start: str = "2023-01-02") -> pd.DataFrame:
    n = len(close_values)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "date": dates,
        "open": close_values,
        "high": [c * 1.01 for c in close_values],
        "low": [c * 0.99 for c in close_values],
        "close": close_values,
        "volume": [1000.0] * n,
        "amount": [c * 1000 for c in close_values],
    })


def test_tech_features_keys():
    close = list(range(100, 165))  # 65 rows
    df = _make_df(close)
    as_of = pd.Timestamp("2023-04-14")
    result = build_tech_features(df, as_of)
    assert result is not None
    expected_keys = {"ma20_gap", "ma60_gap", "rsi_14", "bb_pct", "mom_10d", "mom_20d", "vol_20d"}
    assert set(result.keys()) == expected_keys


def test_tech_features_insufficient_data():
    df = _make_df([100.0] * 30)  # only 30 rows, need 60
    result = build_tech_features(df, pd.Timestamp("2023-02-16"))
    assert result is None


def test_tech_features_rsi_range():
    close = [100 + i for i in range(65)]  # trending up
    df = _make_df(close)
    result = build_tech_features(df, pd.Timestamp("2023-04-14"))
    assert 0 <= result["rsi_14"] <= 100


def test_tech_features_ma20_gap_sign():
    close = [100.0] * 20 + [150.0] * 30 + [200.0] * 15  # recent step-up keeps last close above MA20
    df = _make_df(close)
    as_of = pd.Timestamp(df["date"].iloc[-1])
    result = build_tech_features(df, as_of)
    assert result["ma20_gap"] > 0  # last close > MA20


def test_market_relative_features_keys():
    sym_close = [100.0 + i for i in range(65)]
    bench_close = [1000.0 + i for i in range(65)]
    sym_df = _make_df(sym_close)
    bench_df = _make_df(bench_close)
    as_of = pd.Timestamp(sym_df["date"].iloc[-1])
    result = build_market_relative_features(sym_df, bench_df, as_of)
    assert result is not None
    assert set(result.keys()) == {"alpha_20d", "alpha_60d", "rel_vol"}


def test_market_relative_features_alpha_positive():
    # sym goes up more than bench
    sym_close = [100.0 * (1.01 ** i) for i in range(65)]
    bench_close = [100.0] * 65  # flat
    sym_df = _make_df(sym_close)
    bench_df = _make_df(bench_close)
    as_of = pd.Timestamp(sym_df["date"].iloc[-1])
    result = build_market_relative_features(sym_df, bench_df, as_of)
    assert result["alpha_20d"] > 0
    assert result["alpha_60d"] > 0
