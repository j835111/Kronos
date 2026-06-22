import numpy as np
import pandas as pd
from unittest.mock import patch

from finetune_tw.analog import AnalogEngine


def _make_price_df(
    close_values: list[float], start: str = "2020-01-02"
) -> pd.DataFrame:
    n = len(close_values)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame(
        {
            "date": dates,
            "open": close_values,
            "high": [c * 1.01 for c in close_values],
            "low": [c * 0.99 for c in close_values],
            "close": close_values,
            "volume": [1000.0] * n,
            "amount": [c * 1000 for c in close_values],
        }
    )


def test_analog_engine_fit_and_query():
    close = list(range(100, 200))
    fake_df = _make_price_df(close)
    engine = AnalogEngine(window=10, pred_len=5)

    with patch("finetune_tw.analog.query_symbol", return_value=fake_df), patch(
        "finetune_tw.analog.list_symbols", return_value=["2330.TW"]
    ):
        engine.fit(":memory:", cutoff_date="2020-06-01")

    assert engine._keys.shape[0] > 0
    assert len(engine._matches) == engine._keys.shape[0]

    result = engine.query(np.array(list(range(150, 160)), dtype=float), top_k=5)

    assert result is not None
    assert len(result) == 5
    assert {match["symbol"] for match in result} == {"2330.TW"}
    assert [match["distance"] for match in result] == sorted(
        match["distance"] for match in result
    )


def test_point_in_time_cutoff():
    calls: list[str | None] = []

    def mock_query_symbol(db_path, symbol, start=None, end=None):
        calls.append(end)
        return _make_price_df([100.0] * 20)

    engine = AnalogEngine(window=10, pred_len=5)

    with patch("finetune_tw.analog.query_symbol", side_effect=mock_query_symbol), patch(
        "finetune_tw.analog.list_symbols", return_value=["2330.TW", "2317.TW"]
    ):
        engine.fit(":memory:", cutoff_date="2024-01-10")

    assert calls
    for end_date in calls:
        assert end_date is not None
        assert end_date < "2024-01-10"


def test_featurize_padding():
    engine = AnalogEngine(window=10, pred_len=5)

    feature = engine._featurize(np.array([100.0, 101.0, 102.0], dtype=float))

    assert feature.shape == (9,)
    np.testing.assert_allclose(feature[:7], np.zeros(7))
    assert np.any(feature[-2:] != 0.0)


def test_query_returns_none_unfitted():
    engine = AnalogEngine(window=10, pred_len=5)

    result = engine.query(np.ones(10, dtype=float), top_k=5)

    assert result is None
