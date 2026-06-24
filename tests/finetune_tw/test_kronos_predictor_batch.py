import numpy as np
import pandas as pd

from model.kronos import KronosPredictor


_PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def _make_predictor_stub() -> KronosPredictor:
    predictor = KronosPredictor.__new__(KronosPredictor)
    predictor.price_cols = ["open", "high", "low", "close"]
    predictor.vol_col = "volume"
    predictor.amt_vol = "amount"
    predictor.clip = 5
    return predictor


def _make_df(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10.0 + offset, 11.0 + offset, 12.0 + offset],
            "high": [11.0 + offset, 12.0 + offset, 13.0 + offset],
            "low": [9.0 + offset, 10.0 + offset, 11.0 + offset],
            "close": [10.5 + offset, 11.5 + offset, 12.5 + offset],
            "volume": [100.0, 110.0, 120.0],
            "amount": [1000.0, 1100.0, 1200.0],
        }
    )


def test_prepare_batch_inputs_returns_current_normalization():
    predictor = _make_predictor_stub()
    df_list = [_make_df(0.0), _make_df(5.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=3))
    y_ts = pd.Series(pd.bdate_range("2024-01-04", periods=2))

    x_batch, x_stamp_batch, y_stamp_batch, means, stds, y_index_list = predictor.prepare_batch_inputs(
        df_list=df_list,
        x_timestamp_list=[x_ts, x_ts],
        y_timestamp_list=[y_ts, y_ts],
        pred_len=2,
    )

    expected_x0 = df_list[0][_PRICE_COLUMNS].values.astype(np.float32)
    expected_mean0 = expected_x0.mean(axis=0)
    expected_std0 = expected_x0.std(axis=0)
    expected_norm0 = np.clip((expected_x0 - expected_mean0) / (expected_std0 + 1e-5), -5, 5)

    np.testing.assert_allclose(x_batch[0], expected_norm0, rtol=0, atol=0)
    np.testing.assert_allclose(means[0], expected_mean0, rtol=0, atol=0)
    np.testing.assert_allclose(stds[0], expected_std0, rtol=0, atol=0)
    assert x_stamp_batch.shape == (2, 3, 5)
    assert y_stamp_batch.shape == (2, 2, 5)
    assert list(y_index_list[0]) == list(y_ts)


def test_predict_prepared_batch_matches_predict_batch():
    predictor = _make_predictor_stub()
    df_list = [_make_df(0.0), _make_df(5.0)]
    x_ts = pd.Series(pd.bdate_range("2024-01-01", periods=3))
    y_ts = pd.Series(pd.bdate_range("2024-01-04", periods=2))

    generated = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 10.0, 20.0], [1.5, 2.5, 3.5, 4.5, 11.0, 21.0]],
            [[5.0, 6.0, 7.0, 8.0, 30.0, 40.0], [5.5, 6.5, 7.5, 8.5, 31.0, 41.0]],
        ],
        dtype=np.float32,
    )

    def fake_generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose, return_all_samples=False):
        assert pred_len == 2
        assert return_all_samples is False
        return generated

    predictor.generate = fake_generate

    prepared = predictor.prepare_batch_inputs(
        df_list=df_list,
        x_timestamp_list=[x_ts, x_ts],
        y_timestamp_list=[y_ts, y_ts],
        pred_len=2,
    )

    direct = predictor.predict_batch(
        df_list=df_list,
        x_timestamp_list=[x_ts, x_ts],
        y_timestamp_list=[y_ts, y_ts],
        pred_len=2,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )
    prepared_out = predictor.predict_prepared_batch(
        *prepared,
        pred_len=2,
        T=1.0,
        top_k=1,
        top_p=1.0,
        sample_count=1,
        verbose=False,
    )

    assert [df["close"].tolist() for df in direct] == [df["close"].tolist() for df in prepared_out]
    assert [df.index.tolist() for df in direct] == [df.index.tolist() for df in prepared_out]
